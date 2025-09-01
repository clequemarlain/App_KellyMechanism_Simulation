import os
import numpy as np
import torch

from utils import (
    GameKelly, V_func, Utility, LSW_func,
    plotGame, plotGame_dim_N
)
from config import SIMULATION_CONFIG as config

def x_log_opt(c_vector, a_vector, d_vector, eps, delta, price, bid0, alpha):
    # Compute optimal x for alpha=1 (log utility) via KKT bisection.
    # Re-implement here to avoid external deps.
    def min_fraction(eps, budgets, delta):
        return eps / (eps + torch.sum(budgets) - budgets + delta)

    def max_fraction_LSW(d_vector, c_vector, a_vector):
        return torch.exp((c_vector - d_vector) / a_vector)

    eps_x = min_fraction(eps, c_vector, delta)
    C = torch.sum(c_vector)
    S = C / (C + delta)

    gamma_x = max_fraction_LSW(d_vector, c_vector, a_vector)

    def total_x(lmb):
        lmb = torch.clamp(lmb, min=1e-10)
        x_candidate = a_vector / lmb
        x_candidate = torch.maximum(x_candidate, eps_x)
        x_candidate = torch.minimum(x_candidate, gamma_x)
        return torch.sum(x_candidate) - S

    lmbda_min = torch.min(a_vector / gamma_x)
    lmbda_max = torch.max(a_vector / eps_x)

    if total_x(lmbda_min) <= 0:
        return gamma_x
    if total_x(lmbda_max) > 0:
        return eps_x

    # Bisection on lambda
    lo = lmbda_min.item()
    hi = lmbda_max.item()
    for _ in range(100):
        mid = 0.5*(lo+hi)
        val = total_x(torch.tensor(mid))
        if val > 0:
            lo = mid
        else:
            hi = mid
    lmbda = torch.tensor(0.5*(lo+hi))
    x_star = torch.clamp(a_vector / lmbda, min=eps_x, max=gamma_x)
    return x_star

def main():
    # Ensure results dir
    os.makedirs("results", exist_ok=True)

    # Extract config
    T = config["T"]
    n = config["n"]
    eta = config["eta"]
    price = config["price"]
    a = config["a"]
    mu = config["mu"]
    c = config["c"]
    delta = config["delta"]
    epsilon = torch.tensor(config["epsilon"], dtype=torch.float64)
    Hybrid_funcs = config["Hybrid_funcs"]
    metric = config["metric"]
    lr_vary = config["lr_vary"]
    x_label = config["x_label"]
    y_label = config["y_label"]
    ylog_scale = config["ylog_scale"]
    pltText = config["pltText"]
    plot_step = config["plot_step"]
    lrMethods = list(config["lrMethods"])
    alpha = config["alpha"]
    gamma = config["gamma"]
    tol = config["tol"]
    saveFileName = config["saveFileName"] + metric + f"_alpha{alpha}_gamma{gamma}_n_{n}"

    x_data = np.arange(T)
    bid0 = torch.ones(n, dtype=torch.float64)
    a_vector = torch.tensor([a / (i + 1) ** gamma for i in range(n)], dtype=torch.float64)
    c_vector = torch.tensor([c / (i + 1) ** mu for i in range(n)], dtype=torch.float64)
    dmin = a_vector * torch.log((epsilon + torch.sum(c_vector) - c_vector + delta) / epsilon)
    d_vector = 0.7 * dmin

    y_data_speed, y_data_lsw, y_data_bid = [], [], []
    y_data_avg_bid, y_data_utility, y_data_sw = [], [], []

    set1 = torch.arange(n, dtype=torch.long)
    nb_hybrid = max(1, len(Hybrid_funcs))
    Hybrid_sets = torch.chunk(set1, nb_hybrid)

    # Optional "optimal" curve for alpha=1
    if alpha == 1:
        x_log_optimum = x_log_opt(c_vector, a_vector, d_vector, epsilon, delta, price, bid0, alpha)
        utilities_log_opt = a_vector * torch.log(x_log_optimum) + d_vector
    else:
        x_log_optimum = None
        utilities_log_opt = None

    for lrMethod in lrMethods:
        game_set = GameKelly(n, price, epsilon, delta, alpha, tol)
        Bids, Utilities, error_NE_set = game_set.learning(
            lrMethod, a_vector, c_vector, d_vector, T, eta, bid0, vary=lr_vary,
            Hybrid_funcs=Hybrid_funcs, Hybrid_sets=Hybrid_sets
        )
        SocialWelfare = torch.sum(Utilities, dim=1)
        LSW = torch.sum(torch.minimum(Utilities, c_vector.unsqueeze(0)), dim=1)
        Avg_bids = torch.mean(Bids, dim=1)

        y_data_speed.append(error_NE_set.detach().numpy())
        y_data_lsw.append(LSW.detach().numpy())
        y_data_sw.append(SocialWelfare.detach().numpy())
        y_data_bid.append(Bids.detach().numpy())
        y_data_avg_bid.append(Avg_bids.detach().numpy())
        y_data_utility.append(Utilities.detach().numpy())

        nb_iter = torch.argmin(error_NE_set) if torch.min(error_NE_set) <= tol else torch.tensor(float("inf"))
        print(f"{lrMethod} equilibrium:\n {Bids[-1]},\n Nbre Iteration: {nb_iter} err: {error_NE_set[-1]}")

    if metric == "speed":
        pdf = plotGame(x_data, y_data_speed, x_label, y_label, lrMethods, saveFileName=saveFileName, ylog_scale=ylog_scale, pltText=pltText, step=plot_step)
        print("Saved:", pdf)
    if metric == "lsw":
        if alpha == 1 and utilities_log_opt is not None:
            LSWs_opt = (torch.sum(torch.minimum(utilities_log_opt, c_vector))).detach().numpy()
            y_data_lsw.append(LSWs_opt * np.ones_like(y_data_lsw[0], dtype=float))
            lrMethods.append("Optimal")
        pdf = plotGame(x_data, y_data_lsw, x_label, y_label, lrMethods, saveFileName=saveFileName, ylog_scale=ylog_scale, pltText=pltText, step=plot_step)
        print("Saved:", pdf)
    if metric == "sw":
        if alpha == 1 and utilities_log_opt is not None:
            sw = (torch.sum(utilities_log_opt)).detach().numpy()
            y_data_sw.append(sw * np.ones_like(y_data_sw[0], dtype=float))
            lrMethods.append("Optimal")
        pdf = plotGame(x_data, y_data_sw, x_label, y_label, lrMethods, saveFileName=saveFileName, ylog_scale=ylog_scale, pltText=pltText, step=plot_step)
        print("Saved:", pdf)
    if metric == "bid":
        pdf = plotGame_dim_N(x_data, y_data_bid, x_label, y_label, lrMethods, saveFileName=saveFileName, ylog_scale=ylog_scale, pltText=pltText, step=plot_step)
        print("Saved:", pdf)
    if metric == "utility":
        pdf = plotGame_dim_N(x_data, y_data_utility, x_label, y_label, lrMethods, saveFileName=saveFileName, ylog_scale=ylog_scale, pltText=pltText, step=plot_step)
        print("Saved:", pdf)

if __name__ == "__main__":
    main()