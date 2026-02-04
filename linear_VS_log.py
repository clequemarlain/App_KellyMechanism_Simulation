import numpy as np
import torch
from src.game.utils import *
#from src.game.config import SIMULATION_CONFIG as cfg
from scipy import optimize


from scipy.optimize import root_scalar

from collections import defaultdict



def run_main_gamma_curvature(cfg, GameKelly, lrMethod_fixed="DAQ_F", n_fixed=None):
    """
    Returns:
      out[gamma][alpha] = {"rho_mean":..., "rho_std":..., "SW_T_mean":...}
    """
    T   = cfg["T"]
    eta = cfg["eta"]
    a   = cfg["a"]
    a_min = cfg["a_min"]
    c   = cfg["c"]
    mu  = cfg["mu"]
    eps0 = cfg["epsilon"]
    delta = cfg["delta"]
    tol = cfg["tol"]
    price = cfg["price"]

    Nb_random_sim = cfg["Nb_random_sim"]
    list_gamma = cfg["list_gamma"]

    if n_fixed is None:
        n_fixed = cfg["list_n"][0]  # choose first by default

    alphas = [1.0, 0.0]  # linear vs log

    out = {}

    for gamma in list_gamma:
        # fixed n, vary heterogeneity only
        n = int(n_fixed)

        a_vector = torch.tensor([max(a - i * gamma, a_min) for i in range(n)], dtype=torch.float64)
        c_vector = torch.tensor([max(c - i * mu, eps0) for i in range(n)], dtype=torch.float64)

        # your d construction (kept as-is)
        dmin = a_vector * torch.log((eps0 + torch.sum(c_vector) - c_vector + delta) / eps0)
        d_vector = 0.7 * dmin * 0

        out[gamma] = {}

        for alpha in alphas:
            cfg_local = dict(cfg)
            cfg_local["alpha"] = alpha
            eps = eps0* torch.ones(n)

            s_min = (n - 1) * eps + cfg["delta"]
            s_max = (n - 1) * c + cfg["delta"]
            bid0 = eps * torch.rand(1)
            z_max = BR_alpha_fair(eps, c_vector, bid0, s_min, a_vector, delta, alpha, price, b=0)
            x_max = z_max / (z_max + s_min)

            # z_min = BR_alpha_fair(eps, c_vector, bid0, s_max, a_vector, delta, alpha, price, b=0)
            x_min = eps / (eps + s_max)
            x_min_2 = c_vector / (c_vector + s_max)

            Payoff_min = torch.min(Payoff(x_min, eps0, a_vector, d_vector, alpha, price),
                                   Payoff(x_min_2, c_vector, a_vector, d_vector, alpha, price))
            Payoff_max = torch.max(Payoff(x_max, z_max, a_vector, d_vector, alpha, price))

            # Compute SW(z^opt) for THIS alpha and THIS gamma
            bid0 = (c - eps0) * torch.rand(n, dtype=torch.float64) + a
            x_log_optimum = x_log_opt(c_vector, a_vector, d_vector, eps0* torch.ones(1), delta, price, bid0, alpha)
            Valuation_log_opt = Valuation(x_log_optimum, a_vector, d_vector, alpha)
            SW_opt = (torch.sum(Valuation_log_opt))
            SW_opt = SW_opt.detach()
            rhos = []
            SW_Ts = []
            print(SW_opt)

            for sim in range(Nb_random_sim):
                bid0 = (c - eps0) * torch.rand(n, dtype=torch.float64) + a

                game = GameKelly(n, price, eps0* torch.ones(1), delta, alpha, tol,payoff_min=Payoff_min, payoff_max=Payoff_max)
                Bids, Welfare, Utility_set, error_NE_set = game.learning(
                    lrMethod_fixed, a_vector, c_vector, d_vector, T, eta, bid0,
                    vary=cfg["lr_vary"], Hybrid_funcs=None, Hybrid_sets=None, stop=True
                )

                # Here you used Welfare[0] as "SocialWelfare" (a time series or a scalar?)
                # In your old code you do Welfare[0] directly, so I assume it's time-series.
                SW_series = Welfare[0]
                SW_T = SW_series[-1] #if hasattr(SW_series, "__len__") else SW_series
                print(SW_T)

                rho_T = torch.abs((SW_T - SW_opt) / (SW_opt + 1e-12))
                rhos.append(rho_T.item())
                SW_Ts.append(SW_T.item())# if hasattr(SW_T, "item") else float(SW_T))

            out[gamma][alpha] = {
                "rho_mean": float(np.mean(rhos)),
                "rho_std":  float(np.std(rhos)),
                "SW_T_mean": float(np.mean(SW_Ts)),
                "SW_opt": float(SW_opt.item() if hasattr(SW_opt, "item") else float(SW_opt)),
            }

    return out




