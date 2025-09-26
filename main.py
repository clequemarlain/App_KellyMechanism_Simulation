import os
import numpy as np

import torch, random, time
import streamlit as st

from src.game.utils import *
from src.game.config import SIMULATION_CONFIG as cfg

class SimulationRunner:
    def __init__(self, config):
        self.config = config
        self.results = {}

    def run_simulation(self):
        # Extraire les paramètres
        T = self.config["T"]
        n = self.config["n"]
        eta = self.config["eta"]
        price = self.config["price"]
        a = self.config["a"]
        a_min = self.config["a_min"]
        mu = self.config["mu"]
        Nb_random_sim = self.config["Nb_random_sim"]
        c = self.config["c"]
        delta = self.config["delta"]
        epsilon = self.config["epsilon"]
        Hybrid_funcs = self.config["Hybrid_funcs"]
        Hybrid_sets = self.config["Hybrid_sets"]
        lr_vary = self.config["lr_vary"]
        alpha = self.config["alpha"]
        gamma = self.config["gamma"]
        tol = self.config["tol"]
        lrMethods = self.config["lrMethods"]


        #print(f"Hybrid_sets:{Hybrid_sets}")
        eps = epsilon * torch.ones(1)
        z_sol_equ = solve_quadratic(self.config["n"],  self.config["a"],  self.config["delta"])
        var_init = self.config["var_init"]
        bid0 = (2*var_init) * torch.rand(n) + z_sol_equ - var_init

        c_min = epsilon

        a_vector = torch.tensor(self.config["a_vector"], dtype=torch.float64)#torch.tensor([max(a - i * gamma, a_min) for i in range(n)], dtype=torch.float64)
        c_vector = torch.tensor([max(c -i * mu, c_min) for i in range(n)], dtype=torch.float64)
        dmin = a_vector * torch.log((epsilon + torch.sum(c_vector) - c_vector + delta) / epsilon)
        d_vector = 0.7 * dmin*0

        # Compute optimum
        x_log_optimum = x_log_opt(c_vector, a_vector, d_vector, eps, delta, price, bid0)
        Valuation_log_opt = Valuation(x_log_optimum, a_vector, d_vector, alpha)
        SW_opt = (torch.sum(Valuation_log_opt))
        LSW_opt = torch.sum(torch.minimum(Valuation_log_opt, c_vector)).detach().numpy()

        self.results = {
            'methods': {},
            'optimal': {
                'LSW': LSW_opt,
                'SW': SW_opt,
                'x_opt': x_log_optimum.detach().numpy()
            }
        }

        # Réinitialiser les placeholders
        progress_bar = st.progress(0)
        status_text = st.empty()
        game_set = GameKelly(n, price, eps, delta, alpha, tol)
        Bids_Opt, Welfare_Opt, Utility_set_Opt, error_NE_set_Opt = game_set.learning(
            "SBRD", a_vector, c_vector, d_vector, T, eta, bid0, stop=True,
        )
        for i in range(self.config["Nb_random_sim"]):
            if not self.config["keep_initial_bid"]:
                #bid0 = (c - epsilon) * torch.rand(n) + epsilon
                var_init = self.config["var_init"]
                bid0 = (2 * var_init) * torch.rand(n) + z_sol_equ - var_init
            idx = 0
            NbHybrid = 0

            for  lrMethod in lrMethods:
                lrMethod2 = lrMethod
                Hybrid_funcs, Hybrid_sets = [], []

                if lrMethod == "Hybrid":
                    NbHybrid = NbHybrid+1


                    subset = random.sample(range(self.config["n"]), int(self.config["Nb_A1"][idx]))
                    remaining = [i for i in range(self.config["n"]) if i not in subset]
                    if self.config["Random_set"] and NbHybrid == 1:
                        Hybrid_sets = [subset, remaining]
                    else:
                        Hybrid_sets = self.config["Hybrids"][idx]["Hybrid_sets"]
                    Hybrid_funcs = self.config["Hybrids"][idx]["Hybrid_funcs"]
                    lrMethod2 = f"(A1: {self.config['Nb_A1'][idx]}, A2: {n - self.config['Nb_A1'][idx]})"
                    idx += 1


                game_set = GameKelly(n, price, eps, delta, alpha, tol)
                Bids, Welfare, Utility_set, error_NE_set = game_set.learning(
                    lrMethod, a_vector, c_vector, d_vector, T, eta, bid0,
                    vary=lr_vary, Hybrid_funcs=Hybrid_funcs, Hybrid_sets=Hybrid_sets
                )

                SocialWelfare = Welfare[0]
                Distance2optSW = 1 / n * torch.abs(SW_opt - Welfare[0])
                LSW = Welfare[1]
                Relative_Efficienty_Loss = (SocialWelfare - SW_opt)/SW_opt

                # --- Prepare one simulation result ---
                sim_result = {
                    'Speed': error_NE_set.detach().numpy(),
                    'LSW': LSW.detach().numpy(),
                    'SW': SocialWelfare.detach().numpy(),
                    'Dist_To_Optimum_SW': Distance2optSW.detach().numpy(),
                    'Relative_Efficienty_Loss': Relative_Efficienty_Loss.detach().numpy(),
                    'Bid': Bids[0].detach().numpy(),
                    'SBRD_Opt_Bid': Bids_Opt[0][-1].detach().numpy(),
                    'Avg_Bid': Bids[1].detach().numpy(),
                    'SBRD_Opt_Avg_Bid': Bids_Opt[1].detach().numpy(),
                    'Utility': Utility_set[0].detach().numpy(),
                    'SBRD_Opt_Utility': Utility_set_Opt[0][-1].detach().numpy(),
                    'Avg_Utility': Utility_set[1].detach().numpy(),
                    'Res_Utility': Utility_set[2].detach().numpy(),
                    'final_bids': Bids[0][-1].detach().numpy(),
                    'convergence_iter': torch.argmin(error_NE_set).item() if torch.min(error_NE_set) <= tol else T
                }

                # --- Accumulate for averaging ---
                if lrMethod2 not in self.results["methods"]:
                    # Initialize lists to store multiple runs
                    self.results["methods"][lrMethod2] = {k: [v] for k, v in sim_result.items()}
                else:
                    for k, v in sim_result.items():
                        self.results["methods"][lrMethod2][k].append(v)
            # Mise à jour progression
            progress = (i + 1) / self.config["Nb_random_sim"]
            progress_bar.progress(progress)

            # Mise à jour compteur i/N
            status_text.text(f"Simulation {i + 1}/{self.config["Nb_random_sim"]}")
        # --- After loop: compute averages ---
        for method, metrics in self.results["methods"].items():
            for k, v_list in metrics.items():
                try:
                    self.results["methods"][method][k] = np.mean(np.stack(v_list), axis=0)
                except Exception:
                    # For scalars (like convergence_iter or final_bids)
                    self.results["methods"][method][k] = np.mean(v_list)
        return self.results


if  __name__ == "__main__":
    runner = SimulationRunner(cfg)
    results = runner.run_simulation()

    # Préparation des données pour les graphiques
    x_data = np.arange(cfg['T'])
    y_data = []
    legends = []
    LEGENDS = cfg["lrMethods"]
    for method in LEGENDS:

        # if method == "Hybrid":
        if method in results['methods']:
            if cfg["metric"] == "Speed":
                y_data.append(results['methods'][method]['Speed'])
            elif cfg["metric"] == "LSW":
                y_data.append(results['methods'][method]['LSW'])
            elif cfg["metric"] == "Dist_To_Optimum_SW":
                y_data.append(results['methods'][method]['Dist_To_Optimum_SW'])
            elif cfg["metric"] == "SW":
                y_data.append(results['methods'][method]['SW'])
            elif cfg["metric"] == "Bid":
                y_data.append(results['methods'][method]['Bid'])
            elif cfg["metric"] == "Avg_Bid":
                y_data.append(results['methods'][method]['Avg_Bid'])
            elif cfg["metric"] == "Utility":
                y_data.append(results['methods'][method]['Utility'])
            elif cfg["metric"] == "Avg_Utility":
                y_data.append(results['methods'][method]['Avg_Utility'])
            legends.append(method)

    # Ajout de la valeur optimale si applicable
    if cfg["metric"] in ["LSW", "SW"]:
        if cfg["metric"] == "SW":
            y_data.append(np.full_like(y_data[0], results['optimal']['SW']))

        else:
            y_data.append(np.full_like(y_data[0], results['optimal']['LSW']))
        LEGENDS.append("Optimal")

    if cfg["metric"] in ["Bid", "Avg_Bid", "Utility", "Avg_Utility", "Res_Utility"]:
        save_to =  cfg['metric'] + f"_alpha{cfg['alpha']}_gamma{cfg["gamma"]}_n_{cfg['n']}"
        figpath_plot, figpath_legend =plotGame_dim_N(x_data, y_data, cfg["x_label"], cfg["y_label"], LEGENDS, saveFileName=save_to,
                                                     fontsize=40, markersize=20, linewidth=12,linestyle="-",
                                 ylog_scale=cfg["ylog_scale"], pltText=cfg["pltText"], step=cfg["plot_step"])
    else:

        save_to = cfg['metric'] + f"_alpha{cfg['alpha']}_gamma{cfg["gamma"]}_n_{cfg['n']}"
        figpath_plot, figpath_legend = plotGame(x_data, y_data, cfg["x_label"], cfg["y_label"], LEGENDS,
                                                saveFileName=save_to,fontsize=40, markersize=20, linewidth=12,linestyle="-",
                                                ylog_scale=cfg["ylog_scale"], pltText=cfg["pltText"], step=cfg["plot_step"])

