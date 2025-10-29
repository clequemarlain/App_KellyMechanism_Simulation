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
        bid0 = torch.abs( (2*var_init) * torch.rand(n) + z_sol_equ - var_init)

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


        # Réinitialiser les placeholders
        progress_bar = st.progress(0)
        status_text = st.empty()
        game_set = GameKelly(n, price, eps, delta, alpha, tol)
        Bids_Opt, Welfare_Opt, Utility_set_Opt, error_NE_set_Opt = game_set.learning(
            "SBRD", a_vector, c_vector, d_vector, T, eta, bid0, stop=True,
        )
        z_ne =  Bids_Opt[0][-1]
        jain_index_ne = Bids_Opt[2][-1]
        x_ne = z_ne/(torch.sum(z_ne) + self.config["delta"])
        Valuation_ne = Valuation(x_ne, a_vector, d_vector, alpha)
        SW_ne = (torch.sum(Valuation_ne))
        Potential_ne = log_potential(z_ne,a_vector,price)
        Residual_ne = error_NE_set_Opt[-1]
        self.results = {
            'methods': {},
            'optimal': {
                'LSW': LSW_opt,
                'SW': SW_opt,
                'SW_NE': SW_ne,
                'x_opt': x_log_optimum.detach().numpy(),
                'z_ne' : z_ne,
                'x_ne' : x_ne,
                "Potential_ne":  Potential_ne.detach().numpy(),
                "Residual_ne" : Residual_ne.detach().numpy(),
                "Jain_index_NE": jain_index_ne.detach().numpy(),
            }
        }

        for i in range(self.config["Nb_random_sim"]):
            if not self.config["keep_initial_bid"]:
                if self.config["Random_Initial_Bid"]:
                    bid0 = (c - epsilon) * torch.rand(n) + epsilon
                else:
                    var_init = self.config["var_init"]
                    bid0 =  torch.abs((2 * var_init) * torch.rand(n) + z_sol_equ - var_init)
            idx = 0
            idx_rmfq = 0
            NbHybrid = 0

            copy_keys ={}
            Global_Hybrids_set = []
            if "Hybrid" in lrMethods:
                for percent in self.config["Nb_A1"][:self.config["num_hybrids"]]:
                    Global_Hybrids_set.append(make_subset(self.config["n"],percent))

            #print(f"Global_Hybrids_set:{Global_Hybrids_set}")
            for idxMthd, lrMethod in enumerate(lrMethods):
                lrMethod2 = lrMethod
                Hybrid_funcs, Hybrid_sets = [], []

                if lrMethod == "Hybrid" :
                    NbHybrid = NbHybrid+1

                   # subset = random.sample(range(self.config["n"]), int(self.config["Nb_A1"][idx]))
                   # remaining = [i for i in range(self.config["n"]) if i not in subset]
                   # if self.config["Random_set"] and NbHybrid == 1:
                   #     Hybrid_sets = [subset, remaining]
                    #else:
                    Hybrid_sets = Global_Hybrids_set[(NbHybrid-1)%self.config["num_hybrids"]]#make_subset(self.config["n"],NbHybrid)# self.config["Hybrid_sets"][NbHybrid-1]
                    Hybrid_funcs = self.config["Hybrid_funcs"][NbHybrid-1]
                    #print(NbHybrid)
                    if self.config["num_hybrid_set"]>=1 and self.config["num_hybrids"]>1:

                        lrMethod2 = f"({Hybrid_funcs[0]}: {self.config['Nb_A1'][NbHybrid-1]}, {Hybrid_funcs[1]}: {n - self.config['Nb_A1'][NbHybrid-1]})"
                    key = tuple(Hybrid_funcs + ["Hybrid"])
                    if key not in copy_keys:
                        copy_keys[lrMethod2] = key
                    idx += 1

                elif lrMethod != "SBRD": #self.config["num_lrmethod"]!=0:
                    if lrMethod == "RRM":
                        lrMethod2 = f"RRM_{self.config["RRM_lr"][idx_rmfq]}"
                        idx_rmfq += 1
                    else:
                        lrMethod2 = rf"{lrMethod}"# -- $\eta={self.config["Learning_rates"][idxMthd]}$"
                    key = lrMethod
                    if key not in copy_keys:
                        copy_keys[lrMethod] = (key)

                   # idx += 1


                game_set = GameKelly(n, price, eps, delta, alpha, tol)
                Bids, Welfare, Utility_set, error_NE_set = game_set.learning(
                    lrMethod, a_vector, c_vector, d_vector, T, self.config["Learning_rates"][idxMthd], bid0,
                    vary=lr_vary, Hybrid_funcs=Hybrid_funcs, Hybrid_sets=Hybrid_sets
                )
                SocialWelfare = Welfare[0]
                Distance2optSW = 1 / n * torch.abs(SW_opt - Welfare[0])
                Pareto_check =   (Welfare[2] -Valuation_log_opt*torch.ones_like(Welfare[2]) ) + (z_sol_equ*torch.ones_like(Bids[0]) - Bids[0])
                LSW = Welfare[1]
                Relative_Efficienty_Loss = torch.abs((SocialWelfare - SW_opt)/SW_opt) * 100
                # --- Prepare one simulation result ---
                sim_result = {
                    'Speed': error_NE_set.detach().numpy(),
                    'LSW': LSW.detach().numpy(),
                    'SW': SocialWelfare.detach().numpy(),
                    'Dist_To_Optimum_SW': Distance2optSW.detach().numpy(),
                    'Relative_Efficienty_Loss': Relative_Efficienty_Loss.detach().numpy(),
                    'Bid': Bids[0].detach().numpy(),
                    'Avg_Bid': Bids[1].detach().numpy(),
                    "Jain_Index": Bids[2].detach().numpy(),
                    "Pareto": Pareto_check.detach().numpy(),
                    'SBRD_Opt_Bid': Bids_Opt[0][-1].detach().numpy(),

                    'SBRD_Opt_Avg_Bid': Bids_Opt[1].detach().numpy(),
                    'Payoff': Utility_set[0].detach().numpy(),
                    'SBRD_Opt_Utility': Utility_set_Opt[0][-1].detach().numpy(),
                    'Avg_Payoff': Utility_set[1].detach().numpy(),
                    'Res_Payoff': Utility_set[2].detach().numpy(),
                    'Potential': Utility_set[3].detach().numpy(),
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
        # --- After loop: compute averages ---
        idxHybrid = 0
        self.results_copy = self.results.copy()

        # 🔑 on fige les éléments dans une liste pour éviter le RuntimeError
        for method, metrics in list(self.results_copy["methods"].items()):
            #print(method)
            # Vérifier si la méthode est hybride
            is_hybrid = False
            if method in copy_keys:
                keys = copy_keys[method]
                if keys[-1] == "Hybrid" and self.config["num_hybrid_set"]>=1 and self.config["num_hybrids"]>1 :
                    is_hybrid = True


            for k, v_list in metrics.items():
                # Cas scalaires (pas des tableaux)

                if k in ["convergence_iter"]:
                    self.results["methods"][method][k] = np.mean(v_list)

                else:

                    mean_val = np.mean(np.stack(v_list), axis=0)
                    self.results["methods"][method][k] = mean_val

                   # print(k,self.results["methods"][method][k])



                try:

                    if is_hybrid :#and len(self.config["selected_methods"])==1:


                        # Construire un nom plus clair pour l’hybride
                        eta_val = self.config["Learning_rates"][idxHybrid]
                        key_name = rf"{keys[1]}"# -- $\eta={eta_val}$"
                        if key_name not in self.results["methods"]:
                            self.results["methods"][key_name] = {}

                        if k not in self.results["methods"][key_name]:
                            self.results["methods"][key_name][k] = []

                        # Ajouter la dernière valeur de la courbe

                        if k in ["convergence_iter"]:
                            self.results["methods"][key_name][k] = np.mean(v_list)
                        else:
                            self.results["methods"][key_name][k].append(mean_val[-1])

                    #else:
                    #    print(f" np.mean(v_list){ np.mean(v_list)}")
                    #    self.results["methods"][method][k] = np.mean(v_list)

                except Exception:
                    print()

                    #self.results["methods"][method][k] = np.mean(v_list)

            if is_hybrid:
                idxHybrid += 1  # ⚡ n’incrémenter que si c’est un hybride
            #print(k, self.results["methods"][method][k])

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

