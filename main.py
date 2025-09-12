import os
import numpy as np
import torch, random, time
import streamlit as st

from src.game.utils import *
#from src.gamconfig import SIMULATION_CONFIG as config

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
        bid0 = (c - epsilon) * torch.rand(n) + a

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


        for i in range(self.config["Nb_random_sim"]):
            if not self.config["keep_initial_bid"]:
                bid0 = (c - epsilon) * torch.rand(n) + a

            idx = 0
            for lrMethod in lrMethods:
                lrMethod2 = lrMethod
                Hybrid_funcs, Hybrid_sets = [], []

                if lrMethod == "Hybrid":
                    subset = random.sample(range(self.config["n"]), int(self.config["Nb_A1"][idx]))
                    remaining = [i for i in range(self.config["n"]) if i not in subset]
                    Hybrid_sets = [subset, remaining]

                    #Hybrid_sets = self.config["Hybrids"][idx]["Hybrid_sets"]
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

                # --- Prepare one simulation result ---
                sim_result = {
                    'Speed': error_NE_set.detach().numpy(),
                    'LSW': LSW.detach().numpy(),
                    'SW': SocialWelfare.detach().numpy(),
                    'Dist_To_Optimum_SW': Distance2optSW.detach().numpy(),
                    'Bid': Bids[0].detach().numpy(),
                    'Agg_Bid': Bids[1].detach().numpy(),
                    'Utility': Utility_set[0].detach().numpy(),
                    'Agg_Utility': Utility_set[1].detach().numpy(),
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


