import os
import numpy as np
import torch

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
        init_bid = self.config["init_bid"]
        c = self.config["c"]
        delta = self.config["delta"]
        epsilon = self.config["epsilon"]
        Hybrid_funcs = self.config["Hybrid_funcs"]
        lr_vary = self.config["lr_vary"]
        alpha = self.config["alpha"]
        gamma = self.config["gamma"]
        tol = self.config["tol"]
        lrMethods = self.config["lrMethods"]
        Hybrid_sets = self.config["Hybrid_sets"]

        #print(f"Hybrid_sets:{Hybrid_sets}")
        eps = epsilon * torch.ones(1)
        bid0 = init_bid * torch.ones(n)

        c_min = epsilon
        #d_min = 0

        a_vector = torch.tensor([max(a - i * gamma, a_min) for i in range(n)], dtype=torch.float64)
        c_vector = torch.tensor([max(c -i * mu, c_min) for i in range(n)], dtype=torch.float64)
        dmin = a_vector * torch.log((epsilon + torch.sum(c_vector) - c_vector + delta) / epsilon)
        d_vector = 0.7 * dmin*0

        # Calculer l'optimum
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

        for lrMethod in lrMethods:
            game_set = GameKelly(n, price, eps, delta, alpha, tol)
            Bids, Welfare, Utility_set, error_NE_set = game_set.learning(
                lrMethod, a_vector, c_vector, d_vector, T, eta, bid0,
                vary=lr_vary, Hybrid_funcs=Hybrid_funcs, Hybrid_sets=Hybrid_sets
            )

            SocialWelfare = Welfare[0]
            Distance2optSW = SW_opt - Welfare[0]
            LSW = Welfare[1]

            self.results['methods'][lrMethod] = {
                'error_NE': error_NE_set.detach().numpy(),
                'LSW': LSW.detach().numpy(),
                'SW': SocialWelfare.detach().numpy(),
                'Dist_To_Optimum_SW': Distance2optSW.detach().numpy(),
                'bids': Bids[0].detach().numpy(),
                'utilities': Utility_set[0].detach().numpy(),
                'avg_bids': Utility_set[1].detach().numpy(),
                'final_bids': Bids[0][-1].detach().numpy(),
                'convergence_iter': torch.argmin(error_NE_set).item() if torch.min(error_NE_set) <= tol else T
            }

        return self.results


