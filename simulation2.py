import numpy as np
import torch
from src.game.utils import GameKelly, x_log_opt, LSW_func, Utility


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
        mu = self.config["mu"]
        c = self.config["c"]
        delta = self.config["delta"]
        epsilon = self.config["epsilon"]
        Hybrid_funcs = self.config["Hybrid_funcs"]
        lr_vary = self.config["lr_vary"]
        alpha = self.config["alpha"]
        gamma = self.config["gamma"]
        tol = self.config["tol"]
        lrMethods = self.config["lrMethods"]

        eps = epsilon * torch.ones(1)
        bid0 = torch.ones(n)

        a_vector = torch.tensor([a / (i + 1) ** gamma for i in range(n)], dtype=torch.float64)
        c_vector = torch.tensor([c / (i + 1) ** mu for i in range(n)], dtype=torch.float64)
        dmin = a_vector * torch.log((epsilon + torch.sum(c_vector) - c_vector + delta) / epsilon)
        d_vector = 0.7 * dmin

        # Calculer l'optimum
        x_log_optimum = x_log_opt(c_vector, a_vector, d_vector, eps, delta, price, bid0,alpha)
        utilities_log_opt = Utility(x_log_optimum, a_vector, d_vector, alpha)
        LSW_opt = torch.sum(torch.minimum(utilities_log_opt, c_vector)).detach().numpy()
        SW_opt = torch.sum(utilities_log_opt).detach().numpy()

        self.results = {
            'methods': {},
            'optimal': {
                'LSW': LSW_opt,
                'SW': SW_opt,
                'x_opt': x_log_optimum.detach().numpy()
            }
        }

        set1 = torch.arange(n, dtype=torch.long)
        nb_hybrid = len(Hybrid_funcs)
        Hybrid_sets = torch.chunk(set1, nb_hybrid) if nb_hybrid > 0 else []

        for lrMethod in lrMethods:
            game_set = GameKelly(n, price, eps, delta, alpha, tol)
            Bids, Utilities, error_NE_set = game_set.learning(
                lrMethod, a_vector, c_vector, d_vector, T, eta, bid0,
                vary=lr_vary, Hybrid_funcs=Hybrid_funcs, Hybrid_sets=Hybrid_sets
            )

            SocialWelfare = torch.sum(Utilities, dim=1)
            LSW = torch.sum(torch.minimum(Utilities, c_vector.unsqueeze(0)), dim=1)
            Avg_bids = torch.mean(Bids, dim=1)

            self.results['methods'][lrMethod] = {
                'error_NE': error_NE_set.detach().numpy(),
                'LSW': LSW.detach().numpy(),
                'SW': SocialWelfare.detach().numpy(),
                'bids': Bids.detach().numpy(),
                'utilities': Utilities.detach().numpy(),
                'avg_bids': Avg_bids.detach().numpy(),
                'final_bids': Bids[-1].detach().numpy(),
                'convergence_iter': torch.argmin(error_NE_set).item() if torch.min(error_NE_set) <= tol else T
            }

        return self.results