import torch
# main.py
import os
import numpy as np
import torch
import random
from math import ceil, sqrt

from networkx.algorithms.efficiency_measures import efficiency
from sympy.abc import epsilon
from sympy.codegen.ast import continue_

from utils import *
from config import SIMULATION_CONFIG as cfg
def make_population(n, d, *,
                    alpha=1.0, beta=None,
                    c=1.0, eps=0.05,
                    alpha_het=0.2, c_het=0.2,
                    device="cpu", dtype=torch.float64):
    """
    Returns:
      alpha_i: (n,)
      c_i:     (n,)
      beta_k:  (d,)
      eps:     scalar
    """
    if beta is None:
        beta = torch.linspace(1.0, float(d), d, device=device, dtype=dtype)

    beta_k = torch.as_tensor(beta, device=device, dtype=dtype)

    # simple heterogeneity: decreasing or random (choose what you prefer)
    idx = torch.arange(n, device=device, dtype=dtype)
    alpha_i = torch.clamp(torch.tensor(alpha, device=device, dtype=dtype) + (- alpha_het * idx), min=eps)
    c_i = torch.clamp(torch.tensor(c, device=device, dtype=dtype) + (- c_het * idx), min=d * eps)

    eps = torch.tensor(float(eps), device=device, dtype=dtype)
    return alpha_i, c_i, beta_k, eps

# main.py
import os
import numpy as np
import torch
import random
from math import ceil, sqrt

from sympy.codegen.ast import continue_

from utils import *
from config import SIMULATION_CONFIG as cfg



# -------------------------
# Small helpers (clean main)
# -------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_torch(x, dtype=torch.float64, device="cpu"):
    if torch.is_tensor(x):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


def safe_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mean_stack(v_list):
    """Mean over runs for arrays of same shape."""
    return np.mean(np.stack(v_list, axis=0), axis=0)


def build_vectors(n, a, a_min, gamma, c, c_min, mu, epsilon, delta, dtype, device):
    a_vector = torch.tensor([max(a - i * gamma, a_min) for i in range(n)], dtype=dtype, device=device)
    c_vector = torch.tensor([max(c -  mu, c_min) for i in range(n)], dtype=dtype, device=device)

    # your original d_vector construction (kept)
    dmin = a_vector * torch.log((epsilon + torch.sum(c_vector) - c_vector + delta) / epsilon)
    d_vector = 0.7 * dmin * 0
    return a_vector, c_vector, d_vector


def init_bid(n, d, c, epsilon, method="uniform", dtype=torch.float64, device="cpu"):
    #if method == "uniform":
    return (c - epsilon) * torch.rand((n, d), dtype=dtype, device=device) + epsilon
    # fallback
    #return epsilon * torch.ones((n, d), dtype=dtype, device=device)


def compute_payoff_bounds(n, d, A_matrix, c_vector, d_vector, eps, delta, price, alpha, bid0, Y):
    # Bounds used by your normalization (kept logic)
    s_min = (n - 1) * eps + delta
    s_max = (n - 1) * float(c_vector.max().item()) + delta  # conservative if heterogeneous c_vector

    # max payoff (BR at s_min)
    payoff_mins = []
    payoff_maxs = []
    payoff_maxs_log = []
    payoff_maxs_lin = [0]
    l =0
    for k,alpha_ in enumerate(alpha):
        if alpha_ ==1:

            s_max = (n - 1) * eps + delta
            s_min = (n - 1) * float(c_vector.max().item()) + delta
            z_max = BR_alpha_fair(eps, c_vector, bid0[:, 0], s_min, A_matrix[:,k], delta, alpha_, price, b=0) ###
            z_min = eps
            x_min = z_min[:,k] / (z_min[:,k] + s_min)
            x_max = z_max / (z_max + s_max[:,k])
            payoff_maxs_log.append(torch.max(Payoff_matrix(x_max, z_max, A_matrix[:,k], d_vector, alpha_, price, Y,k=k)))
            l = +1
            payoff_min = Payoff_matrix(x_min, z_min[:, k], A_matrix[:, k], d_vector, alpha_, price, Y, k=k)

        if alpha_==0 :
            z_max = eps

            s_max = (n - 1) * eps + delta
            s_min = (n - 1) * c_vector + delta
            z_max = BR_alpha_fair(eps, c_vector, bid0[:, 0], s_min[:,k], A_matrix[:,k], delta, alpha_, price, b=0)
            z_min = c_vector
            x_min = z_min / (z_min + s_min)
            x_max = z_max / (z_max + s_max[:,k])
            print(z_min)
            payoff_maxs_lin.append(torch.max(Payoff_matrix(x_max, z_max, A_matrix[:,k], d_vector, alpha_, price, Y, k=k)))
        #zT / (zT.sum(dim=0, keepdim=True) + delta)
       # x_max = z_max / (z_max.sum(dim=0, keepdim=True) + s_min)

        # min payoff (two candidates)


            payoff_min = Payoff_matrix(x_min[:,k], z_min[:,k], A_matrix[:,k], d_vector, alpha_, price, Y,k=k )
        #payoff_maxs.append(torch.max(Payoff(x_max, z_max, a_vector, d_vector, [alpha_], price, [beta[k]], Y)))
        payoff_mins.append(torch.min(payoff_min))
    if max(payoff_maxs_lin)>0:
        payoff_maxs =max(payoff_maxs_lin)
    else:
        payoff_maxs = max(payoff_maxs_log)
    print((payoff_mins), (payoff_maxs))
    return sum(payoff_mins), (payoff_maxs)


def compute_optimum_metrics(c_vector, a_vector, d_vector, eps, delta, price, alpha, bid0,Y):
    x_opt = x_log_opt(c_vector, a_vector, d_vector, eps, delta, price, bid0)
    val_opt = 0#Valuation(0.33*torch.ones((bid0.shape[0],len(alpha))), a_vector, d_vector, alpha,Y)
    sw_opt = 0# torch.sum(val_opt)
    lsw_opt =0# torch.sum(torch.minimum(val_opt, c_vector))
    return x_opt, val_opt, sw_opt, lsw_opt

def add_rank_r_noise(A, r, sigma=1.0, mask=None, nonneg=False, eps=0.0):
    """
    Add low-rank noise N=U V^T with rank <= r.
    If mask is provided (0/1), noise is applied entrywise; rank is no longer guaranteed.
    """

    n, d = A.shape
    print(n, r)
    U = torch.randn(n, r, device=A.device, dtype=A.dtype)
    V = torch.randn(d, r, device=A.device, dtype=A.dtype)
    N = sigma * (U @ V.T)  # rank <= r

    if mask is not None:
        N = N * mask  # WARNING: masking can increase rank

    A2 = A + N

    if nonneg:
        A2 = torch.clamp(A2, min=eps)

    return A2
def project_rank_r(A, r, nonneg=False, eps=0.0):
    """
    Best rank-r approximation in Frobenius norm via truncated SVD.
    """
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    Ur = U[:, :r]
    Sr = S[:r]
    Vhr = Vh[:r, :]
    A_r = (Ur * Sr) @ Vhr  # rank <= r

    if nonneg:
        A_r = torch.clamp(A_r, min=eps)
    return A_r

def numerical_rank(A, tol=1e-10):
    S = torch.linalg.svdvals(A)
    return int((S > tol * S.max()).sum().item())

# -------------------------
# Runner
# -------------------------
class SimulationRunner:
    def __init__(self, config):
        self.config = config
        self.results = {}
    def run_simulation_learning_rules(self):
        # --- Config ---
        T = self.config["T"]
        n = self.config["n"]
        d = len(self.config["alpha"])
        eta = self.config["eta"]
        price = self.config["price"]
        alpha = self.config["alpha"]
        beta = self.config["beta"]
        Y = self.config["Y"]
        a = self.config["a"]
        a_min = self.config["a_min"]
        gamma = self.config["gamma"]
        noise_a = self.config["noise"]

        c = self.config["c"]  # FIXED budget
        mu = self.config["mu"]
        # x-axi#self.config["mu_r"]  # heterogeneity levels

        delta = self.config["delta"]
        epsilon = self.config["epsilon"]
        tol = self.config["tol"]

        lrMethod = self.config["lrMethods"][0]
        lr_vary = self.config["lr_vary"]
        Nb_random_sim = self.config["Nb_random_sim"]

        set_seed(self.config.get("seed", 0))

        device = self.config.get("device", "cpu")
        dtype = torch.float64
        eps = to_torch(epsilon, dtype=dtype, device=device)
        mu_r_grid = torch.linspace(0.0, c-10, 10, device=device, dtype=dtype)

        # storage: one entry per mu_r
        results = {
            "final_res": {},
            "mu_r": [],
            "final_bids": {},
            "bids_history": {},
            "loss" :{},
            "utilities":{},
            "d_utilities": {},
            "resource_utilities":{},
            "final_allocations": {},
            "convergence_iter": {},
            "convergence_error": {},
            "jain_index": {},
            "alpha_fair_index": {},
        }

        a_vector, c_vector, d_vector = build_vectors(
            n=n, a=a, a_min=a_min, gamma=gamma,
            c=c, c_min=epsilon, mu=mu,
            epsilon=eps, delta=delta,
            dtype=dtype, device=device
        )
        print("a_vector",a_vector)

        # resource-specific vector beta_k
        #beta = torch.tensor([i+1 for i in range(d)], dtype=dtype, device=device)
        beta = torch.arange(1, d + 1, dtype=dtype, device=device)
        beta = 5.0 * beta / beta.sum()
        beta[0]= torch.ones_like(beta)[0]
        #beta[1]= 0.5*torch.ones_like(beta)[0]


        # (optional) binary mask for structural zeros


        # valuation matrix
        A_matrix =  (beta[None, :] * a_vector[:, None] )
        A0 = beta[None, :] * a_vector[:, None]  # rank 1 (if beta,a nonzero)

        if self.config["noise"] > 0:
            A_matrix = add_rank_r_noise(A0, r=self.config["noise_rank"], sigma=self.config["noise"], mask=None, nonneg=True,
                                        eps=0.0)
        else:
            A_matrix = A0

        print(f"numerical_rank(A):{numerical_rank(A_matrix)}")

        #print(beta[None, :], a_vector[:, None], beta[None, 0]*a_vector[:, None][0] )

        bid0 = init_bid(n, d, c, epsilon, dtype=dtype, device=device)
       # x_opt, val_opt, SW_opt, LSW_opt = compute_optimum_metrics(
       #     c_vector, a_vector, d_vector, eps, delta, price, [alpha[0]], bid0[:, 0],Y
       # )
        eps_vec = epsilon *torch.ones_like(bid0)
        c_vec =  c *torch.ones_like(eps_vec)
        x_min = eps_vec/(eps_vec + torch.sum(torch.max(c_vector)*Y, dim=0) )
        x_max = c_vec/(c_vec + torch.sum(epsilon*Y, dim=0) )

        x_opt, sw_opt = 0,0#optimal_x_multiresource_matrix(A_matrix, alpha_k=alpha, c_vector=c_vector,eps=eps_vec, Y=Y, delta=delta,

        Payoff_min, Payoff_max =1,2 #compute_payoff_bounds(

        Z = epsilon*torch.ones_like(bid0) * Y

        print(f"A_matrix:{A_matrix}, {c_vector}")
        print(beta[None, :], a_vector[:, None], beta[None, 0] * a_vector[:, None][0])

        for lrMethod in self.config["lrMethods"]:
            bids_runs, alloc_runs, conv_runs, residues, payoff_runs, payoff_runs_d, efficiencies = [], [], [], [], [], [], []
            jain_runs, alpha_fair_runs = [], []
            game = GameKelly(n, d, beta, price, eps, delta, alpha, tol, Y,
                             payoff_min=1, payoff_max=2)
            bid0 = init_bid(n, d, c, epsilon, dtype=dtype, device=device)

            bid0 = Q_simplex(bid0, epsilon, c_vector, Y) * Y # ensure feasibility
            print(f"HERE bid0: {bid0}")
            #print(Payoff_max, Payoff_min, a_vector, c_vector)
            for _ in range(Nb_random_sim):
                d_payoff = torch.zeros((n, d), dtype=dtype, device=device)

                Bids, Welfare, Utility, error = game.learning(
                    lrMethod, A_matrix, c_vector, d_vector,
                    T, eta, bid0, vary=lr_vary
                )

                zT = Bids[0][-1]  # (n,d)
                xT = zT / (zT.sum(dim=0, keepdim=True) + delta)
                utility = (Payoff_matrix(xT, zT, A_matrix, d_vector, alpha, price, Y) - Payoff_min)/ (Payoff_max - Payoff_min)

                z_tr = 100*Y*torch.ones_like(zT)/ torch.sum(Y, dim=0)
                x_tr = z_tr / (z_tr.sum(dim=0, keepdim=True) + delta)
                sw =  Valuation(xT, a_vector, d_vector, alpha, beta, Y).sum()

                efficiency = torch.abs((Welfare[0] - sw_opt) / sw_opt)

                print(zT)
                print(f"Budget:{zT.sum(dim=1, keepdim=True)}")
                print(f"xT:{xT}, \n{x_opt}")

                bids_runs.append(Bids[0].cpu().numpy())#.mean(dim=0).cpu().numpy())
                alloc_runs.append(xT.cpu().numpy())#.mean(dim=0).cpu().numpy())
                residues.append(error.cpu().numpy())
                payoff_runs.append(utility.cpu().numpy())
                payoff_runs_d.append(d_payoff.cpu().numpy())
                efficiencies.append(efficiency.cpu().numpy())
                jain_runs.append(Bids[2].cpu().numpy())
                alpha_fair_runs.append(Bids[3].cpu().numpy())

                conv = int(torch.argmin(error)) if torch.min(error) <= tol else T
                conv_runs.append(conv)

            results["mu_r"].append(mu)
            results["bids_history"][lrMethod] = np.mean(bids_runs, axis=0)


            results["final_allocations"][lrMethod] = np.mean(alloc_runs, axis=0)
            results["convergence_iter"][lrMethod] = np.mean(conv_runs)
            results["loss"][lrMethod] = np.mean(efficiencies, axis=0)
            results["convergence_error"][lrMethod] = np.mean(residues, axis=0)
            results["utilities"][lrMethod] =  np.mean(payoff_runs, axis=0)
            results["d_utilities"][lrMethod] = np.mean(payoff_runs_d, axis=0)
            results["jain_index"][lrMethod] = np.mean(jain_runs, axis=0)
            results["alpha_fair_index"][lrMethod] = np.mean(alpha_fair_runs, axis=0)
            zf = torch.tensor(results["bids_history"][lrMethod][-1])
            z_br = game.proj_residual(zf, A_matrix, c_vector)
            results["final_res"][lrMethod] =  torch.norm(z_br - zf, p=2)
            print(f"RESIUAL:{results["final_res"][lrMethod]}, \n {z_br},\n {zf}")

            print("Speed", lrMethod, c, results["convergence_iter"][lrMethod])
        return results
    def run_simulation_budget(self):
        # --- Config ---
        T = self.config["T"]
        n = self.config["n"]
        d = len(self.config["alpha"])
        eta = self.config["eta"]
        price = self.config["price"]
        alpha = self.config["alpha"]
        beta = self.config["beta"]
        Y = self.config["Y"]

        a = self.config["a"]
        a_min = self.config["a_min"]
        gamma = self.config["gamma"]

        c = self.config["c"]  # FIXED budget
        # x-axi#self.config["mu_r"]  # heterogeneity levels

        delta = self.config["delta"]
        epsilon = self.config["epsilon"]
        tol = self.config["tol"]

        lrMethod = self.config["lrMethods"][0]
        lr_vary = self.config["lr_vary"]
        Nb_random_sim = self.config["Nb_random_sim"]

        set_seed(self.config.get("seed", 0))

        device = self.config.get("device", "cpu")
        dtype = torch.float64
        eps = to_torch(epsilon, dtype=dtype, device=device)
        mu_r_grid = torch.linspace(0.0, c-10, 10, device=device, dtype=dtype)

        # storage: one entry per mu_r
        results = {
            "mu_r": [],
            "final_bids": [],
            "final_allocations": [],
            "convergence_iter": [],
        }

        for mu_r in mu_r_grid:
            bids_runs, alloc_runs, conv_runs = [], [], []

            for _ in range(Nb_random_sim):
                # budgets: c_i = max(c - i * mu_r, epsilon)
                a_vector, c_vector, d_vector = build_vectors(
                    n=n, a=a, a_min=a_min, gamma=gamma,
                    c=c, c_min=epsilon, mu=mu_r,
                    epsilon=eps, delta=delta,
                    dtype=dtype, device=device
                )

                bid0 = init_bid(n, d, c, epsilon, dtype=dtype, device=device)
                bid0 = Q_simplex(bid0, epsilon, c_vector,Y)  # ensure feasibility

                game = GameKelly(n, d, beta, price, eps, delta, alpha, tol, Y,
                                 payoff_min=1, payoff_max=2)

                Bids, Welfare, Utility, error = game.learning(
                    lrMethod, a_vector, c_vector, d_vector,
                    T, eta, bid0, vary=lr_vary
                )

                zT = Bids[0][-1]  # (n,d)
                xT = zT / (zT.sum(dim=0, keepdim=True) + delta)
                print(zT,c_vector)
                print(f"xT:{xT}",zT[:,1]/zT[:,0])

                bids_runs.append(zT.cpu().numpy())#.mean(dim=0).cpu().numpy())
                alloc_runs.append(xT.cpu().numpy())#.mean(dim=0).cpu().numpy())

                conv = int(torch.argmin(error)) if torch.min(error) <= tol else T
                conv_runs.append(conv)

            results["mu_r"].append(mu_r)
            results["final_bids"].append(np.mean(bids_runs, axis=0))
            results["final_allocations"].append(np.mean(alloc_runs, axis=0))
            results["convergence_iter"].append(np.mean(conv_runs))

        return results

    def run_simulation(self):
        # --- Config ---
        T = self.config["T"]
        n = self.config["n"]
        d = self.config["R"]
        eta = self.config["eta"]
        price = self.config["price"]
        alpha = self.config["alpha"]
        beta = self.config["beta"]

        a = self.config["a"]
        a_min = self.config["a_min"]
        gamma = self.config["gamma"]

        c = self.config["c"]
        mu = self.config["mu"]
        delta = self.config["delta"]
        epsilon = self.config["epsilon"]
        tol = self.config["tol"]

        lrMethods = self.config["lrMethods"]
        lr_vary = self.config["lr_vary"]

        Nb_random_sim = self.config["Nb_random_sim"]

        # reproducibility
        seed = self.config.get("seed", 0)
        set_seed(seed)

        device = self.config.get("device", "cpu")
        dtype = torch.float64

        eps = to_torch(epsilon, dtype=dtype, device=device)

        # vectors
        c_min = epsilon
        a_vector, c_vector, d_vector = build_vectors(
            n=n, a=a, a_min=a_min, gamma=gamma,
            c=c, c_min=c_min, mu=mu,
            epsilon=eps, delta=delta,
            dtype=dtype, device=device
        )


        # initial bid
        bid0 = init_bid(n, d, c, epsilon, method="uniform", dtype=dtype, device=device)

        # optimum (kept)
        x_opt, val_opt, SW_opt, LSW_opt = compute_optimum_metrics(
            c_vector, a_vector, d_vector, eps, delta, price, alpha, bid0[:, 0]
        )

        # payoff normalization bounds (kept)
        Payoff_min, Payoff_max = 1,2#compute_payoff_bounds(
        #    n, d, a_vector, c_vector, d_vector, eps, delta, price, alpha, bid0
        #)
        #print(Payoff_max)
        # reference NE using SBRD (kept, but now robust for d)
        #print(price)
        game_ref = GameKelly(n, d, beta, price, eps, delta, alpha, tol, payoff_min=Payoff_min, payoff_max=Payoff_max)
        Bids_ref, Welfare_ref, Utility_ref, error_NE_ref = game_ref.learning(
            "DA", a_vector, c_vector, d_vector, T//100, eta, bid0, stop=False
        )

        z_ne = Bids_ref[0][-1]                 # (n,d)
        jain_index_ne = Bids_ref[2][-1]

        # NOTE: your original x_ne was wrong for (n,d); fix: per resource share
        x_ne = torch.stack([z_ne[:, k] / (z_ne[:, k].sum() + delta) for k in range(d)], dim=1)

        eq_log = (n-1)/n*a_vector[0] *beta[1]
        eq_lin = (n-1)/n**2*a_vector[0] *beta[0]
        eq = torch.ones_like(bid0)
        eq[:, 0] = eq_lin * eq[:,0]
        eq[:, 1] = eq_log * eq[:, 1]



        # If your Valuation/Payoff expect single-resource vectors, keep using resource 0;
        # otherwise you should aggregate. Here I keep exactly your original behavior using resource 0.
        Valuation_ne = 1#Valuation(x_ne[:, 0], a_vector, d_vector, alpha)
        SW_ne = 1#torch.sum(Valuation_ne)

        payoff_ne = 1#Payoff(x_ne[:, 0], z_ne[:, 0], a_vector, d_vector, alpha, price)
        payoff_ne_norm = (payoff_ne - Payoff_min) / (Payoff_max - Payoff_min)

        Potential_ne = log_potential(z_ne[:, 0], a_vector, price)
        Residual_ne = error_NE_ref[-1]

        self.results = {
            "methods": {},
            "optimal": {
                "LSW": float(LSW_opt.detach().cpu().item()),
                "SW": float(SW_opt.detach().cpu().item()),
               # "SW_NE": float(SW_ne.detach().cpu().item()),
                "x_opt": safe_np(x_opt),
                "z_ne": safe_np(z_ne),
                "x_ne": safe_np(x_ne),
                "payoff_ne": safe_np(payoff_ne_norm),
                "Potential_ne": safe_np(Potential_ne),
                "Residual_ne": float(Residual_ne.detach().cpu().item()) if torch.is_tensor(Residual_ne) else float(Residual_ne),
                "Jain_index_NE": float(jain_index_ne.detach().cpu().item()),
            }
        }

        # ------------------------------------------------------------
        # Prepare Hybrid subsets once (so it is reproducible + cleaner)
        # ------------------------------------------------------------
        copy_keys = {}
        Global_Hybrids_set = []
        if "Hybrid" in lrMethods:
            for percent in self.config["Nb_A1"][: self.config["num_hybrids"]]:
                Global_Hybrids_set.append(make_subset(n, percent))

        # ------------------------------------------------------------
        # Runs
        # ------------------------------------------------------------
        for run in range(Nb_random_sim):

            # re-init bid
            if self.config.get("Random_Initial_Bid", True):
                bid0 = init_bid(n, d, c, epsilon, method="uniform", dtype=dtype, device=device)

            NbHybrid = 0
            idx_rmfq = 0
            bid0 = eq.clone()

            for lrMethod in lrMethods:
                lrMethod2 = lrMethod
                Hybrid_funcs, Hybrid_sets = [], []

                # --- Hybrid labelling & subset selection ---
                if lrMethod == "Hybrid":
                    NbHybrid += 1
                    Hybrid_sets = Global_Hybrids_set[(NbHybrid - 1) % max(1, self.config["num_hybrids"])]
                    Hybrid_funcs = self.config["Hybrid_funcs"][NbHybrid - 1]

                    # pretty label
                    if self.config.get("num_hybrid_set", 0) >= 1 and self.config["num_hybrids"] > 1:
                        a1 = self.config["Nb_A1"][NbHybrid - 1]
                        lrMethod2 = f"({Hybrid_funcs[0]}: {a1}, {Hybrid_funcs[1]}: {n - a1})"

                    key = tuple(Hybrid_funcs + ["Hybrid"])
                    if lrMethod2 not in copy_keys:
                        copy_keys[lrMethod2] = key

                elif lrMethod != "SBRD":
                    if lrMethod == "RRM_nt":
                        lrMethod2 = f"RRM_nt_{self.config['RRM_lr'][idx_rmfq]}"
                        idx_rmfq += 1
                    if lrMethod not in copy_keys:
                        copy_keys[lrMethod] = lrMethod

                # --- simulate ---
                game = GameKelly(n, d,beta, price, eps, delta, alpha, tol, payoff_min=Payoff_min, payoff_max=Payoff_max)
                Bids, Welfare, Utility_set, error_NE_set = game.learning(
                    lrMethod, a_vector, c_vector, d_vector, T, eta, bid0,
                    vary=lr_vary, Hybrid_funcs=Hybrid_funcs, Hybrid_sets=Hybrid_sets
                )

                #print(print(Q_simplex((n-1)/n, eps, c_vector)))

                SocialWelfare = Welfare[0]
                LSW = Welfare[1]
                Relative_Efficienty_Loss = torch.abs((SocialWelfare - SW_opt) / SW_opt) * 100
                Distance2optSW = (1 / n) * torch.abs(SW_opt - SocialWelfare)

                # Your Pareto_check is inconsistent dimensionally; keep but make it safe:
                # (use last bids only to avoid huge tensor)
                try:
                    Pareto_check = (Welfare[2] - val_opt * torch.ones_like(Welfare[2])) + (Bids_ref[0][-1] - Bids[0][-1])
                except Exception:
                    Pareto_check = torch.zeros(1, dtype=dtype, device=device)
                #print(Utility_set[0].shape,Payoff_min.shape,Payoff_max.shape)
                Payoff_Norm = (Utility_set[0] - Payoff_min) / (Payoff_max - Payoff_min)
                AvgPayoff_norm = (Utility_set[1] - Payoff_min) / (Payoff_max - Payoff_min)

                sim_result = {
                    "Speed": safe_np(error_NE_set),
                    "LSW": safe_np(LSW),
                    "SW": safe_np(SocialWelfare),
                    "Dist_To_Optimum_SW": safe_np(Distance2optSW),
                    "Relative_Efficienty_Loss": safe_np(Relative_Efficienty_Loss),
                    "Bid": safe_np(Bids[0]),
                    "Avg_Bid": safe_np(Bids[1]),
                    "Jain_Index": safe_np(Bids[2]),
                    "Alpha_Fair_Index": safe_np(Bids[3]),
                    "Pareto": safe_np(Pareto_check),
                    "SBRD_Opt_Bid": safe_np(Bids_ref[0][-1]),
                    "SBRD_Opt_Avg_Bid": safe_np(Bids_ref[1]),
                    "Payoff": safe_np(Payoff_Norm),
                    "epsilon_error": safe_np(Utility_set[4]) if len(Utility_set) > 4 else None,
                    "epsilon_error_Hybrid": safe_np(Utility_set[5]) if len(Utility_set) > 5 else None,
                    "SBRD_Opt_Utility": safe_np(Utility_ref[0][-1]) if isinstance(Utility_ref, list) else None,
                    "Avg_Payoff": safe_np(AvgPayoff_norm),
                    #"Res_Payoff": safe_np(Utility_set[2]),
                    #"Potential": safe_np(Utility_set[3]),
                    "final_bids": safe_np(Bids[0][-1]),
                    "convergence_iter": int(torch.argmin(error_NE_set).item()) if torch.min(error_NE_set) <= tol else int(T),
                }

                # accumulate
                if lrMethod2 not in self.results["methods"]:
                    self.results["methods"][lrMethod2] = {k: [v] for k, v in sim_result.items()}
                else:
                    for k, v in sim_result.items():
                        self.results["methods"][lrMethod2][k].append(v)

        print("correct",eq, Q_simplex(eq, epsilon, c_vector))
        c_1 = Q_simplex(eq, epsilon, c_vector)
        print("0",torch.sum(c_1, dim=0), c_1)
        print("1",Payoff(c_1/torch.sum(c_1, dim=0), c_1, a_vector, d_vector, alpha, price))
        print("2", Bids[0][-1],Payoff(Bids[0][-1] / torch.sum(Bids[0][-1], dim=0), Bids[0][-1], a_vector, d_vector, alpha, price))
        # ------------------------------------------------------------
        # Average over runs
        # ------------------------------------------------------------
        results_copy = dict(self.results)  # shallow copy
        for method, metrics in list(results_copy["methods"].items()):
            for k, v_list in metrics.items():
                if k == "convergence_iter":
                    self.results["methods"][method][k] = float(np.mean(v_list))
                else:
                    # keep None fields as None
                    if v_list[0] is None:
                        self.results["methods"][method][k] = None
                    else:
                        self.results["methods"][method][k] = mean_stack(v_list)

        return self.results

import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def add_zoom_inset(
    ax, it, curves_dict, *,
    cfg,
    algo_colors,
    algo_styles,
    mu_markers=None,     # dict: mu -> marker, optional
    mu=None,             # if provided, marker for all curves uses mu_markers[mu]
    ylog=False,
    pct_xmax=None,       # if not None => PercentFormatter(xmax=pct_xmax)
    add_final_lines=False,
    final_as_ticks=False,
):
    if not cfg.get("Add_Zoom", False):
        return

    x_idx_min, x_idx_max = cfg.get("x_zoom_interval", (0, len(it)))
    x_idx_min = max(0, int(x_idx_min))
    x_idx_max = min(int(x_idx_max), len(it))
    assert x_idx_max > x_idx_min, "Invalid x_zoom_interval"

    y_min, y_max = cfg.get("y_zoom_interval", (None, None))
    use_auto_y = (y_min is None) or (y_max is None)

    inset_rect = cfg.get("inset_rect", [0.55, 0.52, 0.42, 0.42])
    axins = ax.inset_axes(inset_rect)

    y_slices = []
    final_levels = []
    final_colors = []

    for name, y in curves_dict.items():
        y = np.asarray(y, dtype=float)
        L = min(len(it), len(y))
        y_plot = y[:L]

        j2 = min(x_idx_max, L)
        x_zoom = it[x_idx_min:j2]
        y_zoom = y_plot[x_idx_min:j2]
        y_slices.append(y_zoom)

        ls, _ = algo_styles.get(name, ("-", None))
        col = algo_colors.get(name, None)

        mk = None
        if (mu is not None) and (mu_markers is not None):
            mk = mu_markers[mu]

        axins.plot(
            x_zoom, y_zoom,
            color=col,
            linestyle=ls,
            marker=mk,
            linewidth=2,
            markersize=5,
            markeredgecolor="black" if mk is not None else None,
        )

        if add_final_lines:
            y_final = float(y_plot[-1])
            final_levels.append(y_final)
            final_colors.append(col)

    # x-limits
    x1, x2 = it[x_idx_min], it[x_idx_max - 1]
    axins.set_xlim(x1, x2)

    # y-limits
    if use_auto_y:
        y1, y2 = _auto_y_limits(
            y_slices,
            ylog=ylog,
            q=cfg.get("zoom_quantiles", (0.02, 0.98)),
            margin=cfg.get("zoom_margin", 0.20),
            eps=cfg.get("zoom_log_eps", 1e-12),
        )
        if add_final_lines and final_levels:
            y2 = max(y2, max(final_levels))
    else:
        y1, y2 = y_min, y_max

    if ylog:
        axins.set_yscale("log")
        y1 = max(y1, cfg.get("zoom_log_eps", 1e-12))

    axins.set_ylim(y1, y2)

    # dashed final lines + (optional) put them as y-ticks
    if add_final_lines and final_levels:
        for yF, col in zip(final_levels, final_colors):
            axins.hlines(yF, x1, x2, colors=col, linestyles="--", linewidth=1.6, alpha=0.9)

        if final_as_ticks:
            yticks = list(axins.get_yticks())
            for yF in final_levels:
                if y1 <= yF <= y2:
                    yticks.append(yF)
            axins.set_yticks(np.unique(np.array(yticks)))

    # formatter
    if pct_xmax is not None:
        axins.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=pct_xmax, decimals=2))

    # style
    axins.grid(alpha=0.25)
    axins.tick_params(axis="both", labelsize=10, length=0)
    axins.set_xticklabels([])
    for lab in axins.get_yticklabels():
        lab.set_fontweight("bold")

    mark_inset(
        ax, axins,
        loc1=cfg.get("loc1", 2),
        loc2=cfg.get("loc2", 4),
        fc="none", ec="black",
        lw=cfg.get("zoom_rect_linewidth", 1),
    )
import matplotlib.pyplot as plt

def save_global_legend(
    legend_path,
    *,
    algo_colors, algo_styles,
    mu_values, mu_markers,
    resource_names, resource_colors,
    user_names, user_markers,
):
    handles = []
    labels  = []

    # Algorithms (style + color)
    for algo in ["OGD", "DA"]:
        ls, _ = algo_styles[algo]
        h = plt.Line2D([0],[0], color=algo_colors[algo], linestyle=ls, linewidth=2)
        handles.append(h)
        labels.append(algo)

    # mu markers
    for mu in mu_values:
        h = plt.Line2D([0],[0], color="black", marker=mu_markers[mu], linestyle="None", markersize=8)
        handles.append(h)
        labels.append(fr"$\mu={mu}$")

    # resources
    for k, nm in enumerate(resource_names):
        h = plt.Line2D([0],[0], color=resource_colors[k], linewidth=4)
        handles.append(h)
        labels.append(fr"Resource $k={nm}$")

    # users
    for i, nm in enumerate(user_names):
        h = plt.Line2D([0],[0], color="black", marker=user_markers[i], linestyle="None", markersize=8)
        handles.append(h)
        labels.append(fr"User $i={nm}$")

    fig_leg = plt.figure(figsize=(11.5, 2.2))
    fig_leg.legend(handles, labels, loc="center", ncol=4, frameon=True,
                   prop={"weight":"bold"}, fontsize=11)
    fig_leg.tight_layout()
    fig_leg.savefig(legend_path, bbox_inches="tight")
    plt.close(fig_leg)


def plot_bids_vs_mu_r(res, betas,alphas,
                           fig_path="figures/bids_vs_mu_r.pdf",
                           legend_path="figures/bids_vs_mu_r_legend.pdf"):

    mu_r = np.array(res["mu_r"])
    Z = np.stack(res["final_bids"], axis=0)  # (M,n,d)

    M, n, d = Z.shape

    linestyles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D", "v"]
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(
        nrows=1, ncols=n,
        figsize=(4.8 * n, 3.2),
        sharey=True
    )

    if n == 1:
        axes = [axes]

    legend_handles = {}

    # -------- main figure (NO legend) --------
    cases_order2 = [str(r"Log - Log"), str(r"Log - Linear"),
                    str(r"Linear - Linear")]

    #mu_np = mu_r_grid.cpu().numpy()

    cases_order3 = {str([1,1]): ["Log", "Log"], str([1,0]): ["Log", "Linear"], str([0,0]): ["Linear", "Linear"], str([0,1]): ["Linear", "Log"],}
    for i in range(n):
        ax = axes[i]

        for k in range(d):
            h, = ax.plot(
                mu_r,
                Z[:, i, k],
                linestyle=linestyles[k % len(linestyles)],
                marker=markers[k % len(markers)],
                linewidth=2,
                markersize=8,
                label=rf"Res. {k+1} ($\beta^{{({k+1})}}{cases_order3[str(alphas)][k]}$)"
            )

            # collect legend entries once
            if h.get_label() not in legend_handles:
                legend_handles[h.get_label()] = h

        ax.set_title(rf"Player ${i+1}$", fontweight="bold")
        ax.set_xlabel(r"Budget heterogeneity $\mu$", fontweight="bold")
        ax.grid(alpha=0.3)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")

        ax.legend(
            frameon=True,
            fontsize=10,
            prop={"weight": "bold"}
        )

    axes[0].set_ylabel(r"Equilibrium bid $z_i^{(k)}$", fontweight="bold")

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # -------- legend only --------
    fig_leg = plt.figure(figsize=(3.2 * d, 1.2))
    fig_leg.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="center",
        ncol=d,
        frameon=True
    )
    fig_leg.tight_layout()
    fig_leg.savefig(legend_path, bbox_inches="tight")
    plt.close(fig_leg)

    print(f"Saved figure to  {fig_path}")
    print(f"Saved legend to  {legend_path}")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter  # **FIX: import**

def plot_bids_alloc_vs_mu_r(
    res, betas, alphas,
    fig_path="figures/bids_alloc_vs_mu_r.pdf",
    legend_path="figures/bids_alloc_vs_mu_r_legend.pdf",

):
    mu_r = np.array(res["mu_r"])
    # color per resource (stable across bid / alloc)
    colors = plt.cm.tab10.colors  # up to 10 resources

    # style by quantity
    BID_STYLE = dict(linestyle="-", linewidth=2.2)
    ALLOC_STYLE = dict(linestyle="--", linewidth=2.2, alpha=0.9)

    # markers only for bids
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    # final_bids: list length M, each (n,d)
    Z = np.stack(res["final_bids"], axis=0)  # (M,n,d)
    M, n, d = Z.shape

    # allocations: list length M, each (n,d)
    X = np.stack(res["final_allocations"], axis=0)  # (M,n,d)

    # SAME linestyle per resource (k)
    linestyles = ["-", "--", ":", "-."]                   # resource k -> linestyle
    #markers = ["o", "s", "^", "D", "v", "P", "X"]         # resource k -> marker

    plt.rcParams.update({"font.size": 12})

    # --- enforce SAME bid y-limits across all players (left axis) ---
    z_min = np.min(Z)
    z_max = np.max(Z)
    pad = 0.03 * (z_max - z_min + 1e-12)
    z_lim = (max(0.0, z_min - pad), z_max + pad)

    # --- enforce SAME allocation y-limits across all players (right axis) ---
    # Option B: keep allocations in [0,1]
    x_lim = (0.0, 0.80)  # **FIX: allocation scale fixed and meaningful**

    fig, axes = plt.subplots(
        1, n, figsize=(4.8 * n, 3.0),
        sharex=True, sharey=True
    )
    if n == 1:
        axes = [axes]

    legend_handles = {}
    right_axes = []

    for i in range(n):
        axL = axes[i]
        axR = axL.twinx()
        right_axes.append(axR)

        # ----- LEFT axis: bids -----
        axL.set_ylim(*z_lim)
        axL.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        # ----- RIGHT axis: allocations -----
        axR.set_ylim(*x_lim)                          # **FIX**
        axR.set_yticks(np.linspace(0.0, 0.80, 6))               # **FIX: clear y-scale**
        axR.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        # **FIX: show right-axis labels (y-scale)**
        axR.tick_params(axis="y", labelright=True, right=True)

        # Optional: reduce clutter (only last panel shows right labels)
        if i != n - 1:
            axR.tick_params(axis="y", labelright=False)  # keep ticks, hide labels
            axR.set_ylabel("")

        # Plot per resource k
        for k in range(d):
            color = colors[k % len(colors)]
            mk = markers[k % len(markers)]

            # ---- BID (solid + marker) ----
            h_bid, = axL.plot(
                mu_r, Z[:, i, k],
                color=color,
                marker=mk,
                markersize=6,
                markevery=max(1, M // 25),
                label=rf"Res. {k + 1} ($\beta^{{({k + 1})}}={betas[k]}$), $z_i^{{({k + 1})}}$",
                **BID_STYLE
            )

            # ---- ALLOCATION (dashed, same color, no marker) ----
            h_alloc, = axR.plot(
                mu_r, X[:, i, k],
                color=color,
                label=rf"Res. {k + 1} ($\beta^{{({k + 1})}}={betas[k]}$), $x_i^{{({k + 1})}}$",
                **ALLOC_STYLE
            )

            # collect legend entries once
            for h in (h_bid, h_alloc):
                if h.get_label() not in legend_handles:
                    legend_handles[h.get_label()] = h

        # cosmetics
        axL.set_title(rf"Player ${i+1}$", fontweight="bold",fontsize=18)
        axL.set_xlabel("")
        axL.grid(alpha=0.3)

        for tick in axL.get_xticklabels() + axL.get_yticklabels():
            tick.set_fontweight("bold")
        for tick in axR.get_xticklabels() + axR.get_yticklabels():
            tick.set_fontweight("bold")

    # shared x label
    fig.text(
        0.5, -0.02,
        r"Budget heterogeneity $\mu$",
        ha="center",
        fontweight="bold",
        fontsize=18
    )

    axes[0].set_ylabel(r"Final bid $z_i^{(k)}$", fontweight="bold",fontsize=18)

    # right label only once (last axis)
    right_axes[-1].set_ylabel(r"Allocation $x_i^{(k)}$", fontweight="bold", fontsize=18)

    # --- save main figure (no legend) ---
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # --- save legend only ---
    fig_leg = plt.figure(figsize=(5.2 * min(d, 3), 2.0))
    fig_leg.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="center",
        ncol=d*2,   # bid vs alloc
        frameon=True,
        fontsize=10,
        prop={"weight": "bold"}
    )
    fig_leg.tight_layout()
    fig_leg.savefig(legend_path, bbox_inches="tight")
    plt.close(fig_leg)

    print(f"Saved figure to  {fig_path}")
    print(f"Saved legend to  {legend_path}")



# ============================================================
# 5) Plot: 3 subplots in one figure (log-log, log-lin, lin-lin)
#    OGD vs DA in each subplot, same config. Save fig + legend separately.
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def plot_residuals_ogd_daq(
    res_by_mu,
    lrMethods,          # e.g. ["OGD", "DAQ"] or ["OGD", "DA"]
    mu_values,          # e.g. [0, 200] (same order as res_by_mu)
    T,
    fig_path="figures/residuals_ogd_daq_mu.pdf",
    legend_path="figures/residuals_ogd_daq_mu_legend.pdf",
):
    """
    res_by_mu: list length M_mu (e.g. 2), where res_by_mu[m][k] is the result for case k:
        k=0 -> log-log, k=1 -> log-lin, k=2 -> lin-lin.
    Each res_by_mu[m][k]["convergence_error"][method] is a (T,) array.

    Plot rule:
      - Same color per algorithm (OGD vs DAQ)
      - Same linestyle per algorithm (OGD '-' , DAQ '--')
      - Different marker per mu value (mu=0 vs mu=200), same color & same linestyle
    """

    cases = ["Log - Log", "Log - Linear", "Linear - Linear"]
    assert len(mu_values) == len(res_by_mu), "mu_values and res_by_mu must have same length"

    # --- styles ---
    algo_color = {"OGD": "tab:blue", "DAQ": "tab:orange"}  # same color per algorithm
    algo_ls    = {"OGD": "-",        "DAQ": "--"}         # same linestyle per algorithm
    color = ["tab:blue", "tab:green", "tab:orange","tab:teal"]
    # markers encode mu (2 values -> 2 markers)
    mu_markers = ["o", "s", "^", "D", "v", "P", "X"]
    mu_to_marker = {mu: mu_markers[i % len(mu_markers)] for i, mu in enumerate(mu_values)}

    # If your method names are not exactly "OGD"/"DAQ", map them here:
    # lrMethods[0] is OGD-like; lrMethods[1] is DA-like
    method_to_algo = {
        lrMethods[0]: "OGD",
        lrMethods[1]: "DAQ",
    }

    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.6), sharex=True, sharey=True)

    legend_handles = {}

    x = np.arange(1, T + 1)

    for k, ax in enumerate(axes):
        color_idx = 0
        for m_idx, mu in enumerate(mu_values):
            res_case = res_by_mu[m_idx][k]  # dict for this case

            for method in lrMethods:
                algo = method_to_algo[method]
                y = np.asarray(res_case["convergence_error"][method])

                # safety if some runs shorter than T
                L = min(T, len(y))
                xx = x[:L]
                yy = y[:L]

                h, = ax.plot(
                    xx, yy,
                  #  color=color[color_idx],
                    linestyle=algo_ls[algo],
                    marker=mu_to_marker[mu],
                    markevery=max(1, L // 18),
                    linewidth=2,
                    markersize=8,
                    label=rf"{algo}, $\mu_r={mu}$",
                )

                lab = h.get_label()
                color_idx += 1
                if lab not in legend_handles:
                    legend_handles[lab] = h

        ax.set_title(cases[k], fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_yscale("log")

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")

    axes[0].set_ylabel(r"Residual", fontweight="bold")
    fig.text(0.5, -0.02, str(r"Time step $(t)$"), ha="center", fontweight="bold")

    # ---- save FIGURE ONLY ----
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # ---- save LEGEND ONLY ----
    fig_leg = plt.figure(figsize=(7.2, 1.4))
    fig_leg.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="center",
        ncol=2*len(lrMethods),              # 2 columns: (OGD,mu=0) (OGD,mu=200) (DAQ,mu=0) (DAQ,mu=200)
        frameon=True,
        prop={"weight": "bold"},
        fontsize=11
    )
    fig_leg.tight_layout()
    fig_leg.savefig(legend_path, bbox_inches="tight")
    plt.close(fig_leg)

    print(f"Saved figure to  {fig_path}")
    print(f"Saved legend to  {legend_path}")





import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


def plot_three_panel_ne_figure_consistent_resource_colors(
    residuals,              # dict: {"OGD": (T,), "DA": (T,)}  (residual curves)
    T,
    z_star,                 # (n,d) NE bids
    Budget,                 # Budget
    x_star,                 # (n,d) NE allocations (shares in [0,1] recommended)
    u_global,               # (n,) global utility per player
    u_local,                # (n,d) local utility per player-resource
    D_mask=None,            # optional (n,d) boolean: True if player i uses resource k
    resource_names=None,    # list length d, e.g. ["CPU","RAM","BW"] or ["1","2","3"]
    player_names=None,      # list length n, e.g. ["1","2","3"] or ["i=1",...]
    fig_path="figures/three_panel_ne.pdf",
    legend_path="figures/three_panel_ne_legend.pdf",
    # Styling knobs
    resource_colors=None,   # list length d; if None use matplotlib defaults cycle
    algo_styles=None,       # dict: algorithm -> (linestyle, marker)
    global_color="0.25",    # color for global utility bars (gray)
    show_resource_legend=False
):
    """
    1x3 figure:
      (1) residuals: compare OGD vs DA
      (2) grouped bars by player i; each resource k has same color across plots
          allocation shown INSIDE bar horizontally as a percentage
      (3) utilities: grouped bars by player i; global + local(k) bars
          local(k) uses same colors as in Panel 2; global uses global_color
    """
    z_star = np.asarray(z_star, dtype=float)
    x_star = np.asarray(x_star, dtype=float)
    u_global = np.asarray(u_global, dtype=float)
    u_local = np.asarray(u_local, dtype=float)

    n, d = z_star.shape
    assert x_star.shape == (n, d)
    assert u_local.shape == (n, d)
    assert u_global.shape == (n,)

    if D_mask is None:
        D_mask = np.ones((n, d), dtype=bool)
    else:
        D_mask = np.asarray(D_mask, dtype=bool)
        assert D_mask.shape == (n, d)

    if resource_names is None:
        resource_names = [f"{k+1}" for k in range(d)]
    if player_names is None:
        player_names = [f"{i+1}" for i in range(n)]
    print(f"D_mask:{D_mask[0,:]}")
    # --- consistent resource colors ---
    if resource_colors is None:
        # grab the default cycle colors deterministically
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        resource_colors = [cycle[(k+2) % len(cycle)] for k in range(d)]
    else:
        assert len(resource_colors) == d

    # --- algorithm styles for panel 1 ---
    if algo_styles is None:
        algo_styles = {
            "OGD": ("-", None),
            "DA": ("--", None),
            "DAQ": ("--", None),
            "DA": ("--", None),
        }

    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.2))
    fontsize = 14

    # =========================
    # Panel 1: Residual curves
    # =========================
    ax = axes[0]
    it = np.arange(1, T + 1)

    for name, y in residuals.items():
        y = np.asarray(y, dtype=float)
        L = min(T, len(y))
        ls, mk = algo_styles.get(name, ("-", None))
        ax.plot(it[:L], y[:L], linestyle=ls, marker=mk, linewidth=2, label=name)

    ax.set_title("(a) Convergence", fontweight="bold", fontsize =fontsize)
    ax.set_xlabel(str(r"Time step $(t)$"), fontweight="bold", fontsize=fontsize)
    ax.set_ylabel(r"Residual", fontweight="bold", fontsize=fontsize)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(frameon=True, prop={"weight": "bold"}, fontsize=10)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    # ==========================================
    # Panel 2: NE bids (grouped by player only)
    # ==========================================
    ax = axes[1]
    ax.set_title("(b) NE bids (allocation in %)", fontweight="bold")

    centers = np.arange(n)
    group_width = 0.9
    bar_w = group_width / d
    offsets = (np.arange(d) - (d - 1) / 2.0) * bar_w

    legend_handles = []
    legend_labels = []
    # bars: one color per resource, repeated across players
    bars_by_k = []
    for k in range(d):
        vals = np.where(D_mask[:, k], z_star[:, k], 0.0)
        b = ax.bar(
            centers + offsets[k],
            vals,
            width=bar_w,
            #label=fr"$k={resource_names[k]}$",
            color=resource_colors[k]
        )
        bars_by_k.append(b)
        # collect legend handle once
        legend_handles.append(b[0])
        legend_labels.append(fr"Resource $k={resource_names[k]}$")

    # allocation text inside bars: horizontal percentage
    # We place it at ~50% of bar height; if very small, we skip
    for i in range(n):
        for k in range(d):
            if not D_mask[i, k]:
                continue
            h = float(z_star[i, k])
            if h <= 0:
                continue
            pct = 100.0 * float(x_star[i, k])
            # Only annotate if bar is not tiny (readability)
            if h < 1e-10:
                continue
            ax.text(
                centers[i] + offsets[k],
                0.72 * h,                      # inside bar
                f"{pct:.0f}%",                 # percentage (0 decimals)
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                rotation=0
            )

    ax.set_xticks(centers)
    ax.set_xticklabels([str(fr"user ${nm}$") for nm in player_names], fontweight="bold", fontsize=fontsize)
    ax.set_ylabel(str(r"Bid $z_i^{(k)\star}$"), fontweight="bold", fontsize=fontsize)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0, top=Budget)


    if show_resource_legend:
        ax.legend(frameon=True, prop={"weight": "bold"}, fontsize=10)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    # ==========================================
    # Panel 3: Utilities (per player only)
    # ==========================================
    ax = axes[2]
    ax.set_title("(c) Utilities at NE", fontsize=15, fontweight="bold")

    # ---- larger fonts, consistent everywhere ----
    LABEL_FS = 14
    TICK_FS = 13
    TEXT_FS = 13

    # --- player colors (distinct from resource colors) ---
    player_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    player_colors = [player_cycle[(k + d + 2) % len(player_cycle)] for k in range(n)]

    # --- hatches for B/W safety ---
    hatches = ["///", "\\\\", "xx", "..", "++", "--", "oo", "**", "||"]
    player_hatches = [hatches[i % len(hatches)] for i in range(n)]

    bars = []
    for i in range(n):
        b = ax.bar(
            centers[i],
            u_global[i],
            width=0.68,
            color=player_colors[i],
           # hatch=player_hatches[i],
            edgecolor="black",
            linewidth=1.2,
        )
        bars.append(b[0])

    # ---- annotate values INSIDE bars (never outside) ----
    ymax = max(1e-12, float(np.max(u_global)))
    for i, b in enumerate(bars):
        h = b.get_height()
        # place text at 70% of bar height (safe even for large fonts)
        ax.text(
            b.get_x() + b.get_width() / 2,
            0.70 * h,
            f"{h:.4g}",
            ha="center",
            va="center",
            fontsize=TEXT_FS,
            fontweight="bold",
            clip_on=True,  # critical: keep text inside axes
        )

    ax.set_xticks(centers)
    ax.set_xticklabels([str(fr"user ${nm}$") for nm in player_names],
                       fontsize=fontsize, fontweight="bold")
    ax.set_ylabel("Utility", fontsize=LABEL_FS, fontweight="bold")

    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.grid(axis="y", alpha=0.3)

    # ---- keep everything visible without pushing text out ----
    ax.set_ylim(bottom=0, top=1.0)

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    # =================================================
    # Save shared legend (Panels 2 & 3)
    # =================================================
    fig_leg = plt.figure(figsize=(6.5, 1.4))
    fig_leg.legend(
        legend_handles,
        legend_labels,
        loc="center",
        ncol=len(legend_labels),
        frameon=True,
        prop={"weight": "bold"},
        fontsize=11,
    )
    fig_leg.tight_layout()
    fig_leg.savefig(legend_path, bbox_inches="tight")
    plt.close(fig_leg)

    print(f"Saved figure  → {fig_path}")
    print(f"Saved legend  → {legend_path}")


import numpy as np

import numpy as np

def _auto_y_limits_percent(y_slices, *, q=(0.02, 0.98), margin=0.15, eps=1e-12):
    """
    y_slices: list of arrays (already sliced to zoom window)
    Returns (y1,y2) in same units as y (NOT multiplied by 100).
    Uses quantiles to avoid spikes + adds margin.
    """
    yy = np.concatenate([np.asarray(v).ravel() for v in y_slices if len(v) > 0])
    yy = yy[np.isfinite(yy)]
    if yy.size == 0:
        return 0.0, 1.0

    lo = np.quantile(yy, q[0])
    hi = np.quantile(yy, q[1])
    if hi <= lo:
        hi = lo + max(abs(lo), eps)

    dy = hi - lo
    y1 = lo - margin * dy
    y2 = hi + margin * dy
    # keep nonnegative for a loss metric
    y1 = max(0.0, y1)
    return y1, y2

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.ticker as mticker

def plot_three_panel_ne_figure_with_effloss(
    residuals,              # dict: {"OGD": (T,), "DA": (T,)} residual curves
    eff_loss,               # dict: {"OGD": (T,), "DA": (T,)} efficiency-loss curves
    T,
    z_star,                 # (n,d) NE bids
    Budget,                 # scalar, y-limit for bids
    x_star,                 # (n,d) NE allocations (shares in [0,1])
    config= None,
    D_mask=None,            # (n,d) bool
    resource_names=None,    # list length d
    player_names=None,      # list length n
    fig_path="figures/three_panel_ne_effloss.pdf",
    legend_path="figures/three_panel_ne_effloss_legend.pdf",
    # Styling
    resource_colors=None,   # list length d
    algo_colors=None,       # dict: algo -> color (shared Panel 1 & 2)
    algo_styles=None,       # dict: algo -> (linestyle, marker)
    show_resource_legend=False,
):
    """
    1x3 figure:
      (a) Residual curves: OGD vs DA
      (b) Efficiency loss over time: same algo colors as (a)
      (c) NE bids grouped by player; resource colors consistent; allocation % inside bars
    Saves: figure + resource legend-only (for panel c).
    """
    z_star = np.asarray(z_star, dtype=float)
    x_star = np.asarray(x_star, dtype=float)
    n, d = z_star.shape
    assert x_star.shape == (n, d)

    if D_mask is None:
        D_mask = np.ones((n, d), dtype=bool)
    else:
        D_mask = np.asarray(D_mask, dtype=bool)
        assert D_mask.shape == (n, d)

    if resource_names is None:
        resource_names = [f"{k+1}" for k in range(d)]
    if player_names is None:
        player_names = [f"{i+1}" for i in range(n)]

    # --- consistent resource colors (panel 3) ---
    if resource_colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        resource_colors = [cycle[(k + 2) % len(cycle)] for k in range(d)]
    else:
        assert len(resource_colors) == d

    # --- consistent algo colors (panel 1 & 2) ---
    if algo_colors is None:
        # keep stable defaults (match your earlier habit)
        algo_colors = {"OGD": "tab:blue", "DA": "tab:orange", "DAQ": "tab:orange", "DA": "tab:orange"}

    # --- algo line styles ---
    if algo_styles is None:
        algo_styles = {
            "OGD": ("-", None),
            "DA": ("--", None),
            "DAQ": ("--", None),
            "DA": ("--", None),
        }

    # --- fonts ---
    fontsize = 13.3
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize ,
        "ytick.labelsize": fontsize ,
        "legend.fontsize": 11,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 3.1))

    it = np.arange(1, T + 1)

    # =========================
    # Panel 1: Residual curves
    # =========================
    ax = axes[0]
    for name, y in residuals.items():
        y = np.asarray(y, dtype=float)
        L = min(T, len(y))
        ls, mk = algo_styles.get(name, ("-", None))
        ax.plot(
            it[:L], y[:L],
            color=algo_colors.get(name, None),
            linestyle=ls, marker=mk,
            linewidth=2,
            label=name
        )
    #ax.set_title("(a) Convergence", fontweight="bold")
    ax.set_xlabel(r"Time step $(t)$", fontweight="bold",fontsize=fontsize+3)
    ax.set_ylabel("Residual", fontweight="bold", fontsize=fontsize+4)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(frameon=True, prop={"weight": "bold"}, loc="best")

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    # ======================================
    # Panel 2: Efficiency loss over time
    # ======================================
    ax = axes[1]
    l =[]
    min_y =[]
    for name, y in eff_loss.items():
        y = np.asarray(y, dtype=float)
        print(name,y[0:10])
        L = min(T, len(y))
        l.append(max(y))
        min_y.append(y[-1])
        ls, mk = algo_styles.get(name, ("-", None))
        ax.plot(
            it[:L], y[:L],
            color=algo_colors.get(name, None),   # SAME as panel 1
            linestyle=ls, #marker=mk,
            #linewidth=2,
            label=name
        )
    #ax.set_title("(b) Efficiency loss", fontweight="bold")
    ax.set_xlabel(str(r"Time step $(t)$"), fontweight="bold", fontsize=fontsize+3)
    ax.set_ylabel(str(r"Efficiency Loss "), fontweight="bold", fontsize=fontsize+4)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))
   # ax.set_yscale("log")  # remove if you prefer linear
    ax.grid(alpha=0.3)
    ax.legend(frameon=True, prop={"weight": "bold"}, loc="best")

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    # ============================
    # Zoom inset for Panel 2
    # ============================
    if config.get("Add_Zoom", False):
        x_idx_min, x_idx_max = config.get("x_zoom_interval", (0, min(T, 200)))
        x_idx_min = max(0, int(x_idx_min))
        x_idx_max = min(int(x_idx_max), T)
        assert x_idx_max > x_idx_min, "Invalid x_zoom_interval"

        y_min, y_max = config.get("y_zoom_interval", (None, None))
        use_auto_y = (y_min is None) or (y_max is None)

        inset_rect = config.get("inset_rect", [0.55, 0.52, 0.42, 0.42])
        axins = ax.inset_axes(inset_rect)

        y_slices = []
        final_levels = []  # store final y[-1] for each curve (within L)
        final_colors = []  # matching colors
        ymax = []

        for name, y in eff_loss.items():
            y = np.asarray(y, dtype=float)
            L = min(T, len(y))
            y_plot = y[:L]

            # zoom window
            j2 = min(x_idx_max, L)
            y_zoom = y_plot[x_idx_min:j2]
            x_zoom = it[x_idx_min:j2]

            y_slices.append(y_zoom)
            ymax.append(max(y_zoom))

            ls, mk = algo_styles.get(name, ("-", None))
            col = algo_colors.get(name, None)

            # main zoom curve
            axins.plot(
                x_zoom,
                y_zoom,
                color=col,
                linestyle=ls,
                marker=mk,
                linewidth=2,
                markersize=5,
            )

            # final value line level: use last available value in y_plot (global final)
            y_final = float(y_plot[-1])
            final_levels.append(y_final)
            final_colors.append(col)

        # x-limits in inset
        x1, x2 = it[x_idx_min], it[x_idx_max - 1]
        axins.set_xlim(x1, x2)

        # y-limits in inset
        if use_auto_y:
            y1, y2 = _auto_y_limits_percent(
                y_slices,
                q=config.get("zoom_quantiles", (0.02, 0.98)),
                margin=config.get("zoom_margin", 0.20),
            )
            # ensure the dashed final lines are visible
            if len(final_levels) > 0:
                y2 = max(y2, max(final_levels))
        else:
            y1, y2 = y_min, y_max

        axins.set_ylim(y1, y2)

        # ---- add dashed horizontal lines for final loss values ----
        # (same color as curve, dashed)


        # percent formatter (set correctly)
        # if rho_t is in [0,1] use xmax=1.0; if already in [0,100], use xmax=100.0
        xmax_pct = config.get("pct_xmax", 1.0)
        axins.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=2))


        axins.grid(alpha=0.25)
        axins.tick_params(axis="both", labelsize=11.5, length=0)
        axins.set_xticklabels([])

        for label in axins.get_yticklabels():
            label.set_fontweight("bold")

        mark_inset(
            ax, axins,
            loc1=2, loc2=4,
            fc="none", ec="black",
            lw=config.get("zoom_rect_linewidth", 1),
        )

    # ==========================================
    # Panel 3: NE bids (grouped by player only)
    # ==========================================
    ax = axes[2]
    #ax.set_title("(c) NE bids (allocation in %)", fontweight="bold")

    centers = np.arange(n)
    group_width = 0.90
    bar_w = group_width / d
    offsets = (np.arange(d) - (d - 1) / 2.0) * bar_w

    legend_handles, legend_labels = [], []
    for k in range(d):
        vals = np.where(D_mask[:, k], z_star[:, k], 0.0)
        b = ax.bar(
            centers + offsets[k],
            vals,
            width=bar_w,
            color=resource_colors[k],
            label = fr"$k={resource_names[k]}$"
        )
        legend_handles.append(b[0])
        legend_labels.append(fr"Resource $k={resource_names[k]}$")

    # Allocation text INSIDE bars (a bit higher than before)
    for i in range(n):
        for k in range(d):
            if not D_mask[i, k]:
                continue
            h = float(z_star[i, k])
            if h <= 1e-12:
                continue
            pct = 100.0 * float(x_star[i, k])
            ax.text(
                centers[i] + offsets[k],
                0.82 * h,          # higher placement inside bar
                f"{pct:.0f}%",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                rotation=0,
                clip_on=True
            )

    ax.set_xticks(centers)
    ax.set_xticklabels([fr"user ${nm}$" for nm in player_names], fontweight="bold")
    ax.set_ylabel(str(r"NE Bids "), fontweight="bold", fontsize=fontsize+4)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=0, top=Budget)

    if show_resource_legend:
        ax.legend(frameon=True, prop={"weight": "bold"}, fontsize=10)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # ==========================
    # Resource legend only
    # ==========================
    fig_leg = plt.figure(figsize=(6.8, 1.45))
    fig_leg.legend(
        legend_handles,
        legend_labels,
        loc="center",
        ncol=len(legend_labels),
        frameon=True,
        prop={"weight": "bold"},
        fontsize=11,
    )
    fig_leg.tight_layout()
    fig_leg.savefig(legend_path, bbox_inches="tight")
    plt.close(fig_leg)

    print(f"Saved figure  → {fig_path}")
    print(f"Saved legend  → {legend_path}")



def plot_ne_bids_vs_budget_by_resource(
    budgets,                 # list/array of c values (increasing)
    zstars_by_budget,        # dict: c -> z_star (n,d)  OR list aligned with budgets
    D_mask,                  # (n,d) bool, True if agent i requires resource k
    resource_names=None,     # list length d (e.g., ["1","2","3"])
    player_names=None,       # list length n (e.g., ["1","2","3"])
    fig_path="figures/ne_bids_vs_budget.pdf",
    legend_path="figures/ne_bids_vs_budget_legend.pdf",
    resource_colors=None,    # list length d; fixed color per resource
):
    """
    For each resource k, plot z_i^{(k)*}(c) vs c for all i who require k.
    - Same color for a resource across all plots.
    - Different marker per agent (consistent across resources).
    - Legend saved separately.
    """

    budgets = np.asarray(budgets, dtype=float)
    assert budgets.ndim == 1 and len(budgets) >= 2

    # ---- get (n,d) from first z_star ----
    if isinstance(zstars_by_budget, dict):
        first = zstars_by_budget[budgets[0]]
    else:
        first = zstars_by_budget[0]
    first = np.asarray(first, dtype=float)
    n, d = first.shape

    D_mask = np.asarray(D_mask, dtype=bool)
    assert D_mask.shape == (n, d)

    if resource_names is None:
        resource_names = [fr"$k={k+1}$" for k in range(d)]
    if player_names is None:
        player_names = [fr"{i+1}" for i in range(n)]

    # ---- fixed resource colors (consistent across all panels) ----
    if resource_colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        resource_colors = [cycle[(k+2) % len(cycle)] for k in range(d)]
    else:
        assert len(resource_colors) == d

    # ---- fixed agent markers (consistent across all panels) ----
    agent_markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    marker_i = {i: agent_markers[i % len(agent_markers)] for i in range(n)}
    # --- enforce SAME bid y-limits across all players (left axis) ---
    z_min = 0.0#np.min(budgets)
    z_max = max([np.max(zstars_by_budget[budgets[i]]) for i in range(len(budgets))])
    pad = 0.03 * (z_max - z_min + 1e-12)
    z_lim = (max(0.0, z_min - pad), z_max + pad)

    # ---- larger text like your recent style ----
    plt.rcParams.update({
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    })

    fig, axes = plt.subplots(1, d, figsize=(5.2 * d, 3.2), sharex=True)
    if d == 1:
        axes = [axes]

    legend_handles = {}
    x = budgets

    for k, ax in enumerate(axes):
        color_k = resource_colors[k]
        if k != 0:
            # Remove Y-axis graduations ONLY
            ax.tick_params(
                axis="y",
                which="both",
                left=False,
                right=False,
                labelleft=False
            )

        # Keep X-axis graduations everywhere
        ax.tick_params(
            axis="x",
            which="major",

        )

        for i in range(n):
            if not D_mask[i, k]:
                continue

            # collect z_i^{(k)*}(c) across budgets
            y = []
            # ----- LEFT axis: bids -----
            ax.set_ylim(*z_lim)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            for c in budgets:
                if isinstance(zstars_by_budget, dict):
                    z_star = np.asarray(zstars_by_budget[c], dtype=float)
                else:
                    idx = np.where(budgets == c)[0][0]
                    z_star = np.asarray(zstars_by_budget[idx], dtype=float)
                y.append(z_star[i, k])
            y = np.asarray(y, dtype=float)


            h_, = ax.plot(
                x, y,
                color="white",  # resource color (marker face color)
                linestyle="-",
                marker=marker_i[i],  # agent marker
                markersize=8,
                linewidth=2,
                markeredgecolor="black",  # 👈 black contour
                markeredgewidth=1.5,  # 👈 thickness of contour
                label=fr"user ${player_names[i]}$"
            )

            lab = h_.get_label()
            if lab not in legend_handles:
                legend_handles[lab] = h_
            h, = ax.plot(
                x, y,
                color=color_k,  # resource color (marker face color)
                linestyle="-",
                marker=marker_i[i],  # agent marker
                markersize=8,
                linewidth=2,
                markeredgecolor="black",  # 👈 black contour
                markeredgewidth=1.5,  # 👈 thickness of contour
                label=fr"user $i={player_names[i]}$"
            )

        ax.set_title(fr"Resource {resource_names[k]}", fontweight="bold")
        ax.set_xlabel(r"Budget $c$", fontweight="bold")
        ax.grid(alpha=0.3)

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        #ax.set_xlabel("")
        #ax.set_xlabel("")
        #ax.grid(alpha=0.3)

    axes[0].set_ylabel(r"NE bid $z_i^{(k)\star}(c)$", fontweight="bold")

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # ---- legend-only file (agents only; resource is encoded by panel color) ----
    fig_leg = plt.figure(figsize=(7.5, 1.4))
    fig_leg.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="center",
        ncol=min(len(legend_handles), 4),
        frameon=True,
        prop={"weight": "bold"},
    )
    fig_leg.tight_layout()
    fig_leg.savefig(legend_path, bbox_inches="tight")
    plt.close(fig_leg)

    print(f"Saved figure  → {fig_path}")
    print(f"Saved legend  → {legend_path}")



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def plot_heterogeneity_three_panel(
    res_by_mu,                # dict mu -> {"residuals":{algo:(T,)}, "eff_loss":{algo:(T,)}, "z_star":(n,d)}
    mu_values=(0,20,40),
    T=3000,
    D_mask=None,              # (n,d) bool
    resource_names=None,      # list length d
    user_names=None,          # list length n
    fig_path="figures/hetero_3panel.pdf",
    legend_path="figures/hetero_legend_resources_users.pdf",
    # styles
    algo_colors=None,         # {"OGD":..., "DA":...}
    algo_styles=None,         # {"OGD":("-",None), "DA":("--",None)}
    mu_markers=None,          # {0:"o", 20:"s", 40:"^"}
    resource_colors=None,     # list length d
    user_markers=None,        # list length n  (panel 3)
    # panel2 zoom config
    zoom_cfg=None,            # dict, see below
    pct_xmax=1.0,             # if eff_loss is in [0,1], else set 100.0
):
    # -------------------------
    # infer sizes / defaults
    # -------------------------
    mu_values = list(mu_values)
    first = res_by_mu[mu_values[0]]
    n, d = np.asarray(first["z_star"]).shape

    if D_mask is None:
        D_mask = np.ones((n,d), dtype=bool)
    else:
        D_mask = np.asarray(D_mask, dtype=bool)

    if resource_names is None:
        resource_names = [str(k+1) for k in range(d)]
    if user_names is None:
        user_names = [str(i+1) for i in range(n)]

    if algo_colors is None:
        algo_colors = {"OGD":"tab:blue", "DA":"tab:orange"}
    if algo_styles is None:
        algo_styles = {"OGD":("-", None), "DA":("--", None)}

    if mu_markers is None:
        base = ["o","s","^","D","v","P","X"]
        mu_markers = {mu: base[i % len(base)] for i, mu in enumerate(mu_values)}

    if resource_colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        resource_colors = [cycle[(k+2) % len(cycle)] for k in range(d)]

    if user_markers is None:
        base = ["o","s","^","D","v","P","X"]
        user_markers = [base[i % len(base)] for i in range(n)]

    if zoom_cfg is None:
        zoom_cfg = {"Add_Zoom": False}

    it = np.arange(1, T+1)

    # -------------------------
    # figure layout
    # -------------------------
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 3.6))

    # =========================
    # Panel 1: Residuals
    # =========================
    ax = axes[0]
    for mu in mu_values:
        curves = res_by_mu[mu]["residuals"]
        for algo in ["OGD", "DA"]:
            y = np.asarray(curves[algo], dtype=float)
            L = min(T, len(y))
            ls, _ = algo_styles[algo]
            ax.plot(
                it[:L], y[:L],
                color=algo_colors[algo],
                linestyle=ls,
                marker=mu_markers[mu],
                markevery=max(1, L//18),
                linewidth=2,
                markersize=7,
                label=fr"{algo}, $\mu={mu}$",
            )

    ax.set_title("(a) Convergence", fontweight="bold")
    ax.set_xlabel(r"Iterations $t$", fontweight="bold")
    ax.set_ylabel("Residual", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(frameon=True, prop={"weight":"bold"}, ncol=2, loc="best")

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    add_zoom_inset(
        axes[0], it, curves_dict=panel1_curves_for_zoom,
        cfg=zoom1_cfg,
        algo_colors=algo_colors,
        algo_styles=algo_styles,
        mu_markers=mu_markers,
        ylog=True,  # residuals typically log
        pct_xmax=None,
        add_final_lines=False
    )

    # =========================
    # Panel 2: Efficiency loss
    # =========================
    ax = axes[1]
    for mu in mu_values:
        curves = res_by_mu[mu]["eff_loss"]
        for algo in ["OGD", "DA"]:
            y = np.asarray(curves[algo], dtype=float)
            L = min(T, len(y))
            ls, _ = algo_styles[algo]
            ax.plot(
                it[:L], y[:L],
                color=algo_colors[algo],
                linestyle=ls,
                marker=mu_markers[mu],
                markevery=max(1, L//18),
                linewidth=2,
                markersize=7,
                label=fr"{algo}, $\mu={mu}$",
            )

    ax.set_title("(b) Efficiency loss", fontweight="bold")
    ax.set_xlabel(r"Iterations $t$", fontweight="bold")
    ax.set_ylabel(r"Loss $\rho_t$", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=pct_xmax, decimals=0))
    ax.grid(alpha=0.3)
    ax.legend(frameon=True, prop={"weight":"bold"}, ncol=2, loc="best")

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    # ---- zoom inset on panel 2 (optional) ----
    if zoom_cfg.get("Add_Zoom", False):
        x_idx_min, x_idx_max = zoom_cfg.get("x_zoom_interval", (int(0.7*T), T))
        x_idx_min = max(0, int(x_idx_min))
        x_idx_max = min(int(x_idx_max), T)
        inset_rect = zoom_cfg.get("inset_rect", [0.56, 0.50, 0.40, 0.45])
        axins = ax.inset_axes(inset_rect)

        final_levels = []
        y_slices = []
        x1, x2 = it[x_idx_min], it[x_idx_max-1]

        for mu in mu_values:
            curves = res_by_mu[mu]["eff_loss"]
            for algo in ["OGD", "DA"]:
                y = np.asarray(curves[algo], dtype=float)
                L = min(T, len(y))
                y_plot = y[:L]
                j2 = min(x_idx_max, L)
                y_zoom = y_plot[x_idx_min:j2]
                x_zoom = it[x_idx_min:j2]
                y_slices.append(y_zoom)

                ls, _ = algo_styles[algo]
                axins.plot(
                    x_zoom, y_zoom,
                    color=algo_colors[algo],
                    linestyle=ls,
                    marker=mu_markers[mu],
                    markersize=5,
                    linewidth=2
                )

                y_final = float(y_plot[-1])
                final_levels.append((y_final, algo_colors[algo]))

        # auto y-limits from data in window
        yy = np.concatenate([v for v in y_slices if len(v)>0])
        yy = yy[np.isfinite(yy)]
        if yy.size == 0:
            y1, y2 = 0.0, 1.0
        else:
            lo = np.quantile(yy, zoom_cfg.get("q_lo", 0.02))
            hi = np.quantile(yy, zoom_cfg.get("q_hi", 0.98))
            if hi <= lo:
                hi = lo + max(abs(lo), 1e-12)
            margin = zoom_cfg.get("margin", 0.25)
            dy = hi - lo
            y1 = max(0.0, lo - margin*dy)
            y2 = hi + margin*dy

        # ensure dashed final lines visible
        if final_levels:
            y2 = max(y2, max(v for v,_ in final_levels))

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        # dashed final lines + force ticks at those values
        yticks = list(axins.get_yticks())
        for yF, col in final_levels:
            axins.hlines(yF, x1, x2, colors=col, linestyles="--", linewidth=1.6, alpha=0.9)
            if y1 <= yF <= y2:
                yticks.append(yF)
        axins.set_yticks(np.unique(np.array(yticks)))
        axins.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=pct_xmax, decimals=1))

        axins.grid(alpha=0.25)
        axins.set_xticklabels([])
        axins.tick_params(axis="both", length=0, labelsize=10)
        for lbl in axins.get_yticklabels():
            lbl.set_fontweight("bold")

        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black",
                   lw=zoom_cfg.get("zoom_rect_linewidth", 1))

    # =========================
    # Panel 3: NE bids vs mu
    # =========================
    ax = axes[2]

    # build arrays: z_star(mu) -> shape (M,n,d)
    Z = np.stack([np.asarray(res_by_mu[mu]["z_star"], dtype=float) for mu in mu_values], axis=0)  # (M,n,d)

    # plot each demanded (i,k) as a curve across mu
    for k in range(d):
        for i in range(n):
            if not D_mask[i, k]:
                continue
            ax.plot(
                mu_values,
                Z[:, i, k],
                color=resource_colors[k],        # resource color
                marker=user_markers[i],          # user marker
                linewidth=2,
                markersize=8,
            )

    ax.set_title(r"(c) NE bids vs heterogeneity $\mu$", fontweight="bold")
    ax.set_xlabel(r"Heterogeneity $\mu$", fontweight="bold")
    ax.set_ylabel(r"Bid $z_i^{(k)\star}(\mu)$", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_xticks(mu_values)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    # ---- create a single legend file for panel 3 (resources + users) ----
    # Resource legend handles
    res_handles = [plt.Line2D([0],[0], color=resource_colors[k], lw=3) for k in range(d)]
    res_labels  = [fr"Resource $k={resource_names[k]}$" for k in range(d)]
    # User legend handles
    usr_handles = [plt.Line2D([0],[0], color="black", marker=user_markers[i], lw=0, markersize=9)
                   for i in range(n)]
    usr_labels  = [fr"User $i={user_names[i]}$" for i in range(n)]

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    fig_leg = plt.figure(figsize=(9.2, 1.8))
    fig_leg.legend(res_handles + usr_handles,
                   res_labels + usr_labels,
                   loc="center",
                   ncol=max(d, n),
                   frameon=True,
                   prop={"weight":"bold"},
                   fontsize=11)
    fig_leg.tight_layout()
    fig_leg.savefig(legend_path, bbox_inches="tight")
    plt.close(fig_leg)

    print(f"Saved figure  → {fig_path}")
    print(f"Saved legend  → {legend_path}")

# ===========================
# Example call
# ===========================

test = False
if __name__ == "__main__" and test:
    alphas_vec = cfg["alphas_vec"]
    mu_values = cfg["mu_r"]
    res = []
    for mu in mu_values:
        res1 = []
        cfg["mu"] = mu
        for alpha_ in alphas_vec:
            cfg["alpha"] = alpha_
            runner = SimulationRunner(cfg)
            res1.append(runner.run_simulation_learning_rules())
        res.append(res1)
    plot_residuals_ogd_daq(
        res, cfg["lrMethods"], mu_values, cfg["T"],
        fig_path="figures/residuals_ogd_daq.pdf",
        legend_path="figures/residuals_ogd_daq_legend.pdf",
    )


test = True
if __name__ == "__main__" and test:
    config = {
        "Add_Zoom": True,
        "x_zoom_interval": (0, 300),
        "y_zoom_interval": (None, None),  # auto
        "zoom_quantiles": (0.05, 0.95),
        "zoom_margin": 0.20,
        "zoom_log_eps": 1e-12,
        "inset_rect": [0.55, 0.50, 0.42, 0.45],
    }

    runner = SimulationRunner(cfg)
    res =runner.run_simulation_learning_rules()
    # residuals from your runs
    residuals = {
        "OGD": res["convergence_error"]["OGD"],  # (T,)
        "DA": res["convergence_error"]["DA"],  # (T,)
    }

    Loss = {
        "OGD": res["loss"]["OGD"],  # (T,)
        "DA": res["loss"]["DA"],  # (T,)
    }

    # NE objects (you said you already have them)
    # z_star: (n,d), x_star: (n,d), u_global: (n,), u_local: (n,d)
    z_star = res["final_bids"]["DA"] # (M,n,d)
    print(f"{res["convergence_iter"]["OGD"]}:")

    print(f"{res["convergence_iter"]["DA"]}:")
    # allocations: list length M, each (n,d)
    x_star = res["final_allocations"]["DA"]  # (n,d)

    u_global = res["utilities"]["DA"]
    u_local =  res["d_utilities"]["DA"]

    plot_three_panel_ne_figure_with_effloss(
        residuals=residuals,
        eff_loss = Loss,
        T=cfg["T"],
        z_star=z_star,
        Budget=cfg["c"] - cfg["mu"],
        x_star=x_star,
        config = config,
       # u_global=u_global,
       # u_local=u_local,
        D_mask=cfg["Y"],
        resource_names=["1", "2", "3"],
        #player_names=["1", "2", "3","4"],
       # fig_path="figures/logloglog_3panel.pdf",
    )

test = False
if __name__ == "__main__" and test:
    runner = SimulationRunner(cfg)
    res = runner.run_simulation_budget()

    betas = cfg["beta"]
    alphas = cfg["alpha"]

    plot_bids_alloc_vs_mu_r(res, betas, alphas)


test = False
if __name__ == "__main__" and test:
    alphas_vec = cfg["alphas_vec"]
    mu_values = cfg["mu_r"]
    res = []
    budgets = []
    zstars_by_budget ={}
    for mu in mu_values:
        res1 = []
        cfg["mu"] = mu
        budget = cfg["c"] - mu
        budgets.append(budget)

        runner = SimulationRunner(cfg)
        res = runner.run_simulation_learning_rules()
        zstars_by_budget[budget] = res["final_bids"][cfg["lrMethods"][0]]

    plot_ne_bids_vs_budget_by_resource(
        budgets=budgets,
        zstars_by_budget=zstars_by_budget,
        D_mask=cfg["Y"],
        resource_names=["1", "2", "3"],
        player_names=["1", "2", "3","4"],
        fig_path="figures/bids_vs_budget.pdf",
        legend_path="figures/bids_vs_budget_legend.pdf",
    )

test = False
if __name__ == "__main__" and test:
    #alphas_vec = cfg["alphas_vec"]
    mu_values = cfg["mu_r"]
    res = []
    mu_values = [0,20,40]
    cfg["c"] = 100
    cfg["mu"] = 0
    res_by_mu = {
    }
    for mu in mu_values:
        res1 = []
        cfg["gamma"] = mu
        runner = SimulationRunner(cfg)
        res = runner.run_simulation_learning_rules()
        res_by_mu[mu] = {"residuals": {"OGD": res["convergence_error"]["DA"], "DA": res["convergence_error"]["OGD"]},
            "eff_loss": {"OGD":  res["loss"]["OGD"], "DA":  res["loss"]["DA"]},
            "eff_loss": {"OGD":  res["loss"]["OGD"], "DA":  res["loss"]["DA"]},
            "z_star":res["final_bids"]["DA"]}


    plot_heterogeneity_three_panel(
        res_by_mu,                # dict mu -> {"residuals":{algo:(T,)}, "eff_loss":{algo:(T,)}, "z_star":(n,d)}
        mu_values=mu_values,
        T=cfg["T"],
        D_mask=cfg["Y"],   )