"""
Heterogeneous (mixed-population) dynamics figure, replacing the
ambiguous original Fig. 5 / Fig. 6.

Addresses Reviewer 1, point 2 ("curves are not explicitly labeled") by
giving every curve an explicit, algorithm-named legend entry instead
of relying on marker shape alone, and by putting the NE reference line
in the same legend rather than as a bare dashed line.

Usage
-----
    from fig_heterogeneous import run_and_plot_heterogeneous_dynamics

    run_and_plot_heterogeneous_dynamics(
        cfg, algo_A1="BR", algo_A2="DA",
        alpha_A1_values=[0.10, 0.80, 0.90],
        fig_path="figures/fig5_BR_vs_DA.pdf",
    )
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import SIMULATION_CONFIG as cfg_default
from main import build_vectors, init_bid, to_torch
from utils import GameKelly, Q_simplex, Valuation_matrix, Payoff_matrix


ALGO_LABELS = {
    "BR": "Best Response (BR)",
    "DA": "Dual Averaging (DA)",
    "OGD": "Online Gradient Descent (OGD)",
}


def _run_mixed_population(cfg, algo_A1, algo_A2, alpha_A1, T, seed=0):
    """
    Runs one Hybrid simulation with a fraction alpha_A1 of agents on
    algo_A1 (always including agent 0) and the rest on algo_A2
    (always including agent 1, so it is comparable across runs).
    Returns per-time-step bids, instantaneous payoffs (all agents),
    the NE reference (via a long DA run at fixed budgets), agent
    index used to represent each algorithm, and the fairness traces.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = cfg["n"]
    d = len(cfg["alpha"])
    alpha = cfg["alpha"]
    beta = torch.as_tensor(cfg["beta"], dtype=torch.float64)
    price = cfg["price"]
    epsilon = cfg["epsilon"]
    delta = cfg["delta"]
    tol = cfg["tol"]
    eta = cfg["eta"]
    c = cfg["c"]
    Y = cfg["Y"] if torch.is_tensor(cfg["Y"]) else torch.as_tensor(cfg["Y"], dtype=torch.float64)
    Y = Y[:n, :d]

    eps = to_torch(epsilon)
    a_vector, c_vector, d_vector = build_vectors(
        n=n, a=cfg["a"], a_min=cfg["a_min"], gamma=cfg.get("gamma", 0.0),
        c=c, c_min=epsilon, mu=cfg.get("mu", 0.0),
        epsilon=eps, delta=delta, dtype=torch.float64, device="cpu",
    )
    A_matrix = (beta[None, :] * a_vector[:, None]).to(torch.float64)

    n_A1 = max(1, min(n - 1, int(round(alpha_A1 * n))))
    subset_A1 = list(range(n_A1))          # includes agent 0
    subset_A2 = list(range(n_A1, n))       # includes agent 1 if n_A1 < 2

    bid0 = init_bid(n, d, c, epsilon, dtype=torch.float64)
    bid0 = Q_simplex(bid0, epsilon, c_vector, Y) * Y

    game = GameKelly(n, d, beta, price, eps, delta, alpha, tol, Y,
                      payoff_min=1, payoff_max=2)

    Bids, Welfare, Utility, error = game.learning(
        "Hybrid", A_matrix, c_vector, d_vector, T, eta, bid0,
        Hybrid_funcs=[algo_A1, algo_A2], Hybrid_sets=[subset_A1, subset_A2],
    )
    traj = Bids[0]           # (T+1, n, d)
    jain_trace = Bids[2]     # (T+1,)

    # NE reference: homogeneous DA run at the same budgets, long horizon
    game_ne = GameKelly(n, d, beta, price, eps, delta, alpha, tol, Y,
                         payoff_min=1, payoff_max=2)
    Bids_ne, *_ = game_ne.learning(
        "DA", A_matrix, c_vector, d_vector, max(500, T), eta, bid0,
    )
    z_ne = Bids_ne[0][-1]

    # instantaneous, normalized payoff per agent at every t
    x_traj = traj / (traj.sum(dim=1, keepdim=True) + delta)  # (T+1, n, d)
    payoff_traj = torch.stack([
        Payoff_matrix(x_traj[t], traj[t], A_matrix, d_vector, alpha, price, Y)
        for t in range(traj.shape[0])
    ])  # (T+1, n)
    p_min, p_max = payoff_traj.min(), payoff_traj.max()
    payoff_norm = (payoff_traj - p_min) / (p_max - p_min + 1e-12)

    x_ne = z_ne / (z_ne.sum(dim=0, keepdim=True) + delta)
    ne_payoff = Payoff_matrix(x_ne, z_ne, A_matrix, d_vector, alpha, price, Y)
    ne_payoff_norm = (ne_payoff - p_min) / (p_max - p_min + 1e-12)

    return {
        "bids": traj.numpy(),                      # (T+1, n, d)
        "payoff": payoff_norm.numpy(),              # (T+1, n)
        "jain": jain_trace.numpy(),                 # (T+1,)
        "rep_A1": 0,
        "rep_A2": subset_A2[0],
        "ne_bid": float(z_ne[0, 0]),
        "ne_payoff": float(ne_payoff_norm[0]),
        "n_A1": n_A1,
        "n_A2": n - n_A1,
    }


def run_and_plot_heterogeneous_dynamics(
    cfg=None, *, algo_A1="BR", algo_A2="DA",
    alpha_A1_values=(0.10, 0.80, 0.90),
    T=300, resource_idx=0, seed=0,
    fig_path="figures/fig5_heterogeneous.pdf",
    legend_path=None,
    fontsize=13, figsize_per_col=3.6,
):
    """
    Produces a (2 rows x len(alpha_A1_values) cols) figure:
    row 0 = instantaneous normalized payoff over time for one
            representative agent per algorithm,
    row 1 = bid over time for the same two agents,
    one column per alpha_A1 mixture level.

    Every curve carries an explicit "<Algorithm> (agent i)" legend
    entry (shared across the whole figure, drawn once), plus the NE
    reference level as a labeled dashed line -- this directly answers
    Reviewer 1's complaint that Fig. 5's curves were not identifiable.
    """
    cfg = dict(cfg_default if cfg is None else cfg)
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
    if legend_path is None:
        base, ext = os.path.splitext(fig_path)
        legend_path = f"{base}_legend{ext}"

    ncols = len(alpha_A1_values)
    fig, axes = plt.subplots(2, ncols, figsize=(figsize_per_col * ncols, 6.4), sharex="col")
    if ncols == 1:
        axes = axes.reshape(2, 1)

    color_A1, color_A2, color_ne = "tab:red", "tab:blue", "black"

    for col, alpha_A1 in enumerate(alpha_A1_values):
        res = _run_mixed_population(cfg, algo_A1, algo_A2, alpha_A1, T, seed=seed)
        t = np.arange(res["payoff"].shape[0])

        ax_pay = axes[0, col]
        ax_pay.plot(t, res["payoff"][:, res["rep_A1"]], color=color_A1, lw=2)
        ax_pay.plot(t, res["payoff"][:, res["rep_A2"]], color=color_A2, lw=2)
        ax_pay.axhline(res["ne_payoff"], color=color_ne, ls="--", lw=1.5)
        ax_pay.set_title(fr"$\alpha_{{{algo_A1}}}={int(round(alpha_A1*100))}\%$"
                          f"  (n={res['n_A1']} vs n={res['n_A2']})", fontsize=fontsize)
        ax_pay.grid(alpha=0.3)
        if col == 0:
            ax_pay.set_ylabel("Inst. Payoff", fontsize=fontsize, fontweight="bold")

        ax_bid = axes[1, col]
        ax_bid.plot(t, res["bids"][:, res["rep_A1"], resource_idx], color=color_A1, lw=2)
        ax_bid.plot(t, res["bids"][:, res["rep_A2"], resource_idx], color=color_A2, lw=2)
        ax_bid.axhline(res["ne_bid"], color=color_ne, ls="--", lw=1.5)
        ax_bid.set_xlabel("Time step (t)", fontsize=fontsize, fontweight="bold")
        ax_bid.grid(alpha=0.3)
        if col == 0:
            ax_bid.set_ylabel("Bid", fontsize=fontsize, fontweight="bold")

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # ---- explicit, shared legend saved separately ----
    handles = [
        plt.Line2D([0], [0], color=color_A1, lw=3),
        plt.Line2D([0], [0], color=color_A2, lw=3),
        plt.Line2D([0], [0], color=color_ne, lw=2, ls="--"),
    ]
    labels = [
        f"{ALGO_LABELS.get(algo_A1, algo_A1)} agent",
        f"{ALGO_LABELS.get(algo_A2, algo_A2)} agent",
        "Nash equilibrium (NE)",
    ]
    fig_leg = plt.figure(figsize=(7.5, 1.2))
    fig_leg.legend(handles, labels, loc="center", ncol=3, frameon=True,
                    prop={"weight": "bold"}, fontsize=fontsize)
    fig_leg.savefig(legend_path, bbox_inches="tight")
    plt.close(fig_leg)

    print(f"Saved figure  -> {fig_path}")
    print(f"Saved legend  -> {legend_path}")
    return fig_path, legend_path


if __name__ == "__main__":
    cfg = dict(cfg_default)
    cfg["n"] = 6
    cfg["alpha"] = [1]
    cfg["beta"] = [1]
    cfg["Y"] = torch.ones((cfg["n"], 1), dtype=torch.float64)
    cfg["c"] = 40.0
    cfg["mu"] = 0.0
    cfg["gamma"] = 0.0

    run_and_plot_heterogeneous_dynamics(
        cfg, algo_A1="BR", algo_A2="DA",
        alpha_A1_values=[0.10, 0.80, 0.90],
        T=150,
        fig_path="figures/fig5_BR_vs_DA.pdf",
    )
