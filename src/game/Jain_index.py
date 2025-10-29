import streamlit as st
#from tabulate import tabulate
from collections import defaultdict
import csv
import torch, csv, random
import pandas as pd
import numpy as np
from collections import defaultdict
#from tabulate import tabulate

#from config import SIMULATION_CONFIG_table as config

def run_jain_vs_gamma(config, GameKelly):
    """
    For each n in config['list_n'] and each gamma in an auto range,
    run SBRD Nb_random_sim times, and store the average Jain index of the
    FINAL bids. Returns (results, list_gamma).

    results[n][gamma] = avg_jain_index \in [0,1]
    """
    import torch
    import numpy as np
    import streamlit as st

    # --- Unpack config
    T        = config["T"]
    a        = config["a"]
    a_min    = config["a_min"]
    mu       = config["mu"]
    c        = config["c"]
    delta    = config["delta"]
    epsilon  = config["epsilon"]
    alpha    = config["alpha"]
    tol      = config["tol"]
    price    = config["price"]
    list_n   = config["list_n"]
    Nb_sim   = int(config["Nb_random_sim"])
    vary_lr  = bool(config["lr_vary"])
    keep_init= bool(config.get("keep_initial_bid", False))
    eta      = float(config.get("eta", 0.1))  # step used only by non-BR algos, not needed for SBRD
    lrMethod = "SBRD"  # as requested

    # Reasonable gamma grid: 0 .. floor(a/max(n)) inclusive
    list_gamma = list(range(0, 1 + int(a / max(list_n))))

    # --- Output dict
    results = {}  # results[n][gamma] = avg_jain

    # UI progress
    total_tasks = len(list_n) * len(list_gamma)
    done = 0
    progress_bar = st.progress(0)
    status_text = st.empty()

    # --- Helper: Jain index (works on torch tensors)
    def jain_index_vec(bids_vec):
        # J(x) = (sum x)^2 / (n * sum x^2)
        z = bids_vec
        if not torch.is_floating_point(z):
            z = z.to(torch.float64)
        n = z.numel()
        s = z.sum()
        s2 = (z**2).sum()
        denom = n * s2.clamp(min=1e-12)
        return torch.clamp((s*s)/denom, 0.0, 1.0)

    # --- Main loops
    for n in list_n:
        results[n] = {}

        # Pre-allocate epsilon vector + initial bid
        eps_vec = torch.tensor(epsilon, dtype=torch.float64)
        bid0 = (c - epsilon) * torch.rand(n, dtype=torch.float64) + epsilon

        for gamma in list_gamma:
            # Build heterogeneous vectors
            a_vec = torch.tensor([max(a - i * gamma, a_min) for i in range(n)], dtype=torch.float64)
            c_vec = torch.tensor([max(c - i * mu, epsilon)  for i in range(n)], dtype=torch.float64)
            dmin  = a_vec * torch.log((epsilon + torch.sum(c_vec) - c_vec + delta) / epsilon)
            d_vec = 0.0 * dmin  # as in your code

            # Run several sims, average Jain of final bids
            jain_vals = []
            for _ in range(Nb_sim):
                if not keep_init:
                    bid0 = (c - epsilon) * torch.rand(n, dtype=torch.float64) + epsilon

                game = GameKelly(n, price, eps_vec, delta, alpha, tol)

                # learning returns (Bids, Welfare, Utility_set, error_NE_set)
                # We only need final bids to compute Jain index
                Bids, Welfare, Utility_set, error_NE_set = game.learning(
                    lrMethod, a_vec, c_vec, d_vec, T, eta, bid0,
                    vary=vary_lr, Hybrid_funcs=[], Hybrid_sets=[]
                )

                final_bids = Bids[0][-1]        # shape (n,)
                jain_vals.append(jain_index_vec(final_bids).item())

            results[n][gamma] = float(np.mean(jain_vals))

            # Progress UI
            done += 1
            progress_bar.progress(done / total_tasks)
            status_text.text(f"Jain index sims: n={n}, γ={gamma}  [{done}/{total_tasks}]")

    return results, list_gamma

import plotly.graph_objects as go

def plot_jain_vs_gamma(results, list_gamma, cfg):
    """
    results[n][gamma] = avg_jain in [0,1]
    Plots Jain(gamma) for each n in results.
    """
    fig = go.Figure()
    color_cycle = cfg.get("colors", [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
    ])

    for i, (n, gamma_map) in enumerate(sorted(results.items(), key=lambda kv: kv[0])):
        y = [gamma_map.get(g, None) for g in list_gamma]
        fig.add_trace(go.Scatter(
            x=list_gamma,
            y=y,
            mode="lines+markers",
            name=f"n={n}",
            line=dict(width=3, color=color_cycle[i % len(color_cycle)]),
            marker=dict(size=8, line=dict(width=1, color="black"))
        ))

    fig.update_layout(
        title="Jain’s Fairness vs γ (per n)",
        xaxis_title="γ",
        yaxis_title="Jain index",
        yaxis=dict(range=[0,1], tickformat=".2f"),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2)
    )
    return fig
