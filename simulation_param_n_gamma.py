import streamlit as st
#from tabulate import tabulate
from collections import defaultdict
import csv
import torch, csv, random
import pandas as pd
import numpy as np
from collections import defaultdict

from lxml.html.builder import LEGEND

#from tabulate import tabulate
from src.game.utils import *

#from config import SIMULATION_CONFIG_table as config

def run_simulation_n_gamma(config, GameKelly):
    """
    Run simulations for different gamma, n, and learning methods,
    average convergence iterations and metrics over Nb_random_sim runs.

    Returns:
        results[gamma][n][method] = {
            'convergence_iter': avg,
            'final_bids': avg,
            'SW': avg,
            'LSW': avg,
            'Dist_To_Optimum_SW': avg,
            ...
        }
    """
    T = config["T"]
    eta = config["eta"]
    price = config["price"]
    a = config["a"]
    mu = config["mu"]
    c = config["c"]
    delta = config["delta"]
    epsilon = config["epsilon"]
    alpha = config["alpha"]
    tol = config["tol"]
    a_min = config["a_min"]

    lrMethods = config["lrMethods"]
    list_n = config["list_n"]
    list_gamma = config["list_gamma"]
    Nb_random_sim = config["Nb_random_sim"]

    results = {}

    for gamma in list_gamma:
        results[gamma] = {}
        for n in list_n:
            results[gamma][n] = {}

            a_vector = torch.tensor([max(a - i * gamma, a_min) for i in range(n)], dtype=torch.float64)
            c_vector = torch.tensor([max(c - i * mu, epsilon) for i in range(n)], dtype=torch.float64)
            dmin = a_vector * torch.log((epsilon + torch.sum(c_vector) - c_vector + delta) / epsilon)
            d_vector = 0.7 * dmin * 0

            # Optimum
            eps = epsilon * torch.ones(n)
            bid0 = (c - epsilon) * torch.rand(n) + a
            x_log_optimum = x_log_opt(c_vector, a_vector, d_vector, eps, delta, price, bid0)
            Valuation_log_opt = Valuation(x_log_optimum, a_vector, d_vector, alpha)
            SW_opt = torch.sum(Valuation_log_opt)
            LSW_opt = torch.sum(torch.minimum(Valuation_log_opt, c_vector)).detach().numpy()

            for lrMethod in lrMethods:
                metrics_accumulator = defaultdict(list)

                for sim in range(Nb_random_sim):
                    eps = epsilon * torch.ones(n)
                    bid0 = (c - epsilon) * torch.rand(n) + a

                    game = GameKelly(n, price, eps, delta, alpha, tol)
                    Bids, Welfare, Utility_set, error_NE_set = game.learning(
                        lrMethod, a_vector, c_vector, d_vector, T, eta, bid0,
                        vary=config["lr_vary"], Hybrid_funcs=None, Hybrid_sets=None, stop=True
                    )

                    min_error = torch.min(error_NE_set)
                    nb_iter = int(torch.argmin(error_NE_set).item()) if min_error <= tol else float('inf')

                    SocialWelfare = Welfare[0]
                    Distance2optSW = 1 / n * torch.abs(SW_opt - Welfare[0])
                    LSW = Welfare[1]

                    # Collect metrics
                    metrics_accumulator['Speed'].append(error_NE_set.detach().numpy())
                    metrics_accumulator["convergence_iter"].append(nb_iter)
                    metrics_accumulator["SW"].append(SocialWelfare.detach().numpy())
                    metrics_accumulator["LSW"].append(LSW.detach().numpy())
                    metrics_accumulator["Dist_To_Optimum_SW"].append(Distance2optSW.detach().numpy())
                    metrics_accumulator["final_bids"].append(Bids[0][-1].detach().numpy())

                # Moyenne des métriques
                averaged_metrics = {}
                for k, v_list in metrics_accumulator.items():
                    try:
                        print(f"v_list:{v_list}")
                        averaged_metrics[k] = np.mean(v_list)
                    except Exception:
                        averaged_metrics[k] = np.mean(v_list)

                results[gamma][n][lrMethod] = averaged_metrics

    return results

import plotly.express as px

import plotly.graph_objects as go

def plot_results_multi_gamma_go(cfg, metric="SW"):
    """
    One plot per learning method with multiple gamma curves (manual go.Scatter).
    X-axis = n, Y-axis = metric
    """
    st.write(f"## 📈 Metric: {metric}")

    results = run_simulation_n_gamma(cfg, GameKelly)

    lrMethods = list(next(iter(next(iter(results.values())).values())).keys())
    colors = cfg.get("colors", ["blue", "red", "green", "orange", "purple", "brown", "cyan"])
    plot_step = cfg.get("plot_step", 1)

    for lrMethod in lrMethods:
        fig = go.Figure()
        x_data_lrMethod, y_data_lrMethod = [], []
        LEGENDS = []

        for j, (gamma, n_dict) in enumerate(results.items()):
            if gamma == "methods":
                continue

            x_data, y_data = [], torch.zeros((len(cfg["list_n"])))
            k=0
            for n, methods in sorted(n_dict.items()):
                if lrMethod in methods and metric in methods[lrMethod]:
                    x_data.append(n)
                    y_data[k] = methods[lrMethod][metric][-1]
                    k+=1

            fig.add_trace(go.Scatter(
                x=x_data[::plot_step],
                y=y_data[::plot_step],
                mode="lines+markers",
                name=f"γ={gamma}",
                line=dict(color=colors[j % len(colors)], width=3)
            ))
            x_data_lrMethod.append(x_data)
            y_data_lrMethod.append(y_data)
            LEGENDS.append(f"γ={gamma}")

        fig.update_layout(
            title=f"{metric} vs n for {lrMethod}",
            xaxis_title="n",
            yaxis_title=metric,
            legend=dict(title="Gamma", orientation="h", y=-0.2),
            template="plotly_white",
            height=500
        )

        # ---- Génération du plot via matplotlib (plotGame) ----
        save_to = f"{metric}_{lrMethod}"
        figpath_plot, figpath_legend = plotGame(
            x_data_lrMethod[0],
            y_data_lrMethod,
            "n",
            metric,
            LEGENDS,
            saveFileName=save_to,
            ylog_scale=cfg["ylog_scale"],
            pltText=False,
            step=cfg["plot_step"]
        )

        # ---- Affichage image matplotlib ----
        #st.image(figpath_plot, caption=f"{metric} vs n for {lrMethod}", use_column_width=True)

        # ---- Boutons de téléchargement ----
        with open(figpath_plot, "rb") as f:
            st.download_button(
                label=f"⬇️ Download Plot ({lrMethod})",
                data=f,
                file_name=f"{save_to}.pdf",
                mime="application/pdf"
            )
        with open(figpath_legend, "rb") as f:
            st.download_button(
                label=f"⬇️ Download Legend ({lrMethod})",
                data=f,
                file_name=f"{save_to}_legend.pdf",
                mime="application/pdf"
            )

        # ---- Affichage Plotly ----
        st.plotly_chart(fig, use_container_width=True)

   # return x_data_gen, y_data_gen