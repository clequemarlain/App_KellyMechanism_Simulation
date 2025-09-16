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

def run_simulation_table_avg(config, GameKelly):
    """
    Run simulations for different gamma, n, and learning methods,
    average convergence iterations over Nb_random_sim simulations.
    Returns a dictionary results[gamma][n][method] = avg_iterations
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

    results = {}  # dict to store results

    for gamma in list_gamma:
        results[gamma] = {}
        for n in list_n:
            results[gamma][n] = {}

            a_vector = torch.tensor([max(a - i * gamma, a_min) for i in range(n)], dtype=torch.float64)
            c_vector = torch.tensor([max(c - i * mu, epsilon) for i in range(n)], dtype=torch.float64)
            dmin = a_vector * torch.log((epsilon + torch.sum(c_vector) - c_vector + delta) / epsilon)
            d_vector = 0.7 * dmin * 0

            for lrMethod in lrMethods:
                iterations_list = []

                for sim in range(Nb_random_sim):
                    eps = epsilon * torch.ones(n)
                    bid0 = bid0 = (c - epsilon) * torch.rand(n) + a
                    Hybrid_funcs, Hybrid_sets = [], []

                    lrMethod2 = lrMethod
                    if lrMethod == "Hybrid":
                        idx = 0
                        subset = random.sample(range(n), int(config["Nb_A1"][idx]))
                        remaining = [i for i in range(n) if i not in subset]
                        Hybrid_sets = [subset, remaining]
                        Hybrid_funcs = config["Hybrids"][idx]["Hybrid_funcs"]
                        lrMethod2 = f"(A1: {config['Nb_A1'][idx]}, A2: {n - config['Nb_A1'][idx]})"
                        idx += 1

                    game = GameKelly(n, price, eps, delta, alpha, tol)
                    Bids, Welfare, Utility_set, error_NE_set = game.learning(
                        lrMethod, a_vector, c_vector, d_vector, T, eta, bid0,
                        vary=config["lr_vary"], Hybrid_funcs=Hybrid_funcs, Hybrid_sets=Hybrid_sets, stop=True
                    )

                    min_error = torch.min(error_NE_set)
                    nb_iter = int(torch.argmin(error_NE_set).item()) if min_error <= tol else float('inf')
                    iterations_list.append(nb_iter)


                # Average over Nb_random_sim
                avg_iterations = np.mean([i for i in iterations_list if i != float('inf')]) \
                    if any(i != float('inf') for i in iterations_list) else float('inf')

                results[gamma][n][lrMethod2] = avg_iterations

    return results


import io
import csv
from collections import defaultdict

def display_results_streamlit_dict(results, config, save_path=None):
    """
    Display a dictionary-style result (results[gamma][n][method] = avg_iterations) in Streamlit as a table.
    """
    list_gamma = config["list_gamma"]
    list_n = config["list_n"]
    lrMethods = config["lrMethods"]

    # Build headers
    headers = ["gamma", "n"] + lrMethods

    # Build table rows
    table_rows = []
    for gamma in sorted(list_gamma):
        for n in sorted(list_n):
            row = [gamma, n]
            for method in lrMethods:
                # Handle hybrid method labels if present
                #method_keys = [k for k in results[gamma][n].keys() if method in k] if method == "Hybrid" else [method]
                metric = results[gamma][n][method]
                #if method_keys:
                #    metric = np.mean([results[gamma][n][k] for k in method_keys])
                #else:
                #    metric = "---"

                # Format ∞
                if isinstance(metric, float) and np.isinf(metric):
                    metric_str = f"<{config["T"]}"#"∞"
                else:
                    metric_str = str(metric)
                row.append(metric_str)
            table_rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(table_rows, columns=headers)

    # Streamlit display
    st.write("### 📊 Comparison of Convergence Time (averaged)")
    st.dataframe(df, use_container_width=True)

    # Prepare CSV for download
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, sep=";", index=False)
    csv_data = csv_buffer.getvalue()
    csv_buffer.close()

    st.download_button(
        label="⬇️ Download Table as CSV",
        data=csv_data,
        file_name="table_results_avg.csv",
        mime="text/csv"
    )
