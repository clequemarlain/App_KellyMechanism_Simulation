import streamlit as st
#from tabulate import tabulate
from collections import defaultdict
import csv
import torch, csv
from collections import defaultdict
#from tabulate import tabulate

#from config import SIMULATION_CONFIG_table as config

def run_simulation(config, GameKelly):
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
    Hybrid_funcs = config["Hybrid_funcs"]
    Hybrid_sets = config["Hybrid_sets"]
    list_n = config["list_n"]
    list_gamma = config["list_gamma"]
    total_runs = len(list_gamma) * len(list_n) * len(lrMethods)
    run_counter = 1

    results = []

    for gamma in list_gamma:
        for n in list_n:

            c_min = epsilon
            d_min = 0

            a_vector = torch.tensor([max(a - i * gamma, a_min) for i in range(n)], dtype=torch.float64)
            c_vector = torch.tensor([max(c - i * mu, c_min) for i in range(n)], dtype=torch.float64)
            dmin = a_vector * torch.log((epsilon + torch.sum(c_vector) - c_vector + delta) / epsilon)
            d_vector = 0.7 * dmin *0

            for lrMethod in lrMethods:
                print(f"[{run_counter}/{total_runs}] gamma={gamma}, n={n}, method={lrMethod}")
                run_counter += 1

                eps = epsilon * torch.ones(n)
                bid0 = torch.ones(n)
                d_vector = torch.zeros(n)
                if lrMethod == "Hybrid":
                    Hybrid_sets = [list(range(0, 2)), list(range(2, int(n)))]

                game = GameKelly(n, price, eps, delta, alpha, tol)
                Bids, Welfare, Utility_set, error_NE_set = game.learning(
                    lrMethod, a_vector, c_vector, d_vector, T, eta, bid0,
                    vary=config["lr_vary"], Hybrid_funcs=Hybrid_funcs, Hybrid_sets=Hybrid_sets
                )
                min_error = torch.min(error_NE_set)
                nb_iter = int(torch.argmin(error_NE_set).item()) if min_error <= tol else float('inf')

                results.append({
                    "gamma": gamma,
                    "n": n,
                    "method": lrMethod,
                    "iterations": nb_iter
                })
    return results

import io
import csv
from collections import defaultdict

def display_results_streamlit(results, config,save_path):
    lrMethods = config["lrMethods"]

    # Organize results
    table_data = defaultdict(lambda: defaultdict(dict))
    for row in results:
        gamma = row["gamma"]
        n = row["n"]
        method = row["method"]
        iters = row["iterations"]
        table_data[gamma][n][method] = iters

    # Build rows
    rows = []
    headers = ["gamma", "n"] + lrMethods

    def fmt(val):
        return "∞" if val == float("inf") else str(val)

    for gamma in sorted(table_data.keys()):
        for n in sorted(table_data[gamma].keys()):
            row = [gamma, n]
            for lrMethod in lrMethods:
                time = table_data[gamma][n].get(lrMethod, "---")
                row.append(fmt(time))
            rows.append(row)

    # Streamlit display
    st.write("### Comparison of Convergence Time")
    st.table([headers] + rows)

    # Prepare CSV in memory for download
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer, delimiter=";")
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
    csv_data = csv_buffer.getvalue()
    csv_buffer.close()

    # Streamlit download button
    st.download_button(
        label="⬇️ Download Table as CSV",
        data=csv_data,
        file_name="table_results.csv",
        mime="text/csv"
    )