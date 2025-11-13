import streamlit as st
#from tabulate import tabulate
from collections import defaultdict
import csv
import torch, csv, random
import pandas as pd
import numpy as np

from src.game.utils import *
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
    epsilon = torch.tensor(config["epsilon"])
    alpha = config["alpha"]
    tol = config["tol"]
    a_min = config["a_min"]

    lrMethods = config["lrMethods"]
    list_n = config["list_n"]
    list_gamma = config["list_gamma"]
    Nb_random_sim = config["Nb_random_sim"]

    results = {}  # dict to store results
    results_epsNE = {}  # dict to store results

    # Réinitialiser les placeholders
    progress_bar = st.progress(0)
    status_text = st.empty()

    for sim_gamma, gamma in enumerate(list_gamma):
        results[gamma] = {}
        results_epsNE[gamma] = {}
        for n in list_n:
            results[gamma][n] = {}
            results_epsNE[gamma][n] = {}

            a_vector = torch.tensor([max(a - i * gamma, a_min) for i in range(n)], dtype=torch.float64)
            c_vector = torch.tensor([max(c - i * mu, epsilon) for i in range(n)], dtype=torch.float64)
            dmin = a_vector * torch.log((epsilon + torch.sum(c_vector) - c_vector + delta) / epsilon)
            d_vector = 0.7 * dmin * 0
            idx = 0
            idx_rmfq =0
            bid0 = (c - epsilon) * torch.rand(n) + epsilon

            s_min = (n - 1) * epsilon + config["delta"]
            s_max = (n - 1) * c + config["delta"]
            z_max = BR_alpha_fair(epsilon, c_vector, bid0, s_min, a_vector, delta, alpha, price, b=0)
            x_max = z_max / (z_max + s_min)

            # z_min = BR_alpha_fair(eps, c_vector, bid0, s_max, a_vector, delta, alpha, price, b=0)
            x_min = epsilon / (epsilon + s_max)
            x_min_2 = c_vector / (c_vector + s_max)

            Payoff_min = torch.min(Payoff(x_min, epsilon, a_vector, d_vector, alpha, price),
                                   Payoff(x_min_2, c_vector, a_vector, d_vector, alpha, price))
            Payoff_max = torch.max(Payoff(x_max, z_max, a_vector, d_vector, alpha, price))

            for idxMthd, lrMethod in enumerate(lrMethods):
                iterations_list = []
                minError_list = []

                iterations_list_epsNE = []
                minError_list_epsNE = []
                copy_keys={}


                Hybrid_funcs, Hybrid_sets = [], []

                lrMethod2 = lrMethod
                if lrMethod == "Hybrid":
                    NbHybrid = NbHybrid + 1

                    # subset = random.sample(range(self.config["n"]), int(self.config["Nb_A1"][idx]))
                    # remaining = [i for i in range(self.config["n"]) if i not in subset]
                    # if self.config["Random_set"] and NbHybrid == 1:
                    #     Hybrid_sets = [subset, remaining]
                    # else:
                    Hybrid_sets = config["Hybrid_sets"][NbHybrid - 1]
                    Hybrid_funcs = config["Hybrid_funcs"][NbHybrid - 1]

                    lrMethod2 = f"({Hybrid_funcs[0]}: {config['Nb_A1'][NbHybrid - 1]}, {Hybrid_funcs[1]}: {n - config['Nb_A1'][NbHybrid - 1]})"
                    key = tuple(Hybrid_funcs + ["Hybrid"])
                    if key not in copy_keys:
                        copy_keys[lrMethod2] = key
                    idx += 1

                if lrMethod == "RRM":
                    lrMethod2 = f"RRM_{config["RRM_lr"][idx_rmfq]}"
                    idx_rmfq += 1
                elif lrMethod != "SBRD":  # self.config["num_lrmethod"]!=0:
                    key = lrMethod
                    if key not in copy_keys:
                        copy_keys[lrMethod] = (key)

                    #lrMethod2 = rf"{lrMethod} -- $\eta={config["Learning_rates"][idxMthd]}$"
                    # print(f"lrMethod2:{lrMethod2}")
                    idx += 1
                for sim in range(Nb_random_sim):
                    eps = epsilon * torch.ones(n)

                    if not config["keep_initial_bid"]:
                        bid0 = (c - epsilon) * torch.rand(n) + epsilon

                    game_set = GameKelly(n, price, torch.tensor(epsilon), delta, alpha, tol, payoff_min=Payoff_min, payoff_max=Payoff_max)

                    Bids, Welfare, Utility_set, error_NE_set = game_set.learning(
                        lrMethod, a_vector, c_vector, d_vector, T, config["Learning_rates"][idxMthd], bid0,
                        vary=config["lr_vary"], Hybrid_funcs=Hybrid_funcs, Hybrid_sets=Hybrid_sets,
                    )


                    min_error = torch.min(error_NE_set)

                    nb_iter = int(torch.argmin(error_NE_set).item()) if min_error <= tol else float('inf')
                    iterations_list.append(nb_iter)
                    minError_list.append(min_error)


                    min_error_epsNE = torch.min(Utility_set[4])

                    nb_iter_epsNE = int(torch.argmin(Utility_set[4]).item()) if min_error_epsNE <= tol else float('inf')
                    iterations_list_epsNE.append(nb_iter_epsNE)
                    minError_list_epsNE.append(min_error_epsNE)


                # Average over Nb_random_sim
                avg_iterations = np.mean([i for i in iterations_list if i != float('inf')]) \
                    if any(i != float('inf') for i in iterations_list) else float('inf')
                avg_iterations_epsNE = np.mean([i for i in iterations_list_epsNE if i != float('inf')]) \
                    if any(i != float('inf') for i in iterations_list_epsNE) else float('inf')

                results[gamma][n][lrMethod2] = avg_iterations
                results[gamma][n][lrMethod2+"error"] = np.mean(minError_list)

                results_epsNE[gamma][n][lrMethod2] = avg_iterations_epsNE
                results_epsNE[gamma][n][lrMethod2+"error"] = np.mean(minError_list_epsNE)
        # Mise à jour progression
        progress = (sim_gamma + 1) /len(list_gamma)
        progress_bar.progress(progress)

        # Mise à jour compteur i/N
        status_text.text(f"Simulation {sim_gamma + 1}/{list_gamma}")

    return results, results_epsNE


import io
import csv
from collections import defaultdict

def display_results_streamlit_dict(results, config, save_path=None):
    """
    Display a dictionary-style result (results[gamma][n][method] = avg_iterations) in Streamlit as a table.
    """
    list_gamma = config["list_gamma"]
    list_n = config["list_n"]
    lrMethods = config["LEGENDS"]

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
                if isinstance(metric, float) and np.isinf(metric):
                    metric_str = results[gamma][n][method+ "error"]#f"<{config["T"]}"#"∞"
                else:
                    metric_str = metric
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
