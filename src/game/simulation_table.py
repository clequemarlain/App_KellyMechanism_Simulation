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

    lrMethods = config["lrMethods"]
    Hybrid_funcs = config["Hybrid_funcs"]
    list_n = config["list_n"]
    list_gamma = config["list_gamma"]
    total_runs = len(list_gamma) * len(list_n) * len(lrMethods)
    run_counter = 1

    results = []

    for gamma in list_gamma:
        for n in list_n:
            set1 = torch.arange(n, dtype=torch.long)
            nb_hybrid = len(Hybrid_funcs)
            Hybrid_sets = torch.chunk(set1, nb_hybrid)
            a_vector = torch.tensor([a / (i + 1) ** gamma for i in range(n)], dtype=torch.float64)
            c_vector = torch.tensor([c / (i + 1) ** mu for i in range(n)], dtype=torch.float64)

            for lrMethod in lrMethods:
                print(f"[{run_counter}/{total_runs}] gamma={gamma}, n={n}, method={lrMethod}")
                run_counter += 1

                eps = epsilon * torch.ones(n)
                bid0 = torch.ones(n)
                d_vector = torch.zeros(n)

                game = GameKelly(n, price, eps, delta, alpha, tol)
                bids, lsw, error = game.learning(
                    lrMethod, a_vector, c_vector, d_vector,
                    T, eta, bid0, vary=False,
                    Hybrid_funcs=Hybrid_funcs,
                    Hybrid_sets=Hybrid_sets
                )

                min_error = torch.min(error)
                nb_iter = int(torch.argmin(error).item()) if min_error <= tol else float('inf')

                results.append({
                    "gamma": gamma,
                    "n": n,
                    "method": lrMethod,
                    "iterations": nb_iter
                })
    return results

import pandas as pd

def display_results_streamlit(results, cfg, save_path="results/table_results.csv"):
    st.subheader("Simulation Results")

    # Ensure results is a DataFrame
    if isinstance(results, pd.DataFrame):
        df = results
    elif isinstance(results, dict):
        df = pd.DataFrame([results])
    elif isinstance(results, list):
        try:
            df = pd.DataFrame(results)
        except Exception as e:
            st.error(f"Cannot convert results list to DataFrame: {e}")
            return
    else:
        st.error(f"Unsupported results type: {type(results)}")
        return

    # Save results safely
    try:
        df.to_csv(save_path, index=False)
        st.success(f"Results saved to {save_path}")
    except Exception as e:
        st.warning(f"Could not save results: {e}")

    # Show table in Streamlit
    st.dataframe(df)

    # Show cfg info
    st.subheader("Configuration")
    if isinstance(cfg, dict):
        st.json(cfg)
    else:
        st.write(cfg)

