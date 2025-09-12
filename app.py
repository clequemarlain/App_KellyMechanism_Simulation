import json, io
import numpy as np
import torch,ast, random,time
import streamlit as st
from pdf2image import convert_from_path
from PIL import Image
import plotly.graph_objects as go
from main import SimulationRunner

from src.game.utils import *
from src.game.config import SIMULATION_CONFIG as DEFAULT_CONFIG
from src.game.config import SIMULATION_CONFIG_table as DEFAULT_CONFIG_TABLE
from src.game.description import ALGO_DESCRIPTIONS
from src.game.simulation_table import run_simulation_table_avg, display_results_streamlit_dict
from simulation_param_n_gamma import *

# -----------------------
# PAGE CONFIG & HEADER
# -----------------------
st.set_page_config(
    page_title="Bidding – Kelly Games",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

col1, col2 = st.columns([1, 5])
with col1:
    st.image("src/game/logo_avgn.png")  # Avignon Univ logo
with col2:
    st.image("src/game/cognetslogo1.png")  # Project logo

st.title("🎮 Bidding Simulator (α-fair Kelly Mechanism)")
st.markdown("""
Simulate and analyze different learning algorithms in a **Kelly Mechanism**-type game setting.
Configure the parameters in the sidebar and view the results live.
""")

# -----------------------
# INTRO GIF (PDF → GIF)
# -----------------------
images1 = convert_from_path('src/game/kellyMechanism-Journal-Page-1.drawio.pdf', dpi=100)
images2 = convert_from_path('src/game/kellyMechanism-Journal-Page-2.drawio.pdf', dpi=100)
images3 = convert_from_path('src/game/kellyMechanism-Journal-Page-3.drawio.pdf', dpi=100)
images4 = convert_from_path('src/game/kellyMechanism-Journal-Page-4.drawio.pdf', dpi=100)

# Collect single-page images
img_list = [images1[0], images2[0], images3[0], images4[0]]

gif_bytes = io.BytesIO()
img_list[0].save(
    gif_bytes,
    format='GIF',
    save_all=True,
    append_images=img_list[1:],
    duration=3*1000, # 3 seconds
    loop=0
)
gif_bytes.seek(0)
st.image(gif_bytes)

# -----------------------
# SIDEBAR CONFIG
# -----------------------


with st.sidebar:

    st.header("⚙️ Configuration")
    cfg = dict(DEFAULT_CONFIG)

    cfg["n"] = st.number_input("Players (n)", 2, 100, cfg["n"], step=1)
    cfg["T"] = st.number_input("Iterations (T)", 10, 10000, cfg["T"], step=10) #st.sidebar.slider("Iterations (T)", 10, 10000, cfg["T"], step=10)
    cfg["Nb_random_sim"] = st.number_input("Number of random simulations", 1, 50, int(cfg["Nb_random_sim"]), step=1)
    cfg["alpha"] = st.sidebar.selectbox("α (fairness)", [0, 1, 2], index=[0, 1, 2].index(cfg["alpha"]))
    cfg["eta"] = st.sidebar.number_input("Learning rate (η)", 0.0001, 5.0, float(cfg["eta"]), step=0.1, format="%.4f")
    cfg["lr_vary"] = st.checkbox(
        "Vary learning rate during training?",
        value=cfg["lr_vary"],  # default = cfg["lr_vary"]
        help="Check this box to enable learning rate variation."
    )
    cfg["keep_initial_bid"] = st.checkbox(
        f"Keep the same initial bid for all {cfg['Nb_random_sim']} simulations.",
        value=False,  # default = cfg["lr_vary"]
       # help="Check this box to enable learning rate variation."
    )

    cfg["price"] = st.sidebar.number_input("Price", 0.0001, 1000.0, float(cfg["price"]), step=0.1, format="%.4f")

    with st.sidebar.expander("Advanced Parameters"):
        cfg["a"] = st.number_input("a (base utility scale)", 0.1, 1e6, float(cfg["a"]), step=10.0, format="%.4f")
        cfg["a_min"] = st.number_input("minimum a (base utility scale)", 0.1, 1e6, float(cfg["a_min"]), step=1.0, format="%.4f")
        cfg["mu"] = st.number_input("μ (c heterogeneity)", 0.0, 4.0, float(cfg["mu"]), step=0.1)
        cfg["c"] = st.number_input("c (budget base)", 0.0001, 1e6, float(cfg["c"]), step=10.0, format="%.4f")
        cfg["delta"] = st.number_input("δ (slack)", 0.0, 10.0, float(cfg["delta"]), step=0.1)
        cfg["epsilon"] = st.number_input("ε (min bid)", 1.0, 1.0*1e2, float(cfg["epsilon"]), step=0.5, format="%.6f")
        cfg["tol"] = st.number_input("Tolerance", 1e-9, 1e-2, float(cfg["tol"]), step=1e-6, format="%.9f")
        cfg["gamma"] = st.number_input("γ (a_i heterogeneity)", 0.0, 100.0, float(cfg["gamma"]), step=0.1)

        cfg["a_vector"] = st.text_area(
            "List of heterogeneous values a_i",
            value=str([max(cfg["a"] - cfg["gamma"] * i, cfg['a_min']) for i in range(cfg["n"])]),
            help="a_i heterogeneity values, e.g: [10,20] for 2 players"
        )

        try:
            # Essayons de parser directement comme une liste Python
            cfg["a_vector"] = ast.literal_eval(cfg["a_vector"])
            if not isinstance(cfg["a_vector"], list):
                raise ValueError
            cfg["a_vector"] = [int(x) for x in cfg["a_vector"]]
        except Exception:
            st.error("Invalid format for a_vector. Please enter a list, e.g. [10, 20, 30].")

        # Convert input string to list of integers
        # Range of number of players
        cfg["list_n"] = st.text_area(
            "List of players (list_n)",
            value=", ".join(str(x) for x in cfg.get("list_n", DEFAULT_CONFIG_TABLE["list_n"])),
            help="Comma-separated list of numbers of players to simulate."
        )
        try:
            cfg["list_n"] = [int(x.strip()) for x in cfg["list_n"].split(",") if x.strip()]
        except:
            st.error("Invalid format for list_n, please enter integers separated by commas.")

        # Range of gamma (heterogeneity)
        cfg["list_gamma"] = st.text_area(
            "List of γ values (list_gamma)",
            value=", ".join(str(x) for x in cfg.get("list_gamma", DEFAULT_CONFIG_TABLE["list_gamma"])),
            help="Comma-separated list of γ (a_i heterogeneity) values."
        )
        # Convert input string to list of floats
        try:
            cfg["list_gamma"] = [float(x.strip()) for x in cfg["list_gamma"].split(",") if x.strip()]
        except:
            st.error("Invalid format for list_gamma, please enter numbers separated by commas.")

    # --- Learning methods selection ---
    lr_methods_all = ["DAQ", "OGD", "SBRD", "NumSBRD", "DAE", "XL", "Hybrid"]

    selected_methods = st.multiselect(
        "Select learning methods",
        lr_methods_all,
        default=["DAQ", "SBRD"]
    )

    cfg["lrMethods"] = selected_methods

    # --- Hybrids configuration ---
    cfg["Hybrids"] = []  # will store multiple hybrid configs
    LEGENDS = [m for m in selected_methods if m != "Hybrid"]

    if "Hybrid" in selected_methods:
        st.info("You selected Hybrid. You can configure multiple hybrid algorithms below.")

        # Number of hybrids
        num_hybrids = st.number_input(
            "How many Hybrid algorithms do you want to configure?",
            min_value=1,
            max_value=10,
            value=1,
            step=1
        )

        hybrid_options = [m for m in lr_methods_all if m != "Hybrid"]

        #selected_methods =  [m for m in selected_methods if m != "Hybrid"]

        h_idx = 1
        # Initialise la liste des k si pas déjà définie
        if "Nb_A1" not in cfg:
            cfg["Nb_A1"] = []
        else:
            cfg["Nb_A1"] = cfg["Nb_A1"]

        for h in range(num_hybrids):
            h_idx += 1
            st.markdown(f"#### ⚙️ Hybrid #{h + 1}")

            if h > 0:
                cfg["lrMethods"].append("Hybrid")

            # --- Select funcs ---
            funcs = st.multiselect(
                f"Select Hybrid funcs for Hybrid #{h + 1}",
                hybrid_options,
                default=hybrid_options[:2],
                key=f"hybrid_funcs_{h}"
            )

            # --- Select k ---
            k = st.number_input(
                f"Number of players in first subset for Hybrid #{h + 1}",
                min_value=1,
                max_value=cfg["n"],
                value=2,
                step=1,
                key=f"hybrid_k_{h}"  # ✅ unique par Hybrid
            )

            # Stocker ce k dans la liste
            cfg["Nb_A1"].append(int(k))

            # Légende avec le k correspondant
            LEGENDS.append(f"(A1: {k}, A2: {cfg["n"] - k})")

            # --- Generate random sets ---
            subset = random.sample(range(cfg["n"]), int(k))
            remaining = [i for i in range(cfg["n"]) if i not in subset]
            cfg["Hybrid_sets"] = [subset, remaining]

            # --- Let user edit sets manually ---
            sets_str = st.text_area(
                f"Hybrid sets for Hybrid #{h + 1} (JSON list of lists)",
                value=json.dumps(cfg["Hybrid_sets"]),
                key=f"hybrid_sets_{h}",
                help="e.g. [[0,1],[2,3]]"
            )

            try:
                sets = json.loads(sets_str)
            except Exception:
                st.error(f"Invalid format for Hybrid_sets #{h + 1}, please enter a valid JSON list of lists")
                sets = []

            # --- Save Hybrid config ---
            cfg["Hybrids"].append({
                "Hybrid_funcs": funcs,
                "Hybrid_sets": sets
            })

    metrics_all = ["Utility", "Bid", "Speed", "SW", "LSW", "Dist_To_Optimum_SW","Agg_Bid", "Agg_Utility"]
    cfg["metric"] = st.sidebar.selectbox("Metric to plot", metrics_all, index=metrics_all.index(cfg["metric"]))

    cfg["ylog_scale"] = st.sidebar.checkbox("Y log scale", value=cfg["ylog_scale"])
    cfg["plot_step"] = st.number_input("Plot step", 1, 1000, int(cfg["plot_step"]), step=1)

    cfg["pltText"] = st.sidebar.checkbox("Display values", value=cfg["pltText"])


    st.sidebar.download_button("⬇️ Download config JSON", data=json.dumps(cfg, indent=2),
                               file_name="config.json", mime="application/json")


# -----------------------
# DESCRIPTION & FORMULA
# -----------------------
selected_algo = st.selectbox("Choose algorithm to describe", list(ALGO_DESCRIPTIONS.keys()))
if selected_algo != "None":
    with st.expander(f"Description of {selected_algo}", expanded=False):
        st.code(ALGO_DESCRIPTIONS[selected_algo], language='python')

if st.checkbox("Show Formulations"):
    st.latex(r"""
    \varphi_i^{\alpha}(x_i) =
    \begin{cases}
    
    a_i\frac{x_i^{1-\alpha}}{1-\alpha} - \lambda z_i, & \alpha \neq 1 \\
    a_i\log(x_i) - \lambda z_i, & \alpha = 1
    \end{cases}
    \quad , \quad
    a_i = a - i\gamma
    \quad, \quad x_i =\frac{z_i}{\sum_{j=1}^n z_j + \delta}
    """)

    st.markdown(r"""
    **Where:**  
    - \($x_i$\)  allocated resource share for player $i$  
    - \($\alpha \ge 0$\) is the fairness parameter  
    - \($a$\) is the base utility scale  
    - \($\gamma \ge 0$\) controls heterogeneity across players
    - \($\lambda$\) the price
    """)


# Checkbox to show metrics info
if st.checkbox("ℹ️ Show information about metrics"):
    st.markdown("""
    ### 📊 Metrics Used in the Simulator  

    - **[Social Welfare (SW)](https://en.wikipedia.org/wiki/Social_welfare_function)**:  
      The aggregate efficiency of the allocation, defined as the sum of agents’ utilities at each iteration.  

    - **Distance to Optimum Social Welfare ($\\text{Dist2SW}^*$):**  
      Since the SW maximization is a concave optimization problem, we solve the KKT conditions via a bisection algorithm to obtain the optimal $\\text{SW}^*$.  
      The distance is:  
      $$
      \\text{Dist2SW}^*(\\mathbf{z}) = \\big| \\text{SW}^* - \\text{SW}(\\mathbf{z}) \\big|
      $$  

    - **Speed:**  
      A performance indicator based on the **Convergence Residual**, defined as the $\\ell_2$-distance  
      $$
      \\|\\text{BR}(\\mathbf{z}(t)) - \\mathbf{z}(t)\\|_2,
      $$  
      which measures how close the system is to a Nash equilibrium.  
      This value decreases as the algorithm converges, and thresholds below $10^{-5}$ are typically treated as equilibrium.
    
    - **[Utility](https://en.wikipedia.org/wiki/Utility)**: A measure of individual satisfaction or payoff.  
    - **[Bid](https://en.wikipedia.org/wiki/Auction)**: The amount an agent submits as demand for resources.  


    - **Average Bids:**  
      The long-run time-averaged bid per agent, as an indicator demand and budget usage.  
      $$
      \\frac{1}{T} \\sum_{t=1}^T \\mathbf{z}_i(t)),
      $$  

    - **Average Utility:**  
      The long-run time-averaged utility per agent:  
      $$
      \\frac{1}{T} \\sum_{t=1}^T \\varphi_i(\\mathbf{z}(t)),
      $$  
      which highlights fairness and satisfaction across players.  
    """)
# -----------------------
# RUN SIMULATION
# -----------------------
if st.button("▶️ Run Simulation"):
    # Conteneur pour le chrono
    chrono_placeholder = st.empty()

    start_time = time.time()

    with st.spinner("Simulating..."):
        # --- Chrono en temps réel ---
        while True:
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
       #     chrono_placeholder.info(f"⏱️ Elapsed time: {minutes:02d}:{seconds:02d}")

            # Ici tu lances la simulation
            runner = SimulationRunner(cfg)
            results = runner.run_simulation()
            break  # on sort de la boucle une fois la simulation finie

    # --- Stop chrono ---
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)

    # Stocker les résultats dans la session
    st.session_state.results = results
    st.session_state.config = cfg

    # Affichage final
    chrono_placeholder.success(f"✅ Simulation finished in {minutes:02d}:{seconds:02d}")

    # -----------------------
    # PLOTLY VISUALISATION
    # -----------------------

def convert_results_to_csv(results):
    # Fonction pour convertir les résultats en CSV
    # Implémentation simplifiée
    return "Simulation,Results,Would,Be,Here\n1,2,3,4,5"


# Affichage des résultats si disponibles
if 'results' in st.session_state:
    results = st.session_state.results
    config = st.session_state.config

    st.header("Simulation Results")

    # Métriques de performance
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of players", config['n'])
    with col2:
        st.metric("Iterations", config['T'])
    with col3:
        st.metric("Parameter  alpha", config['alpha'])

    # Préparation des données pour les graphiques
    x_data = np.arange(config['T'])
    y_data = []
    legends = []
    for method in LEGENDS:

        #if method == "Hybrid":
        if method in results['methods']:
            if cfg["metric"] == "Speed":
                y_data.append(results['methods'][method]['Speed'])
            elif cfg["metric"] == "LSW":
                y_data.append(results['methods'][method]['LSW'])
            elif cfg["metric"] == "Dist_To_Optimum_SW":
                y_data.append(results['methods'][method]['Dist_To_Optimum_SW'])
            elif cfg["metric"] == "SW":
                y_data.append(results['methods'][method]['SW'])
                print("ffdghjhdf", LEGENDS, len(y_data), method)
            elif cfg["metric"] == "Bid":
                y_data.append(results['methods'][method]['Bid'])
            elif cfg["metric"] == "Agg_Bid":
                y_data.append(results['methods'][method]['Agg_Bid'])
            elif cfg["metric"] == "Utility":
                y_data.append(results['methods'][method]['Utility'])
            elif cfg["metric"] == "Agg_Utility":
                y_data.append(results['methods'][method]['Agg_Utility'])
            legends.append(method)

    # Ajout de la valeur optimale si applicable
    if cfg["metric"] in ["LSW", "SW"]:
        if cfg["metric"] == "SW":
            y_data.append(np.full_like(y_data[0], results['optimal']['SW']))

        else:
            y_data.append(np.full_like(y_data[0], results['optimal']['LSW']))
        LEGENDS.append("Optimal")

#    legends = LEGENDS
    # Création du graphique avec Plotly
    fig = go.Figure()
    markers2 = ["circle", "square", "diamond", "cross", "triangle-up", "star"]
    h_idx = 1
    for i, (data, legend) in enumerate(zip(y_data, LEGENDS)):
        if cfg["metric"] in ["Bid", "Agg_Bid", "Utility", "Agg_Utility"]:
            # Pour les graphiques multidimensionnels
            for j in range(data.shape[1]):
                fig.add_trace(go.Scatter(
                    x=x_data[::cfg["plot_step"]],
                    y=data[:, j][::cfg["plot_step"]],
                    mode="lines+markers",  # ✅ ligne + marqueur
                    name=f"{legend} -- Joueur {j + 1}",
                    line=dict(color=colors[i % len(colors)], width=3),  # couleur de ligne
                    marker=dict(
                        symbol=markers2[j % len(markers2)],  # type de marqueur
                        size=10,  # ✅ taille fixe (indépendante de plot_step)
                        line=dict(width=1, color="black")  # contour noir (optionnel pour visibilité)
                    ),
                    opacity=0.8
                ))

        else:
            fig.add_trace(go.Scatter(
                x=x_data[::cfg["plot_step"]],
                y=data[::cfg["plot_step"]],
                mode='lines+markers',
                name=legend,
                line=dict(color=colors[i % len(colors)], width=3),
                #showlegend=False  # 👈 on masque
            ))


    # Configuration du graphique
    y_label_map = {
        "Speed": "Convergence error",
        "LSW": "Liquid Social Welfare",
        "SW": "Social Welfare",
        "Bid": "Bid of  player",
        "Agg_Bid": "Aggregate Bid of player",
        "Utility": "Player utility",
        "Agg_Utility": "Player Aggregate Utility",
        "Dist_To_Optimum_SW": "Distance to the Optimal SW"
    }
    config["y_label"] = y_label_map[cfg["metric"]]
    fig.update_layout(
        title=f"Evolution of {y_label_map[cfg["metric"]]}",
        xaxis_title="Iterations",
        yaxis_title=y_label_map[cfg["metric"]],
        hovermode="x unified",
        height=600,
        template="plotly_white"
    )

    #y_data = {"speed": y_data_speed, "sw": y_data_sw, "lsw": y_data_lsw}
    if cfg["metric"] in ["Bid", "Agg_Bid", "Utility", "Agg_Utility"]:
        save_to =  cfg['metric'] + f"_alpha{cfg['alpha']}_gamma{cfg["gamma"]}_n_{cfg['n']}"
        figpath_plot, figpath_legend =plotGame_dim_N(x_data, y_data, cfg["x_label"], cfg["y_label"], LEGENDS, saveFileName=save_to,
                                 ylog_scale=cfg["ylog_scale"], pltText=cfg["pltText"], step=cfg["plot_step"])
    else:

        save_to = cfg['metric'] + f"_alpha{cfg['alpha']}_gamma{cfg["gamma"]}_n_{cfg['n']}"
        figpath_plot, figpath_legend = plotGame(x_data, y_data, cfg["x_label"], cfg["y_label"], LEGENDS, saveFileName=save_to,
                       ylog_scale=cfg["ylog_scale"], pltText=cfg["pltText"], step=cfg["plot_step"])

    fig.update_layout(
        title=f"Evolution of {y_label_map[cfg['metric']]}",
        xaxis_title="Iterations",
        yaxis_title=y_label_map[cfg['metric']],
        template="plotly_white",
        hovermode="x unified"
    )
    if cfg["ylog_scale"]:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)
    # Affichage des valeurs finales
    st.subheader("Final values")
    cols = st.columns(len(cfg["lrMethods"]) + 1)

    for i, method in enumerate(cfg["lrMethods"]):
        if method in results['methods']:
            with cols[i]:
                st.metric(
                    label=method,
                    value=f"{results['methods'][method]['convergence_iter']} itérations",
                    help=f"Last error: {results['methods'][method]['Speed'][-1]:.6f}"
                )

    with cols[-1]:
        st.metric(
            label="Optimal",
            value=f"LSW: {results['optimal']['LSW']:.2f}",
            help=f"SW: {results['optimal']['SW']:.2f}"
        )
    with col2:
        st.subheader("Outputs")

        # --- Plot PDF ---
        with open(figpath_plot, "rb") as f:
            plot_bytes = f.read()

        # --- Legend PDF ---
        with open(figpath_legend, "rb") as f:
            legend_bytes = f.read()

        # Put buttons on the same row
        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            st.download_button(
                "⬇️ Download Plot PDF",
                data=plot_bytes,
                file_name=figpath_plot,
                mime="application/pdf"
            )

        with btn_col2:
            st.download_button(
                "⬇️ Download Legend PDF",
                data=legend_bytes,
                file_name=figpath_legend,
                mime="application/pdf"
            )

# -----------------------
# SIMULATION TABLE
# -----------------------

if st.button("📊 Run Simulation Table"):
    with st.spinner("Simulating..."):
        results = run_simulation_table_avg(cfg, GameKelly)
        display_results_streamlit_dict(results, cfg, save_path="results/table_results.csv")
    #st.session_state.results = results
    #st.session_state.config = cfg
#if st.button("▶️ Run Simulation Gamma n"):
#    with st.spinner("Simulating..."):
#        #runner = SimulationRunner(cfg)
#        plot_results_multi_gamma_go(cfg, metric=cfg["metric"])

