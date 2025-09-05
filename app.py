import json, io
import numpy as np
import torch
import streamlit as st
from pdf2image import convert_from_path
from PIL import Image
import plotly.graph_objects as go
from main import SimulationRunner

from src.game.utils import GameKelly, plotGame, plotGame_dim_N
from src.game.config import SIMULATION_CONFIG as DEFAULT_CONFIG
from src.game.config import SIMULATION_CONFIG_table as DEFAULT_CONFIG_TABLE
from src.game.description import ALGO_DESCRIPTIONS
from src.game.simulation_table import run_simulation, display_results_streamlit

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
    duration=3000,
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

    cfg["n"] = st.sidebar.slider("Players (n)", 2, 50, cfg["n"])
    cfg["T"] = st.sidebar.slider("Iterations (T)", 10, 5000, cfg["T"], step=10)
    cfg["alpha"] = st.sidebar.selectbox("α (fairness)", [0, 1, 2], index=[0, 1, 2].index(cfg["alpha"]))
    cfg["eta"] = st.sidebar.number_input("Learning rate (η)", 0.0001, 5.0, float(cfg["eta"]), step=0.1, format="%.4f")
    cfg["price"] = st.sidebar.number_input("Price", 0.0001, 1000.0, float(cfg["price"]), step=0.1, format="%.4f")

    with st.sidebar.expander("Advanced Parameters"):
        cfg["a"] = st.number_input("a (base utility scale)", 0.1, 1e6, float(cfg["a"]), step=10.0, format="%.4f")
        cfg["a_min"] = st.number_input("minimum a (base utility scale)", 0.1, 1e6, float(cfg["a_min"]), step=1.0, format="%.4f")
        cfg["mu"] = st.number_input("μ (c heterogeneity)", 0.0, 4.0, float(cfg["mu"]), step=0.1)
        cfg["c"] = st.number_input("c (budget base)", 0.0001, 1e6, float(cfg["c"]), step=10.0, format="%.4f")
        cfg["init_bid"] = st.number_input("initial bid", 0.0001, 1e6, float(cfg["init_bid"]), step=10.0, format="%.4f")
        cfg["delta"] = st.number_input("δ (slack)", 0.0, 10.0, float(cfg["delta"]), step=0.1)
        cfg["epsilon"] = st.number_input("ε (min bid)", 1.0, 1.0*1e2, float(cfg["epsilon"]), step=0.5, format="%.6f")
        cfg["tol"] = st.number_input("Tolerance", 1e-9, 1e-2, float(cfg["tol"]), step=1e-6, format="%.9f")
        cfg["gamma"] = st.number_input("γ (a_i heterogeneity)", 0.0, 10.0, float(cfg["gamma"]), step=0.1)


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
    lr_methods_all = ["DAQ", "OGD", "SBRD", "NumSBRD", "DAH", "XL", "Hybrid"]

    selected_methods = st.multiselect(
        "Select learning methods",
        lr_methods_all,
        default=["DAQ", "SBRD"]
    )

    cfg["lrMethods"] = selected_methods

    # --- If Hybrid is selected, ask for funcs + sets ---
    if "Hybrid" in selected_methods:
        st.info("Hybrid selected: choose methods and sets for Hybrid.")

        # Select the methods to combine
        hybrid_options = [m for m in lr_methods_all if m != "Hybrid"]
        hybrid_methods = st.multiselect(
            "Select Hybrid funcs (methods to combine)",
            hybrid_options,
            default=hybrid_options[:2]
        )
        cfg["Hybrid_funcs"] = hybrid_methods

        # Define subsets of players
        hybrid_sets_str = st.text_area(
            "Hybrid sets (JSON list of lists)",
            value=str([list(range(0, 1)), list(range(1, int(cfg["n"])))]),
            help="Define subsets of players for Hybrid learning, e.g. [[0,1],[2,3]]"
        )
        try:
            cfg["Hybrid_sets"] = json.loads(hybrid_sets_str)
        except Exception:
            st.error("Invalid format for Hybrid_sets, please enter a valid JSON list of lists")

    else:
        cfg["Hybrid_funcs"] = []
        cfg["Hybrid_sets"] = []

    metrics_all = ["utility", "bid", "speed", "SW", "LSW","Dist_To_Optimum_SW"]
    cfg["metric"] = st.sidebar.selectbox("Metric to plot", metrics_all, index=metrics_all.index(cfg["metric"]))

    cfg["ylog_scale"] = st.sidebar.checkbox("Y log scale", value=cfg["ylog_scale"])
    cfg["plot_step"] = st.sidebar.slider("Plot step", 1, 100, int(cfg["plot_step"]))

    cfg["pltText"] = st.sidebar.checkbox("Display values", value=cfg["pltText"])

    st.sidebar.download_button("⬇️ Download config JSON", data=json.dumps(cfg, indent=2),
                               file_name="config.json", mime="application/json")

    st.download_button("⬇️ Download config JSON", data=json.dumps(cfg, indent=2), file_name="config.json")

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
    a_i = \frac{a}{(i+1)^\gamma}
    \quad, \quad x_i =\frac{z_i}{\sum_{j=1}^n z_j + \delta}
    """)

    st.markdown(r"""
    **Where:**  
    - \($x_i$\) = allocated resource share for player $i$  
    - \($\alpha \ge 0$\) is the fairness parameter  
    - \($a$\) is the base utility scale  
    - \($\gamma \ge 0$\) controls heterogeneity across players
    - \($\lambda$\) the price
    """)

# -----------------------
# RUN SIMULATION
# -----------------------
if st.button("▶️ Run Simulation"):

    with st.spinner("Simulating..."):
        runner = SimulationRunner(cfg)
        results = runner.run_simulation()

        # Stockage des résultats dans la session
    st.session_state.results = results
    st.session_state.config = cfg

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

    for method in config['lrMethods']:
        if method in results['methods']:
            if cfg["metric"] == "speed":
                y_data.append(results['methods'][method]['error_NE'])
            elif cfg["metric"] == "LSW":
                y_data.append(results['methods'][method]['LSW'])
            elif cfg["metric"] == "Dist_To_Optimum_SW":
                y_data.append(results['methods'][method]['Dist_To_Optimum_SW'])
            elif cfg["metric"] == "SW":
                y_data.append(results['methods'][method]['SW'])
            elif cfg["metric"] == "bid":
                y_data.append(results['methods'][method]['bids'])
            elif cfg["metric"] == "utility":
                y_data.append(results['methods'][method]['utilities'])
            legends.append(method)

    # Ajout de la valeur optimale si applicable
    if cfg["metric"] in ["LSW", "SW"]:
        if cfg["metric"] == "SW":
            y_data.append(np.full_like(y_data[0], results['optimal']['LSW']))
        else:
            y_data.append(np.full_like(y_data[0], results['optimal']['SW']))
        legends.append("Optimal")

    # Création du graphique avec Plotly
    fig = go.Figure()

    for i, (data, legend) in enumerate(zip(y_data, legends)):
        if cfg["metric"] in ["bid", "utility"]:
            # Pour les graphiques multidimensionnels
            for j in range(data.shape[1]):
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=data[:, j],
                    mode='lines',
                    name=f"{legend} - Joueur {j + 1}",
                    opacity=0.7
                ))
        else:
            # Pour les graphiques simples
            fig.add_trace(go.Scatter(
                x=x_data,
                y=data,
                mode='lines+markers',
                name=legend,
                line=dict(width=3)
            ))

    # Configuration du graphique
    y_label_map = {
        "speed": "Convergence error",
        "lSW": "Liquid Social Welfare (LSW)",
        "SW": "Social Welfare (SW)",
        "bid": "Social Welfare",
        "utility": "Player utilities",
        "Dist_To_Optimum_SW": "Distance to the Optimal Social Welfare"
    }

    fig.update_layout(
        title=f"Evolution of {y_label_map[cfg["metric"]]}",
        xaxis_title="Iterations",
        yaxis_title=y_label_map[cfg["metric"]],
        hovermode="x unified",
        height=600,
        template="plotly_white"
    )

    #y_data = {"speed": y_data_speed, "sw": y_data_sw, "lsw": y_data_lsw}
    print(f"y_data:{y_data}")
    if cfg["metric"] in ["bid", "utility"]:
        save_to = f"plot_{cfg['metric']}"
        figpath=plotGame_dim_N(x_data, y_data, cfg["x_label"], cfg["y_label"], cfg["lrMethods"], saveFileName=save_to,
                                 ylog_scale=cfg["ylog_scale"], step=cfg["plot_step"])
        print(figpath)
    else:
        save_to = f"plot_{cfg['metric']}"
        figpath = plotGame(x_data, y_data, cfg["x_label"], cfg["y_label"], cfg["lrMethods"], saveFileName=save_to,
                       ylog_scale=cfg["ylog_scale"], step=cfg["plot_step"])

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
                    help=f"Last error: {results['methods'][method]['error_NE'][-1]:.6f}"
                )

    with cols[-1]:
        st.metric(
            label="Optimal",
            value=f"LSW: {results['optimal']['LSW']:.2f}",
            help=f"SW: {results['optimal']['SW']:.2f}"
        )
    with col2:
        st.subheader("Outputs")
        with open(figpath, "rb") as f:
            st.download_button("⬇️ Download PDF", data=f.read(), file_name=figpath, mime="application/pdf")
# -----------------------
# SIMULATION TABLE
# -----------------------
if st.button("📊 Run Simulation Table"):
    results = run_simulation(cfg, GameKelly)
    display_results_streamlit(results, cfg, save_path="results/table_results.csv")
