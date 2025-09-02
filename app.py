import json
import numpy as np
import torch,io
import streamlit as st
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from src.game import *
from src.game.utils import GameKelly, Utility, plotGame, plotGame_dim_N
from src.game.config import SIMULATION_CONFIG as DEFAULT_CONFIG
from src.game.config import SIMULATION_CONFIG_table as DEFAULT_CONFIG_TABLE
from src.game.description import ALGO_DESCRIPTIONS

# Add logos side by side at the top
col1, col2 = st.columns([1, 5])

with col1:
    st.image("src/game/logo_avgn.png", use_container_width=True)  # Avignon Univ logo

with col2:
    st.image("src/game/cognetslogo1.png", use_container_width=True)  # Your project logo

st.set_page_config(page_title="Bidding – Kelly Games", layout="wide")

st.title("🎮 Bidding Simulator (α-fair Kelly Mechanism)")

from PIL import Image

# Convert PDF pages to images (assume 1 page each)
images1 = convert_from_path('src/game/kellyMechanism-Journal-Page-1.drawio.pdf', dpi=100)
images2 = convert_from_path('src/game/kellyMechanism-Journal-Page-2.drawio.pdf', dpi=100)
images3 = convert_from_path('src/game/kellyMechanism-Journal-Page-3.drawio.pdf', dpi=100)
images4 = convert_from_path('src/game/kellyMechanism-Journal-Page-4.drawio.pdf', dpi=100)

# Collect single-page images
img_list = [images1[0], images2[0], images3[0], images4[0]]

# Create a GIF in memory
gif_bytes = io.BytesIO()
img_list[0].save(
    gif_bytes,
    format='GIF',
    save_all=True,
    append_images=img_list[1:],  # the rest of the images
    duration=3*1000,  # 1000ms per frame
    loop=0
)
gif_bytes.seek(0)

# Streamlit button to play GIF
#if st.button("Play GIF"):
st.image(gif_bytes, use_container_width=True)

with st.sidebar:
    st.header("Configuration")
    cfg = dict(DEFAULT_CONFIG)

    cfg["n"] = st.number_input(
        "Players (n)", min_value=2, max_value=50, value=cfg["n"], step=1,
        help="Number of players in the game (2 to 50)."
    )

    cfg["alpha"] = st.selectbox(
        "α (fairness)", options=[0, 1, 2], index=[0, 1, 2].index(cfg["alpha"]),
        help="α-fairness parameter: 0 = utilitarian, 1 = proportional fairness, 2 = harmonic fairness."
    )

    cfg["T"] = st.number_input(
        "Iterations (T)", min_value=10, max_value=5000, value=cfg["T"], step=10,
        help="Number of iterations to run the learning algorithm."
    )

    cfg["eta"] = st.number_input(
        "Learning rate (η)", min_value=0.0001, max_value=5.0, value=float(cfg["eta"]), step=0.1, format="%.4f",
        help="Step size for gradient updates in learning algorithms."
    )

    cfg["price"] = st.number_input(
        "Price", min_value=0.0001, max_value=1000.0, value=float(cfg["price"]), step=0.1, format="%.4f",
        help="Price of the resource per unit."
    )

    cfg["a"] = st.number_input(
        "a (base utility scale)", min_value=0.1, max_value=1e6, value=float(cfg["a"]), step=10.0, format="%.4f",
        help="Base scale of utility for each player."
    )

    cfg["mu"] = st.number_input(
        "μ (c heterogeneity)", min_value=0.0, max_value=4.0, value=float(cfg["mu"]), step=0.1,
        help="Heterogeneity parameter for budget variations across players."
    )

    cfg["c"] = st.number_input(
        "c (budget base)", min_value=0.0001, max_value=1e6, value=float(cfg["c"]), step=10.0, format="%.4f",
        help="Base budget for each player."
    )

    cfg["delta"] = st.number_input(
        "δ (slack)", min_value=0.0, max_value=10.0, value=float(cfg["delta"]), step=0.1,
        help="Slack parameter for convergence tolerance."
    )

    cfg["epsilon"] = st.number_input(
        "ε (min bid)", min_value=1e-8, max_value=1e-1, value=float(cfg["epsilon"]), step=1e-4, format="%.6f",
        help="Minimum allowed bid value to prevent division by zero."
    )

    cfg["tol"] = st.number_input(
        "Tolerance", min_value=1e-9, max_value=1e-2, value=float(cfg["tol"]), step=1e-6, format="%.9f",
        help="Numerical tolerance for algorithm convergence."
    )

    cfg["gamma"] = st.number_input(
        "γ (a_i heterogeneity)", min_value=0.0, max_value=4.0, value=float(cfg["gamma"]), step=0.1,
        help="Heterogeneity of utility scaling among players."
    )
    # Liste de toutes les méthodes
    lr_methods_all = ["DAQ", "OGD", "SBRD", "NumSBRD", "DAH", "XL", "Hybrid"]

    # Méthodes normales à sélectionner
    selected_methods = st.multiselect(
        "Select learning methods",
        lr_methods_all,
        default=["DAQ", "SBRD"]
    )

    cfg["lrMethods"] = []

    # Si Hybrid est sélectionné, demander les méthodes à combiner
    if "Hybrid" in selected_methods:
        st.info("Hybrid selected: choose 2 methods to combine for Hybrid")
        hybrid_options = [m for m in lr_methods_all if m != "Hybrid"]
        hybrid_methods = st.multiselect(
            "Select at most 2 methods for Hybrid",
            hybrid_options,
            #default=hybrid_options[:2]
        )
        #if len(hybrid_methods)%cfg["n"] != 0:
        #    st.warning("Hybrid can only combine k methods such that k * Nbre_of_Hybrid_methods = n .")
         #   hybrid_methods = hybrid_methods[:2]
        cfg["Hybrid_funcs"] = hybrid_methods
        #cfg["lrMethods"] = [m for m in selected_methods if m != "Hybrid"] + [("Hybrid", hybrid_methods)]

    cfg["lrMethods"] = selected_methods


    cfg["x_label"] = st.text_input(
        "x_label",
        value=cfg.get("x_label", "X Axis"),
        help="Label X axis."
    )

    cfg["y_label"] = st.text_input(
        "y_label",
        value=cfg.get("y_label", "Y Axis"),
        help="Label Y axis."
    )

    cfg["ylog_scale"] = st.checkbox(
        "Y log scale", value=cfg["ylog_scale"],
        help="Use logarithmic scale on Y-axis for plotting (good for wide-range data)."
    )

    metrics_all = ["None", "utility", "bid", "speed", "sw", "lsw"]
    cfg["metric"] = st.selectbox(
        "Metric to plot", options=metrics_all, index=metrics_all.index(cfg["metric"]),
        help="Select which metric to plot. Choose 'None' to disable plotting."
    )

    cfg["plot_step"] = st.number_input(
        "Plot step", min_value=1, max_value=100, value=int(cfg["plot_step"]), step=1,
        help="Number of points to skip when plotting (reduce for faster plotting)."
    )

    # Range of number of players
    cfg["list_n"] = st.text_area(
        "List of players (list_n)",
        value=", ".join(str(x) for x in cfg.get("list_n", DEFAULT_CONFIG_TABLE["list_n"])),
        help="Comma-separated list of numbers of players to simulate."
    )
    # Convert input string to list of integers
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



st.download_button("⬇️ Download config JSON", data=json.dumps(cfg, indent=2), file_name="config.json", mime="application/json")

run = st.button("Run simulation")

selected_algo = st.selectbox("Choose algorithm to  describe", list(ALGO_DESCRIPTIONS.keys()))

if selected_algo != "None":
    with st.expander(f"Description of {selected_algo}", expanded=True):
        # Affichage direct du texte Python + commentaires
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

col1, col2 = st.columns([1,1])

if run:
    T = int(cfg["T"]); n = int(cfg["n"]); eta = float(cfg["eta"]); price = float(cfg["price"])
    a = float(cfg["a"]); mu = float(cfg["mu"]); c = float(cfg["c"]); delta = float(cfg["delta"])
    epsilon = torch.tensor(float(cfg["epsilon"]), dtype=torch.float64)
    lr_vary = False
    x_label = cfg["x_label"]
    y_label = cfg["y_label"] #r"$\varphi(z)$"
    ylog_scale = bool(cfg["ylog_scale"])
    plot_step = int(cfg["plot_step"])
    lrMethods = list(cfg["lrMethods"])
    alpha = int(cfg["alpha"]); gamma = float(cfg["gamma"])
    tol = float(cfg["tol"])

    x_data = np.arange(T)
    bid0 = torch.ones(n, dtype=torch.float64)
    a_vector = torch.tensor([a / (i + 1) ** gamma for i in range(n)], dtype=torch.float64)
    c_vector = torch.tensor([c / (i + 1) ** mu for i in range(n)], dtype=torch.float64)
    dmin = a_vector * torch.log((epsilon + torch.sum(c_vector) - c_vector + delta) / epsilon)
    d_vector = 0.7 * dmin

    y_data_speed, y_data_lsw, y_data_bid = [], [], []
    y_data_avg_bid, y_data_utility, y_data_sw = [], [], []

    set1 = torch.arange(n, dtype=torch.long)
    nb_hybrid = max(1, len(cfg["Hybrid_funcs"]))
    Hybrid_sets = torch.chunk(set1, nb_hybrid)

    with st.spinner("Simulating..."):
        for lrMethod in lrMethods:
            game_set = GameKelly(n, price, epsilon, delta, alpha, tol)
            Bids, Utilities, error_NE_set = game_set.learning(
                lrMethod, a_vector, c_vector, d_vector, T, eta, bid0, vary=lr_vary,
                Hybrid_funcs=cfg["Hybrid_funcs"], Hybrid_sets=Hybrid_sets
            )

            SocialWelfare = torch.sum(Utilities, dim=1)
            LSW = torch.sum(torch.minimum(Utilities, c_vector.unsqueeze(0)), dim=1)
            Avg_bids = torch.mean(Bids, dim=1)

            y_data_speed.append(error_NE_set.detach().numpy())
            y_data_lsw.append(LSW.detach().numpy())
            y_data_sw.append(SocialWelfare.detach().numpy())
            y_data_bid.append(Bids.detach().numpy())
            y_data_avg_bid.append(Avg_bids.detach().numpy())
            y_data_utility.append(Utilities.detach().numpy())

    with col1:
        st.subheader("Plot")
        if cfg["metric"] in ["utility","bid"]:
            if cfg["metric"] == "utility":
                save_to = "plot_utility"
                figpath = plotGame_dim_N(x_data, y_data_utility, x_label, y_label, lrMethods, saveFileName=save_to, ylog_scale=ylog_scale, step=plot_step)
            else:
                save_to = "plot_bid"
                figpath = plotGame_dim_N(x_data, y_data_bid, x_label, y_label, lrMethods, saveFileName=save_to, ylog_scale=ylog_scale, step=plot_step)
        else:
            save_to = f"plot_{cfg['metric']}"
            data_map = {"speed": y_data_speed, "sw": y_data_sw, "lsw": y_data_lsw}
            figpath = plotGame(x_data, data_map[cfg["metric"]], x_label, y_label, lrMethods, saveFileName=save_to, ylog_scale=ylog_scale, step=plot_step)

        # Display last saved figure
        import matplotlib.image as mpimg

        pages = convert_from_path(figpath, dpi=150)
        img = pages[0]  # première page du PDF

        # Affiche dans Streamlit
        st.image(img, caption=f"Saved: {figpath}")
    with col2:
        st.subheader("Outputs")
        st.write("Last utilities (per method):")
        for i, m in enumerate(lrMethods):
            st.write(f"- {m}: shape {np.array(y_data_utility[i]).shape}")

        with open(figpath, "rb") as f:
            st.download_button("⬇️ Download PDF", data=f.read(), file_name=figpath, mime="application/pdf")


from src.game.simulation_table import run_simulation,  display_results_streamlit


if st.button("Run Simulation Table"):
    results = run_simulation(cfg, GameKelly)
    display_results_streamlit(results,  cfg, save_path="results/table_results.csv")

st.markdown("---")
st.caption("Tip: Deploy on **Streamlit Cloud** or **Hugging Face Spaces** (Python + Streamlit).")