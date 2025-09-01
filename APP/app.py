import json
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

from src.game.utils import GameKelly, Utility, plotGame, plotGame_dim_N
from src.game.config import SIMULATION_CONFIG as DEFAULT_CONFIG
from src.game.description import ALGO_DESCRIPTIONS

# Add logos side by side at the top
col1, col2 = st.columns([1, 5])

with col1:
    st.image("src/game/logo_avgn.png", use_container_width=True)  # Avignon Univ logo

with col2:
    st.image("src/game/cognetslogo1.png", use_container_width=True)  # Your project logo

st.set_page_config(page_title="Edge Pricing & Bidding – Kelly Games", layout="wide")

st.title("🎮 Edge Pricing & Bidding Simulator (α-fair Kelly Mechanism)")

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

    lr_methods_all = ["None", "DAQ", "OGD", "SBRD", "NumSBRD", "DAH", "XL"]
    cfg["lrMethods"] = st.multiselect(
        "Learning Methods", lr_methods_all, default=cfg["lrMethods"],
        help="Select which learning algorithms to run. Choose 'None' if you do not want any."
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

    st.download_button("⬇️ Download config JSON", data=json.dumps(cfg, indent=2), file_name="config.json", mime="application/json")

run = st.button("Run simulation")

selected_algo = st.selectbox("Choose algorithm to  describe", list(ALGO_DESCRIPTIONS.keys()))
#help_button = st.button("❓ Help")

if selected_algo != "None":
    #desc = ALGO_DESCRIPTIONS[selected_algo]
    #st.markdown(f"**{selected_algo}**\n\n{desc.replace('Pseudo-code:', '```python\nPseudo-code:').replace('project onto feasible set','project onto feasible set\n```')}")
    with st.expander(f"Description of {selected_algo}", expanded=True):
        # Affichage direct du texte Python + commentaires
        st.code(ALGO_DESCRIPTIONS[selected_algo], language='python')

#if help_button and selected_algo != "None":
#    st.info(ALGO_DESCRIPTIONS[selected_algo])

#algo = st.selectbox("Choose algorithm", list(ALGO_DESCRIPTIONS.keys()))
#st.write(f"**Description:** {ALGO_DESCRIPTIONS[algo]}")

col1, col2 = st.columns([1,1])

if run:
    T = int(cfg["T"]); n = int(cfg["n"]); eta = float(cfg["eta"]); price = float(cfg["price"])
    a = float(cfg["a"]); mu = float(cfg["mu"]); c = float(cfg["c"]); delta = float(cfg["delta"])
    epsilon = torch.tensor(float(cfg["epsilon"]), dtype=torch.float64)
    lr_vary = False
    x_label = "t"
    y_label = r"$\varphi(z)$"
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
    nb_hybrid = max(1, len(["DAQ","DAH"]))
    Hybrid_sets = torch.chunk(set1, nb_hybrid)

    with st.spinner("Simulating..."):
        for lrMethod in lrMethods:
            game_set = GameKelly(n, price, epsilon, delta, alpha, tol)
            Bids, Utilities, error_NE_set = game_set.learning(
                lrMethod, a_vector, c_vector, d_vector, T, eta, bid0, vary=lr_vary,
                Hybrid_funcs=["DAQ","DAH"], Hybrid_sets=Hybrid_sets
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
        img = mpimg.imread(f"{figpath}")
        st.image(img, caption=f"Saved: {figpath}")

    with col2:
        st.subheader("Outputs")
        st.write("Last utilities (per method):")
        for i, m in enumerate(lrMethods):
            st.write(f"- {m}: shape {np.array(y_data_utility[i]).shape}")

        with open(figpath, "rb") as f:
            st.download_button("⬇️ Download PNG", data=f.read(), file_name=figpath, mime="application/png")

st.markdown("---")
st.caption("Tip: Deploy on **Streamlit Cloud** or **Hugging Face Spaces** (Python + Streamlit).")