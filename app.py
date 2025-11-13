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
from main_table_simulation import run_simulation_table_avg, display_results_streamlit_dict
from src.game.Jain_index import run_jain_vs_gamma, plot_jain_vs_gamma
#from simulation_param_n_gamma import *

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

# --- Title and description ---
st.title("🎮 Bidding Simulator (α-Fair Kelly Mechanism)")
st.markdown("""
Simulate and analyze different **learning algorithms** in a **Kelly Mechanism** game.  
Configure all parameters in the sidebar and visualize convergence, social welfare, and fairness metrics.
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


# =========================================================
# ⚙️ SIDEBAR CONFIGURATION
# =========================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- Copy default config ---
    cfg = dict(DEFAULT_CONFIG)

    # ------------------------
    # 📌 Basic parameters
    # ------------------------
    cfg["n"] = st.number_input("Players (n)", 2, 100, cfg["n"], step=1)

    cfg["T"] = st.number_input("Iterations (T)", 10, 100000, cfg["T"], step=10)
    cfg["T_plot"] = st.number_input("Nb Iterations  to plot (T)", 10, 100000, cfg["T"], step=10)
    cfg["Nb_random_sim"] = st.number_input("Number of simulations", 1, 50, int(cfg["Nb_random_sim"]), step=1)
    cfg["alpha"] = st.selectbox("α (fairness)", [0, 1, 2], index=[0, 1, 2].index(cfg["alpha"]))
    cfg["eta"] = st.number_input(rf"RRM Learning rate ($\beta$)", 1e-7, 100.0, float(cfg["eta"]), step=0.1, format="%.7f")
    cfg["lr_vary"] = st.checkbox("Vary learning rate over time?", value=cfg["lr_vary"])
    cfg["keep_initial_bid"] = st.checkbox("Keep same initial bid for all simulations?", value=False)

    # ------------------------
    # 💰 Game parameters
    # ------------------------
    cfg["price"] = st.number_input("Price (λ)", 1e-4, 1000.0, float(cfg["price"]), step=0.1, format="%.4f")
    cfg["a"] = st.number_input("a (utility scale)", 0.1, 1e6, float(cfg["a"]), step=10.0)
    cfg["gamma"] = st.number_input("γ (heterogeneity)", 0.0, cfg["a"], float(cfg["gamma"]), step=1.0)
    cfg["d_vector"] = st.number_input("d_i (service offset)", 0.0, 1e6, 0.0, step=10.0)

    # ------------------------
    # 🎯 Metric to visualize
    # ------------------------
    metrics_all = [
        "Relative_Efficienty_Loss","Avg_Payoff","Payoff", "epsilon_error",   "Speed",  "Bid", "Potential", "Pareto",
        "SW","Jain_Index", "LSW", "Dist_To_Optimum_SW", "Avg_Bid",  "Res_Payoff"
    ]
    cfg["metric"] = st.selectbox("Metric to plot", metrics_all, index=metrics_all.index(cfg["metric"]))

    cfg["Track"] = st.checkbox("Track the metric over time?", value=True)
    cfg["pltLegend"] = st.checkbox("Plot legend ", value=True)
    cfg["Random_set"] = st.checkbox("Random players' sets?", value=True)

    # ------------------------
    # 🎲 Initial bids
    # ------------------------
    cfg["Random_Initial_Bid"] = st.checkbox("Random initial bids?", value=True)
    if not cfg["Random_Initial_Bid"]:
        cfg["var_init"] = st.number_input("Variance of initial bids", 0.0, 1e6, float(cfg["var_init"]), step=1.0)

    # ------------------------
    # ⚙️ Advanced parameters
    # ------------------------
    with st.expander("Advanced Parameters"):
        cfg["a_min"] = st.number_input("Minimum a_i", 0.1, 1e6, float(cfg["a_min"]), step=1.0)
        cfg["mu"] = st.number_input("μ (budget heterogeneity)", 0.0, 4.0, float(cfg["mu"]), step=0.1)
        cfg["c"] = st.number_input("c (base budget)", 1e-4, 1e6, float(cfg["c"]), step=10.0)
        cfg["delta"] = st.number_input("δ (slack)", 0.0, 10.0, float(cfg["delta"]), step=0.1)
        cfg["epsilon"] = st.number_input("ε (min bid)", 0.0, 100.0, float(cfg["epsilon"]), step=0.05)



        # Heterogeneity vectors
        cfg["a_vector"] = st.text_area(
            "List of heterogeneous a_i",
            value=str([max(cfg["a"] - cfg["gamma"] * i, cfg['a_min']) for i in range(cfg["n"])])
        )
        try:
            cfg["a_vector"] = ast.literal_eval(cfg["a_vector"])
        except Exception:
            st.error("Invalid format for a_vector.")

        # Ranges for tables
        cfg["list_n"] = [int(x) for x in st.text_area(
            "List of n values",
            value=", ".join(str(x) for x in DEFAULT_CONFIG_TABLE["list_n"])
        ).split(",") if x.strip()]
        cfg["list_gamma"] = [float(x) for x in st.text_area(
            "List of γ values",
            value=", ".join(str(x) for x in DEFAULT_CONFIG_TABLE["list_gamma"])
        ).split(",") if x.strip()]


        s_min = (cfg["n"] - 1) * cfg["epsilon"] + cfg["delta"]
        s_max = (cfg["n"] - 1) * cfg["c"] * torch.ones(1) + cfg["delta"]
        z_max = BR_alpha_fair(cfg["epsilon"] * torch.ones(1), cfg["c"] * torch.ones(1), cfg["c"] * torch.ones(1), s_min, torch.tensor(cfg["a_vector"]), cfg["delta"], cfg["alpha"], cfg["price"], b=0)
        x_max = z_max / (z_max + s_min)
        x_min = cfg["epsilon"] * torch.ones(1) / (cfg["epsilon"] * torch.ones(1) + s_max)
        x_min_2 = cfg["c"] * torch.ones(1) / (cfg["c"] * torch.ones(1) + s_max)

        Payoff_min = torch.min(Payoff(x_min, cfg["epsilon"] * torch.ones(1), torch.tensor(cfg["a_vector"]) , cfg["d_vector"], cfg["alpha"], cfg["price"]),
                               Payoff(x_min_2, cfg["c"] * torch.ones(1), torch.tensor(cfg["a_vector"]), cfg["d_vector"], cfg["alpha"], cfg["price"]))
        Payoff_max = torch.max(Payoff(x_max, z_max, torch.tensor(cfg["a_vector"]), cfg["d_vector"], cfg["alpha"], cfg["price"]))
       # print(Payoff_max , Payoff_min,float(cfg["tol"])/ float(Payoff_max - Payoff_min[0])) #/ float(Payoff_max - Payoff_min[0])
        cfg["tol"] = st.number_input("Tolerance", 1e-12, 1e-2, float(cfg["tol"]), step=1e-6)

    # ------------------------
    # 🧠 Learning methods
    # ------------------------
    lr_methods_all = ["DAQ_F", "DAQ_V", "OGD_F", "OGD_F", "OGD_V", "SBRD", "DAE",  "RRM_V", "Hybrid",  "XL", "NumSBRD", ]
    selected_methods = st.multiselect(
        "Select learning methods",
        lr_methods_all,
        default=[ "DAQ_F", "DAQ_V",  "RRM_V", "OGD_F", "OGD_V","SBRD"]
    )
    # ✅ If "Hybrid" is selected, keep only "Hybrid"
    if "Hybrid" in selected_methods:
        selected_methods = ["Hybrid"]
    cfg["lrMethods"] = selected_methods
    cfg["selected_methods"] = selected_methods

    # 🧠 Learning rates
    DEFAULT_CONFIG["Learning_rates"] = [cfg["eta"]] * len(selected_methods)
    LEGENDS = []
    LEGENDS_Hybrid = []

    # --- Default LR ---
    cfg["num_lrmethod"] = 0

    if len(selected_methods) == 1 and selected_methods[0]!="Hybrid" and selected_methods[0]!="SBRD" and  selected_methods[0]!="RRM_nt":#and selected_methods[0] != "Hybrid" and selected_methods[0] != "SBRD":
        # Number of learning rates for the single method
        num_lrmethod = st.number_input(
            "Number of learning rates",
            min_value=1,
            max_value=6,
            value=1,
            step=1
        )
        cfg["num_lrmethod"] = num_lrmethod
        # Input string for learning rates
        lr_input = st.text_area(
            "List of Learning Rates",
            value=", ".join(str(cfg["eta"]) for _ in range(num_lrmethod)),
            help="Comma-separated list of learning rate values."
        )

        # Convert input string to list of floats
        try:
            cfg["Learning_rates"] = [float(x.strip()) for x in lr_input.split(",") if x.strip()]
        except ValueError:
            st.error("Invalid format for Learning_rates, please enter numbers separated by commas.")
            cfg["Learning_rates"] = [cfg["eta"]] * num_lrmethod

        # Methods repeated according to number of learning rates
        cfg["lrMethods"] = [selected_methods[0]] * num_lrmethod


        # Legends
        for i, lr in enumerate(cfg["Learning_rates"]):
            #leg = selected_methods[0]
            LEGENDS.append(rf"{selected_methods[0]}")# -- $\eta={lr}$")

    else:

        # Multiple methods (or contains Hybrid)
        lr_input = st.text_area(
            "List of Learning Rates",
            value=", ".join(str(cfg["eta"]) for _ in selected_methods),
            help="Comma-separated list of learning rate values."
        )

        # Convert input string to list of floats
        try:
            cfg["Learning_rates"] = [float(x.strip()) for x in lr_input.split(",") if x.strip()]
        except ValueError:
            st.error("Invalid format for Learning_rates, please enter numbers separated by commas.")
            cfg["Learning_rates"] = [cfg["eta"]] * len(selected_methods)

        for i,lr in enumerate(selected_methods):
            if selected_methods[i]!="SBRD" and selected_methods[i]!="Hybrid" and selected_methods[i]!="RRM_nt" :
                LEGENDS.append(rf"{selected_methods[i]}")# -- $\eta={lr}$")
            if selected_methods[i]=="SBRD"   :

                LEGENDS.append(selected_methods[i])
    # Only run this block if RRM is selected
    if "RRM_nt" in selected_methods:
        # --- Defaults + UI input ---
        default_rrm_rates = cfg.get("RRM_lr", [cfg["eta"]])
        rates_str_default = ", ".join(str(x) for x in default_rrm_rates)

        rates_str = st.text_area(
            "List of RRM learning rates",
            value=rates_str_default,
            help=r"Comma-separated list of learning rates ($\beta$) values."
        )

        # Parse to list of floats
        try:
            cfg["RRM_lr"] = [float(x.strip()) for x in rates_str.split(",") if x.strip()]
            if not cfg["RRM_lr"]:
                raise ValueError("Empty list")
        except Exception:
            st.error("Invalid RRM learning rates. Using default [0.05].")
            cfg["RRM_lr"] = [cfg["eta"]]

        # --- Expand the selected methods at the RRM position ---
        # Remember first RRM position
        first_idx = selected_methods.index("RRM_nt")

        # Remove all RRM occurrences
        selected_methods = [m for m in selected_methods if m != "RRM_nt"]
        cfg["Learning_rates"] = [m for m in cfg["Learning_rates"] if m != "RRM_nt"]
        LEGENDS = [m for m in LEGENDS if m != "RRM_nt"]

        # Insert one RRM per rate, preserving the original position
        for i, lr in enumerate(cfg["RRM_lr"]):
            selected_methods.insert(first_idx + i, "RRM_nt")
            cfg["Learning_rates"].insert(first_idx + i, lr)
            LEGENDS.insert(first_idx + i, rf"RRM_nt_{lr}")

        # Update cfg
        cfg["lrMethods"] = selected_methods


    cfg["num_hybrids"] = [0]
    cfg["num_hybrid_set"] =[0]


    cfg["y_zoom_interval"] = [0.0,1.0]


    if "Hybrid" in selected_methods :
        st.info("You selected Hybrid. You can configure multiple hybrid algorithms below.")
        func_group  = []
        # Number of hybrids
        num_hybrids = st.number_input(
            "How many Hybrid algorithms do you want to configure?",
            min_value=1,
            max_value=cfg["n"]-1,
            value=1,
            step=1
        )
        x_zoom_interval = st.slider(
            label="🔍 Select X-axis zoom interval",
            min_value=1,
            max_value=num_hybrids,
            value=(1, num_hybrids+1),  # default: full x range
            step=1
        )
        cfg["x_zoom_interval"] = x_zoom_interval

        cfg["num_hybrids"] = num_hybrids
        hybrid_options = [m for m in lr_methods_all if m != "Hybrid"]
        num_hybrid_set = st.number_input(
            "How many Group Hybrid algorithms do you want to configure?",
            min_value=1,
            max_value=cfg["n"]-1,
            value=1,
            step=1
        )
        if num_hybrids>1:
            cfg["lrMethods"] = cfg["lrMethods"] + ["Hybrid"]*(num_hybrid_set*num_hybrids - 1)
        cfg["Hybrid_funcs_"] = []
        cfg["num_hybrid_set"] = num_hybrid_set

        #print(f"num_hybrid_set{num_hybrid_set}")
        for i in range(num_hybrid_set):
            method = st.multiselect(
                f"Select Hybrid funcs ",
                hybrid_options,
                default=["SBRD","DAQ_V"],
                key=f"hybrid_method_{i}"
            )
            #print(method)
            if method[1] not in func_group:
                func_group.append(method[1])
            cfg["Hybrid_funcs_"].append(method)

        #selected_methods =  [m for m in selected_methods if m != "Hybrid"]

        h_idx = 1
        # Initialise la liste des k si pas déjà définie
        if "Nb_A1" not in cfg:
            cfg["Nb_A1"] = []
        else:
            cfg["Nb_A1"] = cfg["Nb_A1"]

        cfg["Hybrid_sets"] =[]
        cfg["Hybrid_funcs"] = []

        for secMeth in range(num_hybrid_set):
            cfg["Learning_rates"] = cfg["Learning_rates"] + [cfg["eta"]] * num_hybrids

            sets = []
            if num_hybrids == 1:
                percent_A1 = st.number_input(
                    "Select percentage of players in first subset (A₁)",
                    min_value=1,
                    max_value=99,
                    value=50,
                    step=1,
                #%format="%d%%",
                    help="Defines the percentage of players assigned to A₁ in the hybrid group."
                )
                # Convert percentage to number of players
                cfg["Nb_A1"] = [max(1, int(cfg["n"] * percent_A1 / 100))]

            else:
                cfg["Nb_A1"] += list(range(1, num_hybrids + 1))
            LEGENDS_Hybrid.append(cfg["Hybrid_funcs_"][secMeth][1])#+rf" -- $\eta={cfg["eta"]}$")
            #LEGENDS_Hybrid.append(cfg["Hybrid_funcs_"][secMeth][1] + rf" -- $\eta={cfg["eta"]}$")
            sets = []  # contiendra la liste finale de [subset, remaining]
            kk = 0
            h_idx = 0

            for h in cfg["Nb_A1"][:num_hybrids]:

                cfg["Hybrid_funcs"].append(cfg["Hybrid_funcs_"][secMeth])
                h_idx += 1

                # --- Construire la première liste : [0, ... autres sauf 1] ---
                # candidats possibles : tous sauf 0 et 1, car 0 sera ajouté manuellement et 1 exclu
                [subset, remaining] = make_subset(cfg["n"], h)
                kk += 1
                #LEGENDS_Hybrid.append(f"({Hybrid_funcs[0]}: {self.config['Nb_A1'][idx]}, {Hybrid_funcs[1]}: {n - self.config['Nb_A1'][idx]})")
                cfg["Hybrid_sets"].append([subset, remaining])
    else:
        x_zoom_interval = st.slider(
            label="🔍 Select X-axis zoom interval",
            min_value=0,
            max_value=cfg["T"],
            value=(0, cfg["T"]),  # default: full x range
            step=1
        )
        cfg["x_zoom_interval"] = x_zoom_interval
    LEGENDS = LEGENDS_Hybrid + LEGENDS
    cfg["LEGENDS"]=LEGENDS



    cfg["Players2See"] = list(range(0, 1))

    if cfg["metric"] in ["Payoff", "Avg_Payoff", "Bid", "Avg_Bid"]:
        cfg["Players2See"] =  st.text_area(
            "List of players to display metrics",
            value=", ".join(str(x) for x in cfg.get("Players2See", cfg["Players2See"])),
            help="Comma-separated list of γ (a_i heterogeneity) values."
        )
        # Convert input string to list of floats
        try:
            cfg["Players2See"] = [int(x.strip()) for x in cfg["Players2See"].split(",") if x.strip()]
        except:
            st.error("Invalid format for Players to See, please enter numbers separated by commas.")

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
with st.expander("Why run this?"):
    st.write("Running the simulation computes the equilibrium for each learning method...")

if st.button("▶️ Run Simulation"):
    # Conteneur pour le chrono
    st.write("""
        This section runs the simulation:
        1. Initialize bids for all selected learning methods  
        2. Evaluate the chosen performance metric  
        3. Repeat steps (1–2) for the specified number of simulations  
        4. Compute the average metric values across all runs  
        """)

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


# =========================================================
# 📊 DISPLAY RESULTS (if available)
# =========================================================
try:
    if 'results' in st.session_state:
        # --- Retrieve data from session ---
        results = st.session_state.results
        config = st.session_state.config

        st.header("📈 Simulation Results")


        # --- Summary metrics ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Players (n)", config['n'])
        with col2:
            st.metric("Iterations (T)", config['T'])
        with col3:
            st.metric("Fairness (α)", config['alpha'])

        # =====================================================
        # 📘 Step 1: Prepare data
        # =====================================================
        x_data = np.arange(config["T"])
        y_data, legends = [], []


        for method in LEGENDS:
            if cfg["num_hybrids"] ==1  and cfg["num_hybrid_set"] ==1:

                y_data.append(results['methods']["Hybrid"][cfg["metric"]])


            else:


                y_data.append(results['methods'][method][cfg["metric"]])
            legends.append(method)
           # print(y_data)
        # --- Add optimal baseline if needed ---
        if cfg["metric"] in ["LSW", "SW"]:
            if cfg["metric"] == "SW":
                y_data.append(np.full_like(y_data[0], results['optimal']['SW']))

            else:
                y_data.append(np.full_like(y_data[0], results['optimal']['LSW']))
            #LEGENDS.append("Optimal")

        # 📘 Step 2: Build Plotly figure
        # =====================================================
        fig = go.Figure()
        markers2 = [
            "pentagon", "star", "x", "cross", "square", "triangle-up",
            "triangle-down", "diamond", "circle"
        ]

        h_idx = 1
        if cfg["Track"] :
            for i, (data, legend) in enumerate(zip(y_data, LEGENDS)):

                if cfg["metric"] in ["Bid", "Avg_Bid", "Payoff", "Avg_Payoff", "Res_Payoff"]:
                    # Pour les graphiques multidimensionnels
                    for j in range(np.array(data).shape[1]):
                        try:
                            ydata = data[0]
                            xdata = x_data[::cfg["plot_step"]]
                        except:
                            ydata = data[:,0][::cfg["plot_step"]]
                            xdata = np.array(np.arange(1, np.array(data).shape[0]+1)[:num_hybrids])/ cfg["n"]*100

                        fig.add_trace(go.Scatter(
                            x=xdata,
                            y=ydata,
                            mode="lines+markers",  # ✅ ligne + marqueur
                            name=f"{legend} -- Player {j + 1}",
                            line=dict(color=("red" if legend == "Optimal" else COLORS_METHODS[legends[i]] if legends[i] in METHODS else colors[i]), width=3),  # couleur de ligne
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
                        line=dict(color=("red" if legend == "Optimal" else COLORS_METHODS[legends[i]] if legends[i] in METHODS else colors[i]), width=3),
                    ))


        # =====================================================
        # 📘 Step 3: Format layout
        # =====================================================
        y_label_map = {
            "Speed": str(rf"$||BR(z(t) -z(t)||_{{2}}$"),
            "LSW": "LSW",
            "SW": "Social Welfare ",
            "Bid": "Player Bid",
            "Avg_Bid": "Average Bid",
            "epsilon_error": rf"$\epsilon(z(t))$",
            "Jain_Index": "Jain Index",
            "Payoff": "Player's Payoff",
            "Avg_Payoff": "Average Payoff",
            "Res_Payoff": "Payoff Residual",
            "Dist_To_Optimum_SW": "Distance to Optimal SW",
            "Relative_Efficienty_Loss": r"$\rho(z(t))$",
            "Pareto": "Pareto Check",
            "Potential": "Potential"
        }
        config["y_label"] = y_label_map[cfg["metric"]]
        cfg["y_label"] = y_label_map[cfg["metric"]]
        fig.update_layout(
            title=f"Evolution of {y_label_map[cfg["metric"]]}",
            xaxis_title="Iterations",
            yaxis_title=y_label_map[cfg["metric"]],
            hovermode="x unified",
            height=600,
            template="plotly_white"
        )

        #z_sol_equ = solve_quadratic(cfg["n"], cfg["a"], cfg["delta"])
        #x_ne = z_sol_equ / (cfg["n"] * z_sol_equ + cfg["delta"])
        x_opt = results['optimal']["x_opt"]
        x_ne = results['optimal']["x_ne"]
        z_ne = results['optimal']["z_ne"]

        payoff_ne = results['optimal']["payoff_ne"]
        Valuation_ne = Valuation(x_ne, cfg["a"], cfg["d_vector"], cfg["alpha"])
        SW_ne = results['optimal']["SW_NE"]
        Jain_idx_ne = results["optimal"]["Jain_index_NE"]
        SW_opt = results['optimal']["SW"]
        Residual_ne = results['optimal']["Residual_ne"]
        RLoss = torch.abs((SW_ne - SW_opt) / SW_opt)*100

        #y_data = {"speed": y_data_speed, "sw": y_data_sw, "lsw": y_data_lsw}
        if cfg["metric"] in ["Bid", "Avg_Bid", "Payoff", "Avg_Payoff", "Res_Payoff","Pareto"]:
            save_to =  cfg['metric'] + f"_alpha{cfg['alpha']}_gamma{cfg["gamma"]}_n_{cfg['n']}"
            try:

                y_data_2 = y_data
                LEGENDS2 = LEGENDS

                # baseline: une ligne plate de payoff_opt avec la bonne longueur

                if cfg["metric"] in ["Payoff", "Avg_Payoff", "Res_Payoff"]:
                    baseline = payoff_ne * np.ones_like(y_data_2[0])
                    y_data_2.append(baseline)


                    LEGENDS2.append("NE")

                elif cfg["metric"] in ["Bid", "Avg_Bid"] :
                    baseline = z_ne.detach().numpy() * np.ones_like(y_data_2[0])
                    y_data_2.append(np.array(baseline))
                    LEGENDS2.append("NE")

                figpath_plot, figpath_legend, figpath_zoom =plotGame_dim_N(cfg,x_data, y_data_2, cfg["x_label"], cfg["y_label"], LEGENDS2, saveFileName=save_to,
                                                                 fontsize=40, markersize=45, linewidth=12,linestyle="--",
                                                                 Players2See=cfg["Players2See"],
                                             ylog_scale=cfg["ylog_scale"], pltText=cfg["pltText"], step=cfg["plot_step"])
            except Exception as e:
                    print("plotGame_dim_N")
               # st.warning(f"⚠️ Error generating static plot: {e}")
               # figpath_plot = figpath_legend = figpath_zoom = None
        else:
            save_to = cfg['metric'] + f"_alpha{cfg['alpha']}_gamma{cfg["gamma"]}_n_{cfg['n']}"
            try:
                xlab = rf"$\alpha_{{{cfg["Hybrid_funcs"][0][0]}}}$"
                figpath_plot, figpath_legend, figpath_zoom = plotGame(cfg,x_data, y_data, cfg["x_label"], cfg["y_label"], LEGENDS,
                                                        saveFileName=save_to,fontsize=45, markersize=45, linewidth=12,linestyle="--",
                                                            ylog_scale=cfg["ylog_scale"], pltText=cfg["pltText"], step=cfg["plot_step"])
            except Exception:
                print("")
        if "Hybrid" in selected_methods and len(selected_methods)==1:



            x_data_2 = np.array(cfg["Nb_A1"]) / cfg["n"] * 100
            if num_hybrid_set>1:
                x_data_2 = np.array(cfg["Nb_A1"][:num_hybrid_set]) / cfg["n"] * 100
            y_data_2 = y_data.copy()
            y_data_2 = [el.detach().cpu().numpy() if hasattr(el, "detach") else np.array(el)
                        for el in y_data_2]
            funcs_ = cfg["Hybrid_funcs"][0]

            if num_hybrids>1 and num_hybrid_set>1:
                x_data_2 = np.array(cfg["Nb_A1"][:num_hybrids]) / cfg["n"] * 100
                save_to2 = cfg['metric'] + f"_alpha{cfg['alpha']}_gamma{cfg["gamma"]}_player"
                xlab = rf"$\alpha_{{{funcs_[0]}}}$"


                if cfg["metric"] in ["Payoff", "Avg_Payoff", "Res_Payoff"]:
                    baseline = payoff_ne * np.ones_like(y_data_2[0])
                    y_data_2.append(np.array(baseline))

                    func_group.append("Non-hybrid")
                elif cfg["metric"] in ["Bid", "Avg_Bid"] :
                    baseline = z_ne.detach().numpy() * np.ones_like(y_data_2[0])
                    y_data_2.append(np.array(baseline))

                elif cfg["metric"] in ["Speed"]:
                    cfg["y_label"] = str(rf"$||BR(z(T) -z(T)||_2$")
                    baseline = Residual_ne * np.ones_like(y_data_2[0])
                    y_data_2.append(np.array(baseline))

                    func_group.append("Non-hybrid")
                elif cfg["metric"] in ["Jain_Index"] :
                    baseline = Jain_idx_ne * np.ones_like(y_data_2[0])

                    y_data_2.append(np.array(baseline))

                    func_group.append("Non-hybrid")
                elif cfg["metric"] == "Relative_Efficienty_Loss":
                    cfg["y_label"] = r"$\rho(z(T))$"
                    baseline = RLoss.detach().numpy() * np.ones_like(y_data_2[0])
                    y_data_2.append(np.array(baseline))
                    func_group.append("Non-hybrid")
                elif cfg["metric"] == "epsilon_error":
                    cfg["y_label"] = r"$\epsilon(z(T))$"
                    baseline = cfg["tol"] * np.ones_like(y_data_2[0])
                    y_data_2.append(np.array(baseline))
                    func_group.append("Non-hybrid")

                figpath_plot, figpath_zoom, figpath_legend = plotGame_Hybrid_last(cfg, x_data_2, y_data_2, xlab, cfg["y_label"],
                                                                                   cfg["lrMethods"],
                                                                                   saveFileName=save_to2, funcs_=func_group,
                                                                                   fontsize=40, markersize=45, linewidth=12,
                                                                                   linestyle="--",
                                                                                   Players2See=cfg["Players2See"],
                                                                                   ylog_scale=cfg["ylog_scale"],
                                                                                   pltText=cfg["pltText"], step=1)



            else:

                y_data_2 = y_data_2

                # baseline: une ligne plate de payoff_opt avec la bonne longueur
                if cfg["metric"] in ["Payoff", "Avg_Payoff", "Res_Payoff"]:
                    baseline = payoff_ne * np.ones_like(y_data_2[0])
                    y_data_2.append(np.array(baseline))
                    func_group.insert(0, cfg["Hybrid_funcs"][0][0])

                    func_group.append("Non-hybrid")
                    if num_hybrids ==1:
                        figpath_plot, figpath_legend, figpath_zoom = plotGame_dim_N(cfg, x_data, y_data_2, cfg["x_label"],
                                                                                    cfg["y_label"], LEGENDS2,
                                                                                    saveFileName=save_to,
                                                                                    fontsize=40, markersize=45,
                                                                                    linewidth=12, linestyle="--",
                                                                                    Players2See=cfg["Players2See"],
                                                                                    ylog_scale=cfg["ylog_scale"],
                                                                                    pltText=cfg["pltText"],
                                                                                    step=cfg["plot_step"])
                    else:
                        xlab = rf"$\alpha_{{{funcs_[0]}}}$"
                        save_to2 = cfg['metric'] + f"_alpha{cfg['alpha']}_gamma{cfg["gamma"]}_player"
                        figpath_plot, figpath_zoom, figpath_legend  = plotGame_dim_N_last(cfg, x_data_2, y_data_2, xlab, cfg["y_label"], cfg["lrMethods"],
                                                                                           saveFileName=save_to2, funcs_=func_group,
                                                                     fontsize=40, markersize=45, linewidth=12,linestyle="--",
                                                                     Players2See=cfg["Players2See"],
                                                ylog_scale=cfg["ylog_scale"], pltText=cfg["pltText"], step=cfg["plot_step"])
                else:
                    if cfg["metric"] in ["Bid", "Avg_Bid"] :
                        baseline = z_ne.detach().numpy() * np.ones_like(y_data_2[0])
                        y_data_2.append(np.array(baseline))
                        func_group.insert(0, cfg["Hybrid_funcs"][0][0])

                        func_group.append("Non-hybrid")
                    elif cfg["metric"] in ["Relative_Efficienty_Loss"]:
                        cfg["y_label"] = r"$\rho(z(T))$"
                        baseline = RLoss.detach().numpy() * np.ones_like(y_data_2[0])
                        y_data_2.append(np.array(baseline))
                        func_group.append("Non-hybrid")
                    elif cfg["metric"] == "epsilon_error":
                        cfg["y_label"] = r"$\epsilon(z(T))$"
                        baseline = cfg["tol"] * np.ones_like(y_data_2[0])
                        y_data_2.append(np.array(baseline))
                        func_group.append("Non-hybrid")

                    save_to2 = cfg['metric'] + f"_alpha{cfg['alpha']}_gamma{cfg["gamma"]}_player"
                    #xlab = rf"$A\alpha_{{{funcs_[0]}}}$"
                    xlab = cfg["x_label"]
                    x_data_2 = np.array(cfg["Nb_A1"]) / cfg["n"] * 100
                    if cfg["metric"] in ["Speed"]:
                        cfg["y_label"] =  "‖BR(z(T)) − z(T)‖"

                    if num_hybrid_set == 1 and num_hybrids == 1:
                        x_data_2 = x_data
                    else:
                        xlab = rf"$\alpha_{{{funcs_[0]}}}$"
                    figpath_plot, figpath_zoom, figpath_legend  = plotGame_dim_N_last(cfg, x_data_2, y_data_2, xlab, cfg["y_label"], cfg["lrMethods"],
                                                                                       saveFileName=save_to2, funcs_=func_group,
                                                                 fontsize=40, markersize=45, linewidth=12,linestyle="--",
                                                                 Players2See=cfg["Players2See"],
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
        for i, method in enumerate(LEGENDS):#(cfg["lrMethods"]):
            if method in results['methods']:
                with cols[i]:
                    st.metric(
                        label=method,
                        value=f"{results['methods'][method]['convergence_iter']} iterations",
                        help=f"Last error: {results['methods'][method]['Speed'][-1]:.6f}"
                    )

        with cols[-1]:
            st.metric(
                label="Optimal",
                value=f"LSW: {results['optimal']['LSW']:.2f}",
                help=f"SW: {results['optimal']['SW']:.2f}"
            )


        # --- Plot PDF ---

        with open(figpath_plot, "rb") as f:
            plot_bytes = f.read()
        #with open(figpath_plot2, "rb") as f:
        #    plot_bytes2 = f.read()

        # --- Legend PDF ---
        with open(figpath_legend, "rb") as f:
            legend_bytes = f.read()
        # --- Zoom PDF ---
        with open(figpath_zoom, "rb") as f:
            zoom_bytes = f.read()

        # Put buttons on the same row
        # --- Download buttons in one row ---
            # =====================================================
            # 📘 Step 6: PDF Downloads
            # =====================================================
            st.subheader("📂 Download Outputs")
            btn_cols = st.columns(3)
            try:
                with open(figpath_plot, "rb") as f1, open(figpath_legend, "rb") as f2, open(figpath_zoom, "rb") as f3:
                    btn_cols[0].download_button("⬇️ Plot PDF", f1, file_name=figpath_plot)
                    btn_cols[1].download_button("⬇️ Legend PDF", f2, file_name=figpath_legend)
                    btn_cols[2].download_button("⬇️ Zoom PDF", f3, file_name=figpath_zoom)
            except:
                st.info("PDF files not available yet.")
    else:
         st.info("ℹ️ No results yet. Click ▶️ Run Simulation to start.")

except Exception:
       st.info("ℹ️ No results available yet. Please press **▶️ Run Simulation** to start.")

# -----------------------
# SIMULATION TABLE
# -----------------------

if st.button("📊 Run Simulation Table"):
    with st.spinner("Simulating..."):
        results_table, results_table_eps = run_simulation_table_avg(cfg, GameKelly)
        st.success("Done.")

        st.session_state.results_table = results_table
        st.session_state.results_table_eps = results_table_eps
        st.session_state.config = cfg
        #display_results_streamlit_dict(results_table, cfg, save_path="results/table_results.csv")

if "results_table" in st.session_state:
    try:
        results_table = st.session_state.results_table
        display_results_streamlit_dict(results_table, cfg, save_path="results/table_results.csv")
    except Exception:
        st.info("ℹ️ No results available yet. Please press **📊 Run Simulation Table** to start.")
if "results_table_eps" in st.session_state:
    try:
        results_table_eps = st.session_state.results_table_eps
        display_results_streamlit_dict(results_table_eps, cfg, save_path="results/table_results.csv")
    except Exception:
        st.info("ℹ️ No results available yet. Please press **📊 Run Simulation Table** to start.")

if st.button("📈 Run Jain(gamma)"):

    with st.spinner("Running Jain index simulations over γ and n..."):
        jain_results, gamma_grid = run_jain_vs_gamma(cfg, GameKelly)
    st.success("Done.")

    st.session_state.jain_results = jain_results
    st.session_state.config = cfg
    st.session_state.gamma_grid = gamma_grid

if 'jain_results' in st.session_state:
    # --- Retrieve data from session ---
    jain_results = st.session_state.jain_results
    config = st.session_state.config
    gamma_grid = st.session_state.gamma_grid

    fig = plot_jain_vs_gamma(jain_results, gamma_grid, config)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("📂 Download Outputs")
    figpath_plot = plotGame_Jain(cfg,jain_results, gamma_grid,
    ylog_scale=False, fontsize=40, markersize=40, linewidth=12,
    linestyle="-", pltText=False, step=1,tol=1e-6
)
    btn_cols = st.columns(1)
    try:
        with open(figpath_plot, "rb") as f1:
            btn_cols[0].download_button("⬇️ Plot PDF", f1, file_name=figpath_plot)
    except:
        st.info("PDF files not available yet.")
    else:
         st.info("ℹ️ No results yet. Click 📈 Run Jain(gamma) to start.")