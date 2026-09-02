"""User-facing guide to the experiments exposed by EdgeKelly Lab."""

import streamlit as st

st.set_page_config(page_title="Experiment Guide · EdgeKelly Lab", page_icon="🧭", layout="wide")

st.title("Experiment guide")
st.write(
    "Use this page to choose the right workflow before launching a potentially long run. "
    "All computations are started from the **Simulator** page."
)
st.caption("Return to **Simulator** from the page navigation to run an experiment.")

st.info(
    "Recommended first run: 5–10 players, 500 iterations, one random repetition, "
    "α = 1, and OGD_F plus BR. Increase the horizon and repetitions only after "
    "checking the configuration."
)

experiments = [
    ("1 · Learning dynamics", "Compare bid, payoff, welfare, fairness, or residual trajectories for several algorithms.", "Choose a metric and learning methods in the sidebar, then press Run Simulation."),
    ("2 · Convergence versus μ", "Record the first iteration at which the Nash residual reaches the tolerance.", "Use the log plot for trends, the heatmap for comparison, and ranked bars for one μ."),
    ("3 · Parameter-sweep tables", "Create numerical summaries across lists of player counts and heterogeneity values.", "Set list_n and list_gamma under Advanced parameters."),
    ("4 · Fairness versus heterogeneity", "Track Jain's index as player valuations become more heterogeneous.", "Use this for a direct fairness comparison across γ and n."),
    ("5 · Linear versus logarithmic utility", "Compare α = 0 with α = 1 using the same heterogeneity grid.", "This isolates the effect of utility curvature."),
    ("6 · Multi-resource allocation", "Model several capacities, resource requirements, and per-resource importance factors.", "Select algorithms together, inspect residuals, allocations, and player shares, then prepare PDFs."),
]

for title, purpose, usage in experiments:
    with st.container(border=True):
        st.subheader(title)
        st.write(purpose)
        st.caption(usage)

st.header("Reading the results")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Residual")
    st.write("A value approaching zero indicates proximity to a fixed point or Nash equilibrium. Always check the displayed definition.")
with col2:
    st.subheader("Efficiency")
    st.write("Higher normalized efficiency is better. Efficiency loss and Price of Anarchy use the opposite direction: lower is better.")
with col3:
    st.subheader("Fairness")
    st.write("Jain's index lies between 0 and 1; values closer to 1 indicate more even allocations.")

st.header("Performance guidance")
st.markdown(
    """
- Runtime grows with players × iterations × methods × repetitions.
- Exact best-response residuals are more expensive than iterate differences. Use a sampling interval above 1 for exploration.
- In the multi-resource experiment, exact BR is available for α = 0, 1, and 2.
- Use **T_plot** to shorten displayed curves without rerunning a stored simulation.
- Generate PDFs after selecting the algorithm and α you intend to export.
"""
)
