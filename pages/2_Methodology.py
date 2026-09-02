"""Concise methodology reference for EdgeKelly Lab."""

import streamlit as st

st.set_page_config(page_title="Methodology · EdgeKelly Lab", page_icon="📘", layout="wide")

st.title("Methodology and notation")
st.caption("Return to **Simulator** from the page navigation to run an experiment.")

st.header("Kelly allocation")
st.write("Players submit nonnegative bids. A bid receives a proportional share of the available resource.")
st.latex(r"x_i=C\frac{z_i}{\sum_j z_j+\delta}")
st.write("Here C is capacity and δ is allocation slack. In the multi-resource game, this equation is applied independently to every resource r.")

st.header("α-fair utility")
st.latex(r"V^\alpha(x)=\begin{cases}x,&\alpha=0,\\ \log x,&\alpha=1,\\ x^{1-\alpha}/(1-\alpha),&\text{otherwise.}\end{cases}")
st.write("Increasing α places progressively more emphasis on allocations received at the lower end of the distribution.")

st.header("Learning algorithms")
algorithm_rows = [
    {"Method": "BR", "Update": "Synchronous best response", "Step schedule": "None", "Interpretation": "Each player maximizes current payoff against opponents' previous bids."},
    {"Method": "OGD_F", "Update": "Projected current gradient", "Step schedule": "Fixed horizon", "Interpretation": "Stable projected gradient ascent with η proportional to 1/√T."},
    {"Method": "OGD_V", "Update": "Projected current gradient", "Step schedule": "Varying", "Interpretation": "Projected gradient ascent with η_t proportional to 1/√t."},
    {"Method": "DAQ_F", "Update": "Projected cumulative gradient", "Step schedule": "Fixed horizon", "Interpretation": "Quadratic dual averaging using all gradients observed so far."},
    {"Method": "RRM_V", "Update": "Regularized-Robbins Monro", "Step schedule": "Varying", "Interpretation": "Reinforcement-style score accumulation with η_t proportional to 1/√t."},
    {"Method": "NumBR", "Update": "Numerical best response", "Step schedule": "None", "Interpretation": "Uses numerical optimization when an analytical response is unavailable."},
    {"Method": "XL", "Update": "Exponential learning", "Step schedule": "Configured", "Interpretation": "Maps accumulated scores to feasible actions through a smooth choice map."},
]
st.dataframe(algorithm_rows, use_container_width=True, hide_index=True)

st.header("Core diagnostics")
st.latex(r"R_{BR}^t=\lVert z^{BR}(z^t)-z^t\rVert_2")
st.write("The best-response residual directly tests the Nash fixed-point condition.")
st.latex(r"R_{step}^t=\lVert z^{t+1}-z^t\rVert_2")
st.write("The iterate difference is cheaper, but a small step alone does not always establish Nash convergence.")
st.latex(r"J(x)=\frac{(\sum_i x_i)^2}{n\sum_i x_i^2}")
st.write("Jain's index equals 1 for an equal allocation among the included players.")

st.header("Reproducibility")
st.write(
    "Download the JSON configuration from the simulator sidebar and retain it with exported figures. "
    "Report the random-repetition count, horizon, tolerance, residual definition, α, δ, ε, and learning-rate schedule."
)
