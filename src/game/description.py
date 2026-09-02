"""Plain-language descriptions displayed in the simulator."""

ALGO_DESCRIPTIONS = {
    "None": "",
    "BR": (
        "**Best response (BR).** At each iteration, every player chooses the bid "
        "that maximizes its current payoff while the other players' previous bids "
        "are held fixed. The simulator has analytical responses for α = 0, 1, and 2."
    ),
    "OGD": (
        "**Online gradient descent/ascent (OGD).** Each player follows its current "
        "payoff gradient and projects the result onto its feasible bid interval or "
        "budget set. Fixed and decreasing step-size variants are available."
    ),
    "DAQ": (
        "**Dual averaging with quadratic regularization (DAQ).** Players accumulate "
        "past payoff gradients, scale the resulting score, and project it onto the "
        "feasible action set. Accumulation can make updates smoother than OGD."
    ),
    "DAH": (
        "**Dual averaging with historical gradients (DAH).** This score-based rule "
        "uses a weighted history of gradients to reduce sensitivity to an individual "
        "iteration and improve update stability."
    ),
    "RRM": (
        "**Regularized reinforcement learning (RRM).** Players maintain a cumulative "
        "gradient score and map it back to feasible bids. The varying variant weights "
        "new gradients with a decreasing step size."
    ),
    "XL": (
        "**Exponential learning (XL).** Players update internal scores from payoff "
        "gradients and use a smooth exponential choice map to obtain feasible bids."
    ),
    "NumBR": (
        "**Numerical best response (NumBR).** This follows best-response dynamics but "
        "solves each player's optimization numerically when a closed form is not used."
    ),
}
