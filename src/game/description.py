# descriptions.py

ALGO_DESCRIPTIONS = {
    "None": "",

    "SBRD": """Synchronous Best-Response Dynamics (SBRD):
An iterative learning algorithm where all agents update their strategies simultaneously,
computing the best response at each step, typically uses analytical expressions for best responses for alpha = 0, 1, and 2
. """,

    "OGD": """Online Gradient Descent (OGD):
A no-regret learning algorithm where players iteratively adjust their bids
based on the gradient of their utility function.
Converges under convexity assumptions. """,

    "DAQ": """Dual Averaging with Quadratic Regularization (DAQ):
A dual-based learning algorithm where each player updates bids
via accumulated gradients and regularization. Balances stability and convergence
in budget-constrained resource allocation games. """,

    "DAH": """Dual Averaging with Historical Gradients (DAH):
Extension of DAQ that incorporates a history of past gradients
to smooth updates and improve stability. """,

    "XL": """Extra-Gradient Learning (XL):
A two-step gradient method to improve convergence in games with strategic interactions.
Uses a prediction step before the actual update. """,

    "NumSBRD": """Numerical Synchronous Best-Response Dynamics (NumSBRD):
Similar to SBRD, but typically uses numerical approximation best responses (10^-5). """
}
