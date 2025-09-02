SIMULATION_CONFIG = {
    "lrMethods": ["DAQ","OGD","SBRD"],
    "Hybrid_funcs": ["DAQ", "DAH"],
    "T": 100,
    "alpha": 1,
    "n": 3,
    "eta": 0.5,
    "price": 1.0,
    "a": 100,
    "mu": 0,
    "c": 4000,
    "delta": 0.1,
    "epsilon": 1e-3,
    "tol": 1e-5,
    "lr_vary": False,
    "IdxConfig": 1,
    "x_label": "Time step (t)",
    "metric": "utility",
    "y_label": r"$\varphi (z)$",
    "ylog_scale": False,
    "plot_step": 10,
    "saveFileName": "results/plot_",
    "pltText": False,
    "gamma": 0.5
}

SIMULATION_CONFIG_table = {
    # Simulation parameters
    "T": 4000,                     # Total number of iterations in the learning process
    "eta": 0.5,                    # Step size for updating bids
    "price": 1.0,                  # Price parameter in the utility or game setup
    "a": 100,                      # Parameter controlling the heterogeneity of utility functions
    "mu": 0,                       # Exponent applied to c_vector to adjust costs
    "c": 4000,                     # Base value for c_vector (resource costs)
    "delta": 0.1,                  # Regularization parameter in the game formulation
    "epsilon": 1e-3,               # Epsilon parameter to avoid numerical issues (e.g., division by zero)
    "alpha": 1,                    # Fairness parameter for the alpha-fair utility (alpha = 0 means log utility)
    "tol": 1e-5,                   # Tolerance threshold for stopping criteria (convergence)
    "save_path": "results_table.csv",  # Path to save the result (.csv file)

    # Learning methods to compare in the simulation
    "lrMethods":  ["DAH","DAQ", "SBRD"],  # List of learning methods: SBRD = Best Response, DAQ = Dual Averaging Quadratic, XL = Extra Learning
    "Hybrid_funcs": ["DAQ", "DAH"],

    # Range of values for experiment parameters
    "list_n": [2, 3, 5],       # List of numbers of players to simulate
    "list_gamma": [0.0, 0.5]  # List of heterogeneity exponents for a_vector
}