SIMULATION_CONFIG = {
    "lrMethods": ["DAQ","DAE","OGD","SBRD"],
    "Hybrid_funcs": ["DAQ", "DAE"],
    "T": 2000,  # Number of iterations in the learning process
    "alpha": 1,  # Fairness parameter in utility (e.g., α-fair utility)
    "n": 10,  # Number of players in the game
    "eta": 0.5,  # Step size for the learning update
    "price": 1.0,  # Price parameter in the game (can represent a resource price)
    "a": 1000,  # Parameter for the utility function heterogeneity (a_i)
    "a_min": 1,
    "mu": 0,  # Exponent controlling the heterogeneity of the c_vector
    "c": 4000,  # Constant part of the c_vector
    "delta": 0.1,  # Delta parameter (could model uncertainty, slack, or safety margin)
    "epsilon": 1,  # Regularization term (to avoid division by zero, for stability)
    "Nb_random_sim": 1,
    "Players2See": [1],
    "tol": 1e-5,  # Tolerance threshold for considering the game as converged
    "lr_vary": False,  # Learning rate vary or not
    "var_init": 100,
    "IdxConfig": 1,  # Configuration index to select the regularizer or the response method
    "x_label": "Time step (t)",  # Label for the x-axis in the output plot
    "metric": "SW",  # "Speed" or "lpoa",or "LSW" , "Bid", "utility", "Dist-2-Opt-SW"
    "y_label": "SW",
    # "||BR(z) -z||"; Social Welfare, Distance To Optimal SW r"$\varphi (z)$" Label for the y-axis in the output plot (error between best response and current state)
    "ylog_scale": False,
    # Whether to use a logarithmic scale on the y-axis in the plot, recommended for speed's convergence plot
    "plot_step": 200,
    "saveFileName": "Hybrid_OGD+SBRD_",  # Prefix for the filename where results/plots are saved Hybrid_DAQ+SBRD
    "pltText": True,  # Whether to display text annotations on the plot
    "gamma": 0.0  # Exponent controlling the heterogeneity of the a_vector
}
SIMULATION_CONFIG["Hybrid_sets"] = [list(range(0, 1)), list(range(1, int(SIMULATION_CONFIG["n"])))]
SIMULATION_CONFIG["legends"] = SIMULATION_CONFIG["lrMethods"]

SIMULATION_CONFIG_table = {
    # Simulation parameters
    "save_path": "results_table.csv",  # Path to save the result (.csv file)

    # Learning methods to compare in the simulation
    "lrMethods":  ["DAH","DAQ", "SBRD"],  # List of learning methods: SBRD = Best Response, DAQ = Dual Averaging Quadratic, XL = Extra Learning
    "Hybrid_funcs": ["DAQ", "DAH"],

    # Range of values for experiment parameters
    "list_n": [2, 3, 5, 10],       # List of numbers of players to simulate
    "list_gamma": [0, 5, 10]  # List of heterogeneity exponents for a_vector
}