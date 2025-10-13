import numpy as np
import torch
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker

from scipy.optimize import root_scalar
import sympy as sp

colors = [
    "slategray",  # Gris ardoise
    "brown",  # Marron
    "magenta",  # Magenta
    "teal",  # Bleu-vert

    "salmon",      # Saumon
    "lime",        # Vert clair
    "navy",        # Bleu marine
    "coral",       # Corail
    "darkgreen",   # Vert foncé
    "orchid",      # Orchidée (rose-violet)

    "darkkhaki"    # Kaki clair
    "gold",  # Doré
    "black",  # Noir
    "crimson",  # Rouge profond
    "darkcyan",  # Cyan foncé
    "indigo",  # Bleu indigo
]
METHODS = ["DAQ", "DAE", "OGD", "SBRD"]
COLORS_METHODS = {
   "DAQ"  : "darkorange",  # Orange foncé
   "DAE"  : "royalblue",  # Bleu vif
   "OGD"  : "green",  # Vert
   "SBRD" : "purple",  # Violet
}

colors22 = [



    "slategray",  # Gris ardoise
    "brown",  # Marron
    "magenta",  # Magenta
    "teal",  # Bleu-vert
]
MARKERS_METHODS = {
    "DAQ": "s",  # Orange foncé
    "DAE": "^",  # Bleu vif
    "OGD": "v",  # Vert
    "SBRD": "D", # Violet
}



markers = ["H", "d","*","p", "|", "s", "^", "v", "D", "*", "p", "x", "+", "|","s", "^", "v", "D", "*", "p", "x", "+", "|"]
markers22 = ["H", "d","*","p"]

def solve_quadratic(n, a, delta):
    delta = torch.tensor(delta)
    a = torch.tensor(a)
    n = torch.tensor(n)

    A = n
    B = delta - a * (n - 1)
    C = -a * delta

    disc = B**2 - 4*A*C
    sqrt_disc = torch.sqrt(disc)

    z1 = (-B + sqrt_disc) / (2*A)
    z2 = (-B - sqrt_disc) / (2*A)

    return z1#, z2

def solve_nonlinear_eq(a, s, alpha, eps, c_vector, price=1.0, max_iter=100, tol=1e-3):
    """
    Solves for z in: price * (z + s_i)^(2 - alpha) * z^alpha = a_i * s_i
    for each i, using the bisection method.
    """
    a = a.numpy()
    s = s.numpy()
   # c_vector = c_vector.numpy()
    n = len(a)
    z_list = []

    for i in range(n):
        def f(z):
            return price * (z + s[i]) ** (2 - alpha) * z ** alpha - a[i] * s[i]

        # Ensure the bracket is valid
        lower_bound = tol
        upper_bound = c_vector[i] / price

        if f(lower_bound) * f(upper_bound) > 0:
            br = Q1(lower_bound*torch.ones(n), eps, c_vector, price)
            return br

        sol = root_scalar(f, bracket=[lower_bound, upper_bound], method='bisect', xtol=tol)

        if  sol.converged:
            z_list.append(sol.root)
        else:
            z_list.append(lower_bound)

    br = Q1(torch.tensor(z_list, dtype=torch.float32), eps, c_vector, price)
    return br
def compute_optimal_x(c_vector, a_vector, eps, delta: float, d_vector: np.ndarray):
    def min_fraction(eps: np.ndarray, budgets: np.ndarray, delta: float):
        return eps / (eps + np.sum(budgets) - budgets + delta)

    def max_fraction_LSW(d_vector: np.ndarray, c_vector: np.ndarray, a_vector: np.ndarray):
        return np.exp((c_vector - d_vector) / a_vector)
    a_vector = np.array(a_vector)
    eps= np.array(eps)
    d_vector= np.array(d_vector)
    c_vector= np.array(c_vector)

    eps_x = min_fraction(eps, c_vector, delta)
    C = np.sum(c_vector)
    S = C / (C + delta)  # Total resource available
    gamma_x = max_fraction_LSW(d_vector, c_vector, a_vector)

    # Define the function that computes the total allocated z for a given lambda.

    def total_x(lagrange_mult: float):
        # Ideal candidate from first-order condition: a_i / lambda
        lagrange_mult = np.maximum(lagrange_mult, 1e-10*np.ones_like(lagrange_mult))
        x_candidate = a_vector / lagrange_mult
        # Clip each coordinate to the allowable interval [eps, z_sat]
        x_candidate = np.maximum(x_candidate, eps_x)
        x_candidate = np.minimum(x_candidate, gamma_x)
        return np.sum(np.array(x_candidate)) - S

    lmbda_min = np.min(a_vector / gamma_x)
    lmbda_max = np.max(a_vector / eps_x)
    if total_x(lmbda_min) <= 0:
        return gamma_x
    if total_x(lmbda_max) > 0:
        return eps_x

    #print(f"total_x(lmbda_min):{total_x(lmbda_min)},\n total_x(lmbda_max):{total_x(lmbda_max)}")
    lmbda = optimize.bisect(total_x, lmbda_min, lmbda_max)

    return np.minimum(np.maximum(a_vector / lmbda, eps_x), gamma_x)


def x_log_opt(c_vector, a_vector, d_vector, eps, delta, price, bid0):
    x_opt = compute_optimal_x(c_vector, a_vector, eps, delta, d_vector)#gradient_descent(bid0,c_vector, a_vector, eps, delta, d_vector,price)#
    x_opt = torch.tensor(x_opt, dtype=torch.float64)
    #x_opt = gradient_descent(bid0,c_vector, a_vector, eps, delta, d_vector,price)
    return x_opt# LSW_func(x_opt, c_vector, a_vector, d_vector, 1)


def V_func(x, alpha):
    if alpha == 1:
        V = torch.log(x)
    else:
        V = 1 / (1 - alpha) * (x) ** (1 - alpha)
    return V

def Q1(acc_gradient, eps, c, price):
    return torch.minimum(torch.maximum(eps/price, acc_gradient), c/price)


def Q2(acc_gradient, eps, c, price):
    return torch.maximum(eps/price, torch.minimum(torch.exp(acc_gradient - 1), c/price))





def BR_alpha_fair(eps, c_vector, z: torch.Tensor, p,
                  a_vector: torch.Tensor, delta, alpha, price: float, b=0):
    """Compute the best response function for an agent."""
    #p = torch.tensor(p, dtype=torch.float32)  # Ensure p is a tensor
    a_vector = a_vector.to(dtype=torch.float32)

    if alpha == 0:
        br = -p + torch.sqrt(a_vector * p / price)


    elif alpha == 1:
        if b == 0:
            br = (-p + torch.sqrt(p ** 2 + 4 * a_vector * p / price)) / 2
        else:
            #valid = (p > 0) & (p <= a_vector / (b * price))
            discriminant = p ** 2 + 4 * a_vector * p * (1 + b) / price
            br = (-p * (2 * b + 1) + torch.sqrt(discriminant)) / (2 * (1 + b))

    elif alpha == 2:
        br = torch.sqrt(a_vector * p / price)

    return  Q1(br, eps, c_vector, price)

def Valuation(x, a_vector, d_vector, alpha):
    V = V_func(x, alpha)
    return a_vector * V + d_vector
def Payoff(x, z, a_vector, d_vector, alpha, price):
    U = Valuation(x, a_vector, d_vector, alpha) - price*z
    return U

def SW_func(x, budgets, a_vector, d_vector, alpha):
    V = a_vector * V_func(x, alpha)
    sw = torch.sum(V)
    return sw

def LSW_func(x, budgets, a_vector, d_vector, alpha):
    V = a_vector *V_func(x, alpha)
    V_bounded = torch.minimum(V, budgets)
    lsw = torch.sum(V_bounded)
    return lsw


import numpy as np


def compute_G(a_i, delta, epsilon,c, n):
    """
    Compute Lipschitz constant G for grad(phi_i).

    Parameters
    ----------
    a_i : float
        Coefficient a_i in phi_i
    delta : float
        Regularization parameter delta
    epsilon : float
        Minimum allowed bid (z_i >= epsilon)
    n : int
        Number of players

    Returns
    -------
    G : float
        Upper bound of Lipschitz constant
    """
    smin = n * epsilon + delta
    smax = n * c + delta
    term_main = a_i * (1 / epsilon) + 1
    term_others = (n - 1) * (a_i / (n*epsilon + delta)) ** 2 *0
    G = np.sqrt(term_main ** 2 + term_others)
    G = max(abs(a_i*smax/(epsilon *(epsilon+smax)) - 1), abs(a_i*smin/(epsilon *(epsilon+smin)) - 1))
    return G

def compute_G_DAE(a_i, delta, epsilon,c, n):
    """
    Compute Lipschitz constant G for grad(phi_i).

    Parameters
    ----------
    a_i : float
        Coefficient a_i in phi_i
    delta : float
        Regularization parameter delta
    epsilon : float
        Minimum allowed bid (z_i >= epsilon)
    n : int
        Number of players

    Returns
    -------
    G : float
        Upper bound of Lipschitz constant
    """
    smin = delta + n * epsilon
    smax = delta + n * c

    def G(z, s):
        inner = a_i * s / (z * (z + s)) - 1.0
        return z * (inner ** 2)

    # Evaluate corners
    corners = [
        ('(eps,smin)', epsilon, smin),
        ('(eps,smax)', epsilon, smax),
        ('(c,smin)', c, smin),
        ('(c,smax)', c, smax),
    ]

    G_vals = {name: G(z, s) for (name, z, s) in corners}
    name_max = max(G_vals, key=G_vals.get)
    G_max = G_vals[name_max]
    G = np.sqrt(G_max)
    return G


class GameKelly:
    def __init__(self, n: int, price: float,
                 epsilon, delta, alpha, tol):


        self.n = n


        self.price = price
        self.epsilon = epsilon
        self.delta = delta
        self.alpha = alpha
        self.tol = tol

    def fraction_resource(self, z):
        return z / (torch.sum(z) + self.delta)



    def grad_phi(self,phi, bids):
        z = bids.clone().detach()
        #x = self.fraction_resource(z)
        s = torch.sum(z) - z +self.delta

        jacobi = torch.autograd.functional.jacobian(phi, z)#self.a_vector * s / (z + s)**(2 - self.alpha) * z**(self.alpha) - self.price #

        return jacobi.diag()

    def check_NE(self, z: torch.tensor, a_vector, c_vector, d_vector,):
        p = torch.sum(z) - z + self.delta
        if self.alpha  not in [0,1,2]:
            err = torch.norm(solve_nonlinear_eq(a_vector, p, self.alpha, self.epsilon, c_vector, self.price, max_iter=1000, tol=self.tol)
                                           - z)#, self.tol * torch.ones(1))
        else:
            err =  torch.norm(BR_alpha_fair(self.epsilon, c_vector, z, p,
                                            a_vector, self.delta, self.alpha, self.price,
                                            b=0) - z)#, self.tol * torch.ones(1))

        return err   # torch.norm(self.grad_phi(z))

    def AverageBid(self, z,t):
        z_copy = z.clone()
        z_t = 1/t*torch.sum(z,dim=0)
        return z_t
    def Regret(self,  bids,t,a_vector, c_vector, d_vector,):
        def phi( z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector
        n = bids[0].shape[0]
        p = torch.sum(bids[t-2]) - bids[t-2] + self.delta
        z_t = BR_alpha_fair(self.epsilon, c_vector, bids[t-2], p,
                                         a_vector, self.delta, self.alpha, self.price, b=0)

        Reg = 1/n * torch.sum(torch.abs(phi(bids[t-1]) - phi(z_t)))
        return torch.maximum(Reg,self.tol*torch.ones(1))

    def XL(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad,p=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):

        def phi( z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector

        acc_grad_copy = acc_grad.clone()
        grad_t = self.grad_phi(phi, bids)

        if vary:
            acc_grad_copy += grad_t / (t ** eta)
        else:
            acc_grad_copy += grad_t * eta
        z_t = torch.maximum(self.epsilon / self.price, c_vector / (1 + torch.exp(-acc_grad_copy)))
        return z_t, acc_grad_copy

    def Hybrid(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, p=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):

        acc_grad_copy = acc_grad.clone()

        z_t = torch.zeros_like(bids)
        for idx_set, func in enumerate(Hybrid_funcs):
            func_ = getattr(self, func)
            p_ = p[idx_set]
            #print(f"Hybrid_sets:{Hybrid_sets}, {Hybrid_sets[idx_set][:]},{z_t[Hybrid_sets[idx_set]],}")
            z_t[Hybrid_sets[idx_set]], acc_grad_copy[Hybrid_sets[idx_set]] = func_(t, a_vector[Hybrid_sets[idx_set]], c_vector[Hybrid_sets[idx_set]],
                                                                    d_vector[Hybrid_sets[idx_set]], eta, bids[Hybrid_sets[idx_set]], acc_grad[Hybrid_sets[idx_set]],p=p_, vary=vary)
        return z_t, acc_grad_copy



    def OGD2(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad,p=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):

        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector

        grad_t = self.grad_phi(phi, bids)


        if vary:
            eta_t = 1 / (t ** eta) if t > 0 else eta
        else:
            eta_t = eta
        z_candidate = bids + eta_t * grad_t
        z_t = Q1(z_candidate, self.epsilon, c_vector, self.price)
        return z_t, acc_grad

    def OGD(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad,
            p=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):

        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector

        grad_t = self.grad_phi(phi, bids)

        # Taille de domaine (diamètre) : borne supérieure
        D = c_vector - self.epsilon ##np.linalg.norm(c_vector - self.epsilon)  # si c est borne sup

        # Constante Lipschitz
        n = len(bids)
        # ⚠️ ici je suppose a_vector est scalaire (pour un joueur).
        # Si c’est un vecteur (multi-joueur), on prend max(a_i).
        a_i = torch.max(a_vector)
        G = torch.zeros_like(a_vector)
        for i in range(n):
            G[i] = compute_G(a_vector[i], self.epsilon,c_vector[i], self.epsilon, n)

        # Step-size (theorem 3.1 Hazan)
        if vary:
            # Step-size (theorem 3.1 Hazan)
            eta_t = D / (G * np.sqrt(t)) if t > 0 else eta
        else:
            eta_t = eta  # fallback pour t=0

        # Update rule
        z_candidate = bids + eta_t * grad_t
        z_t = Q1(z_candidate, self.epsilon, c_vector, self.price)

        return z_t, acc_grad

    def DAQ(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad,p=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):

        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector

        acc_grad_copy = acc_grad.clone()
        grad_t = self.grad_phi(phi, bids)
        n = len(bids)

        D = torch.sqrt(c_vector **2 -  self.epsilon **2) # si c est borne sup
        # ⚠️ ici je suppose a_vector est scalaire (pour un joueur).
        # Si c’est un vecteur (multi-joueur), on prend max(a_i).
        a_i = torch.max(a_vector)
        G = torch.zeros_like(a_vector)
        for i in range(n):
            G[i] = compute_G(a_vector[i], self.epsilon,c_vector[i], self.epsilon, n)

        if vary:
        # Step-size (theorem 3.1 Hazan)
            eta_t = D / (G * np.sqrt(t)) if t > 0 else eta
        else:
            eta_t = eta  # fallback pour t=0

        if t%200==0:
            print(f"eta_t :{eta_t}")
        acc_grad_copy += grad_t * eta_t
        z_t = Q1(acc_grad_copy, self.epsilon, c_vector, self.price)
        return z_t, acc_grad_copy

    def DAE(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad,p=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector

        acc_grad_copy = acc_grad.clone()
        grad_t = self.grad_phi(phi, bids)
        # Taille de domaine (diamètre) : borne supérieure


        # Constante Lipschitz
        n = len(bids)
        D = torch.sqrt(c_vector * torch.log(c_vector) -  self.epsilon * torch.log(self.epsilon))  # si c est borne sup
        # ⚠️ ici je suppose a_vector est scalaire (pour un joueur).
        # Si c’est un vecteur (multi-joueur), on prend max(a_i).
        a_i = torch.max(a_vector)
        G = torch.zeros_like(a_vector)
        for i in range(n):
            G[i] = compute_G_DAE(a_vector[i], self.epsilon, c_vector[i], self.epsilon, n)


        # Step-size (theorem 3.1 Hazan)
        if vary:
            # Step-size (theorem 3.1 Hazan)
            eta_t = D / (G * np.sqrt(t)) if t > 0 else eta
        else:
            eta_t = eta  # fallback pour t=0
        acc_grad_copy += grad_t * eta_t
        z_t = Q2(acc_grad_copy, self.epsilon, c_vector, self.price)

        return z_t, acc_grad_copy

    def SBRD(self, t, a_vector, c_vector, d_vector,  eta, bids, acc_grad, p=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        #p = torch.sum(bids) - bids + self.delta
        z_t = BR_alpha_fair(self.epsilon, c_vector, bids, p,
                                         a_vector, self.delta, self.alpha, self.price, b=0)

        return z_t, acc_grad

    def Fict_SBRD(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, p=0, vary=False,
            Hybrid_funcs=None, Hybrid_sets=None):
        """
        Synchronous Fictitious Play Best-Response Dynamics (Fict_SBRD).

        Each player best-responds to the empirical average of opponents' past bids
        rather than only the current bids.
        """
        # Best response to the averaged bids
        z_t = BR_alpha_fair(self.epsilon, c_vector, bids, p,
                                         a_vector, self.delta, self.alpha, self.price, b=0)
        # Empirical average of past bids (up to iteration t)

        if vary:
            avg_bids = eta/t * acc_grad  # acc_grad stores cumulative bids
        else:
            avg_bids = eta * acc_grad

        # Update cumulative history for averaging
        acc_grad = z_t + avg_bids
        z_t = Q1(acc_grad, self.epsilon, c_vector, self.price)

        return z_t, acc_grad

    def NumSBRD(self,t, a_vector, c_vector, d_vector,  eta, bids, acc_grad, p=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        #p = torch.sum(bids) - bids + self.delta
        z_t = solve_nonlinear_eq(a_vector, p, self.alpha, self.epsilon,  c_vector, self.price, max_iter=100,tol=self.tol)

        z_t = Q1(z_t, self.epsilon, c_vector, self.price)
        return z_t, acc_grad

    def learning(self, func, a_vector, c_vector, d_vector, n_iter: int, eta, bids, vary: bool = False, stop=False, Hybrid_funcs=None, Hybrid_sets=None):
        func = getattr(self, func)


        acc_grad = torch.zeros(self.n, dtype=torch.float64)
        matrix_bids = torch.zeros((n_iter + 1, self.n), dtype=torch.float64)
       # agg_bids = matrix_bids.clone()
        vec_LSW = torch.zeros(n_iter + 1, dtype=torch.float64)
        vec_SW = torch.zeros(n_iter + 1, dtype=torch.float64)
        utiliy = torch.zeros((n_iter + 1, self.n), dtype=torch.float64)
        utiliy_residual = utiliy.clone()
        valuation = utiliy.clone()
        #agg_utility = torch.zeros((n_iter + 1, self.n), dtype=torch.float64)
        error_NE = torch.zeros(n_iter + 1, dtype=torch.float64)
        matrix_bids[0] = bids.clone()
       # agg_bids[0] = bids.clone()
        error_NE[0] = self.check_NE(bids,a_vector, c_vector, d_vector)
        utiliy[0] = Payoff(self.fraction_resource(matrix_bids[0]), matrix_bids[0], a_vector, d_vector, self.alpha, self.price)
        valuation[0] = Valuation(self.fraction_resource(matrix_bids[0]), a_vector, d_vector, self.alpha)
        #agg_utility[0] = utiliy[0].clone()
        vec_LSW[0] = LSW_func(self.fraction_resource(matrix_bids[0]), c_vector, a_vector, d_vector, self.alpha)
        vec_SW[0] = SW_func(self.fraction_resource(matrix_bids[0]), c_vector, a_vector, d_vector, self.alpha)
        z_br = BR_alpha_fair(self.epsilon, c_vector, matrix_bids[0], torch.sum(matrix_bids[0]) - matrix_bids[0] + self.delta, a_vector, self.delta, self.alpha, self.price,
                             b=0)
        utiliy_residual[0] = torch.abs(utiliy[0] - Payoff(self.fraction_resource(z_br), z_br, a_vector, d_vector, self.alpha, self.price))


        k = 0

        for t in range(1, n_iter + 1):

            k = t
            p = torch.sum(matrix_bids[t-1]) - matrix_bids[t-1] + self.delta
            matrix_bids[t], acc_grad = func(t, a_vector, c_vector, d_vector, eta, matrix_bids[t-1], acc_grad, p=p, vary=vary, Hybrid_funcs=Hybrid_funcs, Hybrid_sets=Hybrid_sets)
            error_NE[t] = self.check_NE(matrix_bids[t], a_vector, c_vector, d_vector,)
            vec_LSW[t] = LSW_func(self.fraction_resource(matrix_bids[t]), c_vector, a_vector, d_vector, self.alpha)

            z_br = BR_alpha_fair(self.epsilon, c_vector, matrix_bids[t], p, a_vector, self.delta, self.alpha, self.price, b=0)

            vec_SW[t] = SW_func(self.fraction_resource(matrix_bids[t]), c_vector, a_vector, d_vector, self.alpha)
            utiliy[t] = Payoff(self.fraction_resource(matrix_bids[t]), matrix_bids[t], a_vector, d_vector, self.alpha, self.price)
            utiliy_residual[t] = (utiliy[t] - Payoff(self.fraction_resource(z_br), z_br, a_vector, d_vector, self.alpha, self.price))
            valuation[t] = Valuation(self.fraction_resource(matrix_bids[t]), a_vector, d_vector, self.alpha)
            #agg_utility[t] = agg_utility[t-1] +  1/(t+1) * utiliy[t]
            err = torch.min(error_NE[:k])#round(float(torch.min(error_NE[:k])),3)
            #agg_bids[t] = 1/(t+1) * torch.sum(matrix_bids[:t], dim=0)#self.AverageBid(matrix_bids, t)
            if stop and err <= self.tol:
                break
        col = torch.arange(1, k + 1)
        agg_bids = torch.cumsum(matrix_bids[:k, :], dim=0)/ col.unsqueeze(1).expand(-1,self.n)
        Bids = [matrix_bids[:k, :], agg_bids[:k, :]]
        sw = torch.sum(utiliy[:k, :], dim=1); lsw = vec_LSW[:k]
        agg_utility = torch.cumsum(utiliy[:k, :], dim=0)

        agg_utility = agg_utility / col.unsqueeze(1).expand(-1,self.n)

        Utility_set = [utiliy[:k, :], agg_utility[:k, :], utiliy_residual[:k, :]]

        Welfare = [vec_SW[:k], vec_LSW[:k], valuation[:k]]
        return Bids, Welfare, Utility_set, torch.maximum(error_NE[:k],torch.tensor(self.tol))  #matrix_bids[:k, :], vec_LSW[:k], error_NE[:k], Avg_bids[:k, :], utiliy[:k, :]


def plotGame(config,
    x_data, y_data, x_label, y_label, legends, saveFileName,
    ylog_scale, fontsize=40, markersize=40, linewidth=12,
    linestyle="-", pltText=False, step=1,tol=1e-6
):
    plt.figure(figsize=(18, 12))
    y_data = np.array(y_data)

    plt.rcParams.update({'font.size': fontsize})
    x_data_copy = x_data.copy()

    if ylog_scale:
        plt.yscale("log")

    # --- Plot curves ---
    l=0
    for i, legend in enumerate(legends):
        color = "red" if legends[i] == "Optimal" else colors[i]
        marker = "" if legends[i] ==  "Optimal" else markers[i]
        try:
            if config["lrMethods"][i] in METHODS:

                if config["lrMethods"].count(config["lrMethods"][i])>1 and config["Learning_rates"][i]==0.05:
                    color = COLORS_METHODS[config["lrMethods"][i]]
                    marker = MARKERS_METHODS[config["lrMethods"][i]]
                elif config["lrMethods"].count(config["lrMethods"][i])==1:

                    color = COLORS_METHODS[config["lrMethods"][i]]
                    marker = MARKERS_METHODS[config["lrMethods"][i]]
            if config["Hybrid_funcs_"][i][1] in METHODS and config["Learning_rates"][i]==0.05:
                color = COLORS_METHODS[config["Hybrid_funcs_"][i][1]]
                marker = MARKERS_METHODS[config["Hybrid_funcs_"][i][1]]
                l+=1
        except:
            print()

        plt.plot(
            x_data[::step],
            (y_data[i])[::step],
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=1 * markersize,
            color=color,
            label=f"{legend}",
            markeredgecolor="black",
        )

        if pltText:
            last_x = x_data[-1]
            #if config["lrMethods"][i]=="OGD":
            #    last_y =8.32e-2
            #else:
            last_y = y_data[i][-1]
            plt.text(
                last_x, last_y,
                f"{last_y:.2e}",
                fontweight="bold",
                fontsize=fontsize,
                bbox=dict(facecolor='white', alpha=0.7),
                verticalalignment='bottom', horizontalalignment='right'
            )

    # --- Axis formatting ---
    ax = plt.gca()

    # --- Axis formatting ---
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")



    # --- Save plot without legend ---
    figpath_plot = f"{saveFileName}_plot.pdf"
    plt.savefig(figpath_plot, format="pdf")

    if l==len(legends):
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    plt.ylabel(str(f"{y_label}"), fontweight="bold", fontsize=2*fontsize)
    plt.xlabel(str(f"{x_label}"), fontweight="bold", fontsize=2*fontsize)
    plt.legend(frameon=False,prop={'weight': 'bold'})
    plt.grid(True)
    plt.tight_layout()

    # --- Save plot without legend ---
    figpath_plot = f"{saveFileName}_plot.pdf"
    plt.savefig(figpath_plot, format="pdf")

    # --- Build legend handles ---
    legend_handles = [
        Line2D([0], [0], color=("red" if legends[k] == "Optimal" else  COLORS_METHODS[legends[k]] if legends[k] in METHODS else colors[k]),
               markeredgecolor="black", linestyle=linestyle if linestyle != "" else "-",
               marker=("" if legends[k] == "Optimal" else MARKERS_METHODS[legends[k]] if legends[k] in METHODS else markers[k] ),
               markersize=markersize, linewidth=linewidth)
        for k in range(len(legends))
    ]

    # --- Save legend separately ---
    fig_legend = plt.figure(figsize=(12, 2))  # wide & short for horizontal layout
    ax = fig_legend.add_subplot(111)
    ax.axis("off")

    ax.legend(
        legend_handles,
        legends,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        prop={"weight": "bold", "size": fontsize},
        ncol=len(legends),  # ✅ all items on one line
        loc="center",  # ✅ centered in the figure
        bbox_to_anchor=(0.5, 0.5)
    )

    figpath_legend = f"{saveFileName}_legend.pdf"
    fig_legend.savefig(figpath_legend, format="pdf", bbox_inches="tight")
    plt.close(fig_legend)

    fig_zoom = plt.figure(figsize=(18, 12))
    ax_zoom = fig_zoom.add_subplot(111)

    for i in range(len(legends)-1):
        color = "red" if legends[i] == "Optimal" else colors[i]
        marker = "" if legends[i] ==  "Optimal" else markers[i]
        if legends[i] in METHODS:
            color = COLORS_METHODS[legends[i]]
            marker = MARKERS_METHODS[legends[i]]
        #n = y_data[i].shape[1]


        x_vals = x_data[-2:]  # 5 derniers x
        y_vals = (y_data[i])[-2:]  # 5 derniers y

        ax_zoom.plot(
        x_vals,
        y_vals,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=2*markersize,
        color=color,
        label=f"{legends[i]}",
        markeredgecolor="black",
        )
        last_x, last_y = x_vals[-1], y_vals[-1]
        ax_zoom.text(
            last_x, last_y,
            f"{last_y:.3e}",
            fontweight="bold",
            fontsize=2*markersize,
            bbox=dict(facecolor="white", alpha=0.7),
            verticalalignment="bottom",
            horizontalalignment="right"
        )

    # même axes que principal (pas de zoom)
    ax_zoom.set_xlim(x_data[-2], x_data[-1])
    ax_zoom.set_ylim(plt.ylim())  # reprendre les bornes du plot principal



    for label in ax_zoom.get_xticklabels() + ax_zoom.get_yticklabels():
        label.set_fontweight("bold")
    ax_zoom.tick_params(axis="both", labelsize=2*markersize)
    ax_zoom.set_ylabel("", fontweight="bold")
    ax_zoom.yaxis.label.set_size(1.5*markersize)
    ax_zoom.set_xlabel(f"", fontweight="bold")

    ax_zoom.set_xticks([])  # Supprime les graduations
    ax_zoom.set_xlabel("")  # Supprime le label
    ax_zoom.spines["bottom"].set_visible(False)  # Cache la ligne de l’axe

    ax_zoom.spines["top"].set_visible(False)

    ax_zoom.grid(True)
    fig_zoom.tight_layout()

    figpath_zoom = f"{saveFileName}_zoom.pdf"
    fig_zoom.savefig(figpath_zoom, format="pdf")
    plt.close(fig_zoom)

    return figpath_plot, figpath_legend, figpath_zoom
def plotGame_dim_N(
    x_data, y_data, x_label, y_label, legends, saveFileName,
    ylog_scale, Players2See=[1,2], fontsize=40, markersize=40, linewidth=12, linestyle="-",
    pltText=False, step=1
):
    plt.figure(figsize=(18, 12))
    y_data = np.array(y_data)

    plt.rcParams.update({'font.size': fontsize})

    linestyles = ["-", "--", ":", "--"]

    if ylog_scale:
        plt.yscale("log")

    # --- Plot curves ---
    legend_handles = []
    legends2 = []
    for i in range(len(legends)):
        color = (
            "red" if legends[i] == "Optimal" else COLORS_METHODS[legends[i]] if legends[i] in METHODS else
            colors[
                i])
        #n = y_data[i].shape[1]

        for j in Players2See:
            legends2.append(f"{legends[i]} -- Player {j+1}" if legends[i] in METHODS else legends[i])
            # --- Build legend handles ---
            legend_handles.append(
                Line2D([0], [0], color=color
                       , marker=markers[j], markersize=markersize, markeredgecolor="black", linestyle=linestyle,
                       linewidth=linewidth)
        )
            #print(y_data)
            plt.plot(
                x_data[::step],
                (y_data[i])[:, j][::step],
                linestyle=linestyle,
                linewidth=linewidth,
                marker=markers[j],
                markersize=2*markersize,
                color=color,
                markeredgecolor="black",
            )

            if pltText:
                last_x = len((y_data[i])[:, j]) - 1
                last_y = (y_data[i])[:, j][-1]
                plt.text(
                    last_x, last_y,
                    f"{last_y:.2e}",
                    fontweight="bold",
                    fontsize=fontsize,
                    bbox=dict(facecolor='white', alpha=0.7),
                    verticalalignment='bottom', horizontalalignment='right'
                )

    # --- Axis formatting ---
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight("bold")

    plt.ylabel(f"{y_label}", fontweight="bold")
    plt.xlabel(f"{x_label}", fontweight="bold")
    plt.grid(True)
    plt.tight_layout()

    # --- Save plot without legend ---
    figpath_plot = f"{saveFileName}_plot.pdf"
    plt.savefig(figpath_plot, format="pdf")



    # --- Save legend separately ---
    rows = len(Players2See)  # ✅ always 2 rows
    n_items = len(legends2)
    ncol = int(np.ceil(n_items / rows))  # répartir équitablement
    fig_legend = plt.figure(figsize=(12, 2))  # wide & short for horizontal layout
    ax = fig_legend.add_subplot(111)
    ax.axis("off")

    ax.legend(
        legend_handles,
        legends2,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        prop={"weight": "bold", "size": fontsize},
        ncol=ncol,#len(legends2),  # ✅ all items on one line
        loc="center",  # ✅ centered in the figure
        bbox_to_anchor=(0.5, 0.5)
    )

    figpath_legend = f"{saveFileName}_legend.pdf"
    fig_legend.savefig(figpath_legend, format="pdf", bbox_inches="tight")
    plt.close(fig_legend)


    fig_zoom = plt.figure(figsize=(18, 12))
    ax_zoom = fig_zoom.add_subplot(111)

    for i in range(len(legends)):
        color = (
            "red" if legends[i] == "Optimal"
            else COLORS_METHODS[legends[i]] if legends[i] in METHODS
            else colors[i]
        )
        n = y_data[i].shape[1]

        for j in Players2See:
            x_vals = x_data[-5:]  # 5 derniers x
            y_vals = (y_data[i])[-5:, j]  # 5 derniers y

            ax_zoom.plot(
                x_vals,
                y_vals,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=markers[j],
                markersize=2*markersize,
                color=color,
                markeredgecolor="black",
            )
            last_x, last_y = x_vals[-1], y_vals[-1]
            ax_zoom.text(
                last_x, last_y,
                f"{last_y:.3e}",
                fontweight="bold",
                fontsize=80,
                bbox=dict(facecolor="white", alpha=0.7),
                verticalalignment="bottom",
                horizontalalignment="right"
            )

    # même axes que principal (pas de zoom)
    ax_zoom.set_xlim(x_data[-5], x_data[-1])
    ax_zoom.set_ylim(plt.ylim())  # reprendre les bornes du plot principal



    for label in ax_zoom.get_xticklabels() + ax_zoom.get_yticklabels():
        label.set_fontweight("bold")
    ax_zoom.tick_params(axis="both", labelsize=2*markersize)
    ax_zoom.set_ylabel("",fontweight="bold")
    ax_zoom.yaxis.label.set_size(2*markersize)
    ax_zoom.set_xlabel(f"", fontweight="bold")

    ax_zoom.set_xticks([])  # Supprime les graduations
    ax_zoom.set_xlabel("")  # Supprime le label
    ax_zoom.spines["bottom"].set_visible(False)  # Cache la ligne de l’axe

    ax_zoom.spines["top"].set_visible(False)

    ax_zoom.grid(True)
    fig_zoom.tight_layout()

    figpath_zoom = f"{saveFileName}_zoom.pdf"
    fig_zoom.savefig(figpath_zoom, format="pdf")
    plt.close(fig_zoom)

    return figpath_plot, figpath_legend, figpath_zoom


def plotGame_dim_N_last(
        x_data, y_data, x_label, y_label, legends, saveFileName, funcs_=["SBRD","DAE"],
        ylog_scale=False, Players2See=[1, 2], fontsize=40,
        markersize=40, linewidth=12, linestyle="-",
        pltText=False, step=1, tol=1e-3
):
    plt.figure(figsize=(18, 12))
    #y_data = np.array(y_data, dtype=object)  # s'assurer que les sous-tableaux passent bien
    y_data_Hybrid = y_data[0]
    plt.rcParams.update({'font.size': fontsize})

    if ylog_scale:
        plt.yscale("log")

    legend_handles = []
    curves = []
    funcNo_NE = [i for i in funcs_ if i!="NE"]

    for j, fc in enumerate(funcs_):
        curve = []


        for i in range(len(y_data_Hybrid)):
            #print(f"y_data_Hybrid{y_data_Hybrid[i]}")
            curve.append((y_data_Hybrid[i][j]))
        curves.append(
            curve
        )


    for j, fc in enumerate(funcs_):

        color = "red" if fc == "NE" else colors[j]
        marker = "" if fc == "NE" else markers[j % len(markers)]
        if fc in METHODS:
            color = COLORS_METHODS[fc]
            marker = MARKERS_METHODS[fc]

        legend_handles.append(
            Line2D([0], [0], color=color,
                   marker=marker,
                   markersize=markersize,
                   markeredgecolor="black",
                   linestyle=linestyle,
                   linewidth=linewidth)
        )

        if fc == "NE":
            # tracer une ligne horizontale rouge à la valeur k
            print(y_data[-1][-1][0])

        else:
            # tracer l’évolution (jusqu’à la dernière valeur)

            curve = curves[j]
            print(f"{fc}", curve, x_data)
            plt.plot(
                x_data,
                curve,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=1.25 * markersize,
                color=color,
                label=f"{funcs_[j]}",
                markeredgecolor="black",
            )

        if pltText:
            plt.text(x_data[-1], curve[-1], f"{curve[-1]:.3f}",
                fontweight="bold",
                fontsize=fontsize,
                bbox=dict(facecolor="white", alpha=0.7),
                verticalalignment="bottom",
                horizontalalignment="right")

    # légendes et labels
    # légendes et labels

    if funcs_[-1]== "NE":
        curve = [y_data[-1][-1][0]]
        plt.axhline(
            y=curve,
            color="red",
            linestyle=linestyle,
            linewidth=linewidth,
            label=f"NE"
        )

        if pltText:
            plt.text(x_data[-1], curve[-1], f"{curve[-1]:.3f}",
                     fontweight="bold",
                     fontsize=fontsize,
                     bbox=dict(facecolor="white", alpha=0.7),
                     verticalalignment="bottom",
                     horizontalalignment="right")

    ax = plt.gca()

    # --- Axis formatting ---
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

   # ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # ✅ ticks entiers
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    plt.ylabel(str(f"{y_label}"), fontweight="bold", fontsize=2*fontsize)
    plt.xlabel(str(f"{x_label}"), fontweight="bold", fontsize=2*fontsize)
    plt.grid(True)

    # 🔑 Horizontal legend
    plt.legend(
       # loc="lower left",
        #bbox_to_anchor=(0.5, -0.15),  # ✅ sous la figure
        #ncol=len(funcs_),  # ✅ labels sur une seule ligne
        frameon=False,
        prop={'weight': 'bold'}
    )
    plt.tight_layout()



    # --- Save plot without legend ---
    figpath_plot = f"{saveFileName}_plot.pdf"
    plt.savefig(figpath_plot, format="pdf")




    return figpath_plot, figpath_plot, figpath_plot


def plotGame2(config,
    x_data, y_data, x_label, y_label, legends, saveFileName,
    ylog_scale, fontsize=40, markersize=40, linewidth=12,
    linestyle="-", pltText=False, step=1
):
    plt.figure(figsize=(18, 12))
    y_data = np.array(y_data)

    plt.rcParams.update({'font.size': fontsize})
    x_data_copy = x_data.copy()

    if ylog_scale:
        plt.yscale("log")

    # --- Plot curves ---

    for i, legend in enumerate(legends):
        color = "red" if legends[i] == "Optimal" else colors[i]
        marker = "" if legends[i] ==  "Optimal" else markers[i]

        try:
            if config["lrMethods"][i] in METHODS:

                if config["lrMethods"].count(config["lrMethods"][i])>1 and config["Learning_rates"][i]==0.05:
                    color = COLORS_METHODS[config["lrMethods"][i]]
                    marker = MARKERS_METHODS[config["lrMethods"][i]]
                elif config["lrMethods"].count(config["lrMethods"][i])==1:

                    color = COLORS_METHODS[config["lrMethods"][i]]
                    marker = MARKERS_METHODS[config["lrMethods"][i]]

            if config["Hybrid_funcs_"][i][1] in METHODS and config["Learning_rates"][i]==0.05:
                color = COLORS_METHODS[config["Hybrid_funcs_"][i][1]]
                marker = MARKERS_METHODS[config["Hybrid_funcs_"][i][1]]


        except:
            print()


        plt.plot(
            x_data[::step],
            (y_data[i])[::step],
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=1 * markersize,
            color=color,
            label=f"{legend}",
            markeredgecolor="black",
        )

        if pltText:
            last_x = len(y_data[i]) - 1
            #if config["lrMethods"][i]=="OGD":
            #    last_y =8.32e-2
            #else:
            last_y = y_data[i][-1]
            plt.text(
                last_x, last_y,
                f"{last_y:.2e}",
                fontweight="bold",
                fontsize=fontsize,
                bbox=dict(facecolor='white', alpha=0.7),
                verticalalignment='bottom', horizontalalignment='right'
            )

    # --- Axis formatting ---
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight("bold")

    plt.ylabel(str(f"{y_label}"), fontweight="bold", fontsize=2*fontsize)
    plt.xlabel(str(f"{x_label}"), fontweight="bold", fontsize=2*fontsize)
    plt.legend(frameon=False,prop={'weight': 'bold'})
    plt.grid(True)
    plt.tight_layout()

    # --- Save plot without legend ---
    figpath_plot = f"{saveFileName}_plot.pdf"
    plt.savefig(figpath_plot, format="pdf")

    # --- Build legend handles ---
    legend_handles = [
        Line2D([0], [0], color=("red" if legends[k] == "Optimal" else  COLORS_METHODS[legends[k]] if legends[k] in METHODS else colors[k]),
               markeredgecolor="black", linestyle=linestyle if linestyle != "" else "-",
               marker=("" if legends[k] == "Optimal" else MARKERS_METHODS[legends[k]] if legends[k] in METHODS else markers[k] ),
               markersize=markersize, linewidth=linewidth)
        for k in range(len(legends))
    ]

    # --- Save legend separately ---
    fig_legend = plt.figure(figsize=(12, 2))  # wide & short for horizontal layout
    ax = fig_legend.add_subplot(111)
    ax.axis("off")

    ax.legend(
        legend_handles,
        legends,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        prop={"weight": "bold", "size": fontsize},
        ncol=len(legends),  # ✅ all items on one line
        loc="center",  # ✅ centered in the figure
        bbox_to_anchor=(0.5, 0.5)
    )

    figpath_legend = f"{saveFileName}_legend.pdf"
    fig_legend.savefig(figpath_legend, format="pdf", bbox_inches="tight")
    plt.close(fig_legend)



    return figpath_plot, figpath_legend, figpath_plot