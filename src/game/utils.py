#utils.py

import torch, random
from scipy import optimize


from scipy.optimize import root_scalar
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import numpy as np
import matplotlib.pyplot as plt

colors = [
    "slategray",  # Gris ardoise
    "brown",  # Marron
    #"magenta",  # Magenta
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
METHODS = [
    rf"DAQ$_F$", rf"DAQ$_V$",
    "DAE",
    rf"OGD$_F$", rf"OGD$_V$",
    "SBRD",
    rf"RRM$_F$", rf"RRM$_V$"
]

legend_map = {
    "RRM_V": rf"RRM$_V$",
    "RRM_F": rf"RRM$_F$",
    "DAQ_V": rf"DAQ$_V$",
    "DAQ_F": rf"DAQ$_F$",
    "OGD_V": rf"OGD$_V$",
    "OGD_F": rf"OGD$_F$",
}

COLORS_METHODS = {
    rf"DAQ$_F$": "darkorange",   # dark orange
    rf"DAQ$_V$": "brown",   # lighter orange (same hue family)

    "DAE": "#3182bd",         # royal blue

    rf"OGD$_F$": "green",   # medium green
    rf"OGD$_V$": "teal",   # lighter green

    "SBRD": "purple",        # violet

    rf"RRM$_F$": "slategray",   # dark gray
    rf"RRM$_V$": "magenta"#"royalblue",   # lighter gray, same tone family
}

MARKERS_METHODS = {
    rf"DAQ$_F$": "s",             # square
    rf"DAQ$_V$": "D",             # pentagon (close shape)
    "DAE": "x",                   # triangle up
    rf"OGD$_F$": "v",             # triangle down
    rf"OGD$_V$": "^",             # triangle marker variant
    "SBRD": "*",                  # diamond
    rf"RRM$_F$": "H",             # hexagon
    rf"RRM$_V$": "p",             # star
}



markers = [ "d","p", "s", "^", "v", "D", "*", "p", "x", "+", "|","s", "^", "v", "D", "*", "p", "x", "+", "|"]
markers22 = ["H", "d","*","p"]
colors22 = [
    "slategray",  # Gris ardoise
    "brown",  # Marron
    "magenta",  # Magenta
    "teal",  # Bleu-vert
]
def make_subset(n, h):
    """
    Retourne un couple [subset, remaining] basé sur n et h.
    subset contient toujours 0 + (h-1) autres éléments tirés au hasard.
    remaining contient les éléments restants.

    Parameters
    ----------
    n : int
        Taille totale (cfg["n"])
    h : int
        Taille du sous-ensemble incluant 0

    Returns
    -------
    subset : list
        Sous-ensemble contenant 0
    remaining : list
        Eléments restants
    """
    if n <= 0:
        raise ValueError("n doit être > 0")
    if h < 1:
        raise ValueError("h doit être >= 1 car subset contient toujours 0")

    # candidats possibles pour compléter subset (exclure 0)
    candidates = list(set(range(n)) - {0, 1})

    # éviter erreur random.sample si h est trop grand
    h_effective = max(0, min(h - 1, len(candidates)))

    # tirer h-1 éléments aléatoires parmi les candidats
    others = random.sample(candidates, h_effective)

    # Construire subset
    subset = [0] + others

    # éléments restants
    remaining = [i for i in range(n) if i not in subset]

    return [subset, remaining]
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
def log_potential(
    z: torch.Tensor,
    a: torch.Tensor,
    price: float | int | torch.Tensor,
    mode: str = "auto",
) -> torch.Tensor:
    """
    Compute the log-case potential for the bidding game with shares x_i = z_i / sum_j z_j.

    Modes:
      - "exact":    uses Φ_ex = a * sum log z - a * log(sum z) - price * sum z
                    (valid when all a_i are equal to the same a > 0)
      - "weighted": uses Φ_w  = sum log z - log(sum z) - price * sum (z_i / a_i)
                    (weighted potential for heterogeneous a_i > 0)
      - "auto":     if all a_i are (numerically) equal -> exact; else -> weighted

    Args:
        z   : (n,) tensor, bids (must be > 0 for log)
        a   : (n,) tensor of positive a_i
        price: scalar (float/int/tensor)
        mode: "auto" | "exact" | "weighted"
        eps : small clamp to keep logs stable

    Returns:
        Scalar tensor (0-dim): the potential value.
    """
    if z.ndim != 1 or a.ndim != 1:
        raise ValueError("z and a must be 1-D tensors")
    if z.shape != a.shape:
        raise ValueError("z and a must have the same shape")
    if (a <= 0).any():
        raise ValueError("All entries of 'a' must be > 0")

    #z = torch.clamp_min(z, eps)
    S = z.sum()#torch.clamp_min(, eps)
    price = torch.as_tensor(price, dtype=z.dtype, device=z.device)

    # Decide mode
    if mode == "auto":
        a0 = a[0]
        if torch.allclose(a, a0.expand_as(a), rtol=1e-6, atol=1e-8):
            mode_eff = "exact"
        else:
            mode_eff = "weighted"
    else:
        mode_eff = mode.lower()
        if mode_eff not in {"exact", "weighted"}:
            raise ValueError("mode must be 'auto', 'exact', or 'weighted'")

    if mode_eff == "exact":
        a0 = a[0]
        # (Optionally enforce equality)
        if not torch.allclose(a, a0.expand_as(a), rtol=1e-6, atol=1e-8):
            raise ValueError("Exact potential requires all a_i equal; use mode='weighted' or 'auto'.")
        potential = a0 * torch.log(z).sum() - a0 * torch.log(S) - price * z.sum()
    else:  # weighted
        potential = torch.log(z).sum() - torch.log(S) - price * (z / a).sum()

    return potential

def SW_func(x, budgets, a_vector, d_vector, alpha):
    V = a_vector * V_func(x, alpha) + d_vector
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
    smin = (n - 1) * epsilon + delta
    smax = (n - 1) * c + delta
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
        inner = abs(a_i * s / (z * (z + s)) - 1.0)
        return inner#z * (inner ** 2)

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
    G = G_max#np.sqrt(G_max)
    return G
def max_diag_grad_norm(a_vec, c, eps, delta):
    """
    Compute G = max_{z in [1,c]^n} || (∂φ_i/∂z_i)_i ||_2
    for φ_i(z) = a_i log(z_i / (sum(z)+δ)) + d_i - z_i.

    Parameters
    ----------
    a_vec : array-like, shape (n,)
        Positive coefficients a_i.
    c : float
        Upper bound of the box [1, c], with c >= 1.
    delta : float
        Positive constant δ.

    Returns
    -------
    G : float
        Exact maximum of the 2-norm over the box [1,c]^n.
    arg_info : dict
        Information about the maximizing corner:
        - 'K': number of variables set to c
        - 's': s_K = δ + K*c + (n-K)*1
        - 'z_pattern': array of {1,c} giving one maximizing assignment
    """
    a = a_vec# np.asarray(a_vec, dtype=float)
    n = a.shape[-1]

    best_val = -np.inf
    best_info = None
    s_min = (n-1) * eps +delta
    s_max = (n-1) * c + delta
    f_min = torch.max(torch.sqrt(torch.sum(((a_vec *(1/eps) - 1 / s_min) - 1 )**2)), torch.sqrt(torch.sum(((a_vec *(1/c) - 1 / s_min) - 1 )**2)))
    f_max = torch.max(torch.sqrt(torch.sum(((a_vec * (1 / eps) - 1 / s_max) - 1) ** 2)),
                      torch.sqrt(torch.sum(((a_vec * (1 / c) - 1 / s_max) - 1) ** 2)))
    G =torch.max(f_min, f_max)
    return G

class GameKelly:
    def __init__(self, n: int, price: float,
                 epsilon, delta, alpha, tol, payoff_min=None,payoff_max=None):


        self.n = n


        self.price = price
        self.epsilon = epsilon
        self.delta = delta
        self.alpha = alpha
        self.tol = tol
        self.payoff_min = payoff_min
        self.payoff_max = payoff_max

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
            z_br = solve_nonlinear_eq(a_vector, p, self.alpha, self.epsilon, c_vector, self.price, max_iter=1000, tol=self.tol)
            err = torch.norm(z_br - z)#,float('inf'))#/(z.shape[-1] **(0.5)*torch.max(c_vector - self.epsilon))#, self.tol * torch.ones(1))
        else:
            z_br = BR_alpha_fair(self.epsilon, c_vector, z, p,
                                            a_vector, self.delta, self.alpha, self.price,
                                            b=0)
            err =  torch.norm(z_br - z)#,float('inf'))#/(z.shape[-1] **(0.5)*torch.max(c_vector - self.epsilon))#, self.tol * torch.ones(1))

        return err   # torch.norm(self.grad_phi(z))

    def epsilon_error(self, z: torch.tensor, a_vector, c_vector, d_vector):
        p = torch.sum(z) - z + self.delta
        if self.alpha  not in [0,1,2]:
            z_br = solve_nonlinear_eq(a_vector, p, self.alpha, self.epsilon, c_vector, self.price, max_iter=1000, tol=self.tol)
            payoff_zbr = Payoff(self.fraction_resource(z_br), z_br, a_vector, d_vector, self.alpha, self.price)
            payoff_z = Payoff(self.fraction_resource(z), z, a_vector, d_vector, self.alpha, self.price)
            err = payoff_zbr - payoff_z#torch.norm(z_br - z)#/(z.shape[-1] **(0.5)*torch.max(c_vector - self.epsilon))#, self.tol * torch.ones(1))
        else:
            z_br = BR_alpha_fair(self.epsilon, c_vector, z, p,
                                            a_vector, self.delta, self.alpha, self.price,
                                            b=0)
            payoff_zbr = (Payoff(self.fraction_resource(z_br), z_br, a_vector, d_vector, self.alpha, self.price) - self.payoff_min) / (self.payoff_max - self.payoff_min)
            payoff_z = (Payoff(self.fraction_resource(z), z, a_vector, d_vector, self.alpha, self.price) - self.payoff_min) / (self.payoff_max - self.payoff_min)
            err = payoff_zbr - payoff_z

        return torch.max(torch.max(torch.abs(err)), torch.tensor(self.tol))   #

    def epsilon_error_Hybid(self, z: torch.tensor, a_vector, c_vector, d_vector,Hybrid_sets):
        p = torch.sum(z) - z + self.delta
        if self.alpha  not in [0,1,2]:
            z_br = solve_nonlinear_eq(a_vector, p, self.alpha, self.epsilon, c_vector, self.price, max_iter=1000, tol=self.tol)
            payoff_zbr = Payoff(self.fraction_resource(z_br), z_br, a_vector, d_vector, self.alpha, self.price)
            payoff_z = Payoff(self.fraction_resource(z), z, a_vector, d_vector, self.alpha, self.price)
            err = payoff_zbr - payoff_z#torch.norm(z_br - z)#/(z.shape[-1] **(0.5)*torch.max(c_vector - self.epsilon))#, self.tol * torch.ones(1))
        else:
            z_br = BR_alpha_fair(self.epsilon, c_vector, z, p,
                                            a_vector, self.delta, self.alpha, self.price,
                                            b=0)
            payoff_zbr = (Payoff(self.fraction_resource(z_br), z_br, a_vector, d_vector, self.alpha, self.price) - self.payoff_min) / (self.payoff_max - self.payoff_min)
            payoff_z = (Payoff(self.fraction_resource(z), z, a_vector, d_vector, self.alpha, self.price) - self.payoff_min) / (self.payoff_max - self.payoff_min)
            err = payoff_zbr - payoff_z

        eps_err = torch.zeros(len(Hybrid_sets))
        for i in range(len(Hybrid_sets)):
            eps_err[i] =  torch.max(torch.max(torch.abs(err[Hybrid_sets[i]])), torch.tensor(self.tol))

        #print(payoff_z)
        return eps_err#torch.max(torch.max(err[Hybrid_sets[0]]), torch.tensor(self.tol)), torch.max(torch.max(err[Hybrid_sets[1]]), torch.tensor(self.tol))   # torch.norm(self.grad_phi(z))

    def AverageBid(self, z,t):
        z_copy = z.clone()
        z_t = 1/t*torch.sum(z,dim=0)
        return z_t
    def Jain_index(self,z):
        n = z.shape[0]                       # players on last axis
        s  = torch.sum(z)
        s2 = torch.sum(z**2)

        return s**2 /(n * s2)
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



    def OGD_V(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad,p=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):

        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector

        grad_t = self.grad_phi(phi, bids)
        D = c_vector - self.epsilon ##np.linalg.norm(c_vector - self.epsilon)  # si c est borne sup
        # Constante Lipschitz
        n = len(bids)
        G = torch.zeros_like(a_vector)
        for i in range(n):
            G[i] = compute_G(a_vector[i], self.epsilon,c_vector[i], self.epsilon, n)

        eta_t = D / (2 * G * np.sqrt(t))
        z_candidate = bids + eta_t * grad_t
        z_t = Q1(z_candidate, self.epsilon, c_vector, self.price)
        return z_t, acc_grad

    def OGD_F(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad,
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
        G = torch.zeros_like(a_vector)
        for i in range(n):
            G[i] = compute_G(a_vector[i], self.epsilon,c_vector[i], self.epsilon, n)

        # Step-size (theorem 3.1 Hazan)
        #if vary:
            # Step-size (theorem 3.1 Hazan)
        #    eta_t = 100/t**eta if t > 0 else eta
        #else:
        eta_t =  D / (G * np.sqrt(self.T))  # fallback pour t=0

        # Update rule
        z_candidate = bids + eta_t * grad_t
        z_t = Q1(z_candidate, self.epsilon, c_vector, self.price)

        return z_t, acc_grad
    def RRM_F(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, p=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):

        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector

        acc_grad_copy = acc_grad.clone()
        grad_t = self.grad_phi(phi, bids)
        n = len(bids)

        D = c_vector -  self.epsilon #torch.sqrt(c_vector **2 -  self.epsilon **2) # si c est borne sup
        G = torch.zeros_like(a_vector)
        for i in range(n):
            G[i] = compute_G(a_vector[i], self.epsilon, c_vector[i], self.epsilon, n)

        #G = max_diag_grad_norm(a_vector, c_vector,self.epsilon, self.delta)
        eta_t =  D / (G * np.sqrt(self.T))
        acc_grad_copy += grad_t * eta_t
        z_t = Q1(acc_grad_copy, self.epsilon, c_vector, self.price)
        return z_t, acc_grad_copy
    def RRM_V(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, p=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):

        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector

        acc_grad_copy = acc_grad.clone()
        grad_t = self.grad_phi(phi, bids)
        n = len(bids)

        D = c_vector -  self.epsilon # si c est borne sup
        G = torch.zeros_like(a_vector)
        for i in range(n):
            G[i] = compute_G(a_vector[i], self.epsilon, c_vector[i], self.epsilon, n)

        eta_t = D / (2 * G * np.sqrt(t))
        acc_grad_copy += grad_t * eta_t
        z_t = Q1(acc_grad_copy, self.epsilon, c_vector, self.price)
        return z_t, acc_grad_copy
    def DAQ_V(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, p=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):

        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector

        acc_grad_copy = acc_grad.clone()
        grad_t = self.grad_phi(phi, bids)
        n = len(bids)

        D = c_vector -  self.epsilon #torch.sqrt(c_vector **2 -  self.epsilon **2) # si c est borne sup

        G = torch.zeros_like(a_vector)
        for i in range(n):
            G[i] = compute_G(a_vector[i], self.epsilon, c_vector[i], self.epsilon, n)
        eta_t = D / (2*G * np.sqrt(t))

        acc_grad_copy += grad_t
        z_t = Q1(acc_grad_copy * eta_t, self.epsilon, c_vector, self.price)
        return z_t, acc_grad_copy

    def DAQ_F(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, p=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):

        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector

        acc_grad_copy = acc_grad.clone()
        grad_t = self.grad_phi(phi, bids)
        n = len(bids)

        D = c_vector -  self.epsilon#torch.sqrt(c_vector **2 -  self.epsilon **2) # si c est borne sup
        G = torch.zeros_like(a_vector)
        for i in range(n):
            G[i] = compute_G(a_vector[i], self.epsilon, c_vector[i], self.epsilon, n)

        # Step-size (theorem 3.1 Hazan)

        eta_t =  D / (G * np.sqrt(self.T))  #self.T fallback pour t=0

        acc_grad_copy += grad_t
        z_t = Q1(acc_grad_copy * eta_t, self.epsilon, c_vector, self.price)
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
            #G[i] = compute_G_DAE(a_vector[i], self.epsilon, c_vector[i], self.epsilon, n)
            G[i] = compute_G(a_vector[i], self.epsilon, c_vector[i], self.epsilon, n)


        # Step-size (theorem 3.1 Hazan)
        if vary:
            # Step-size (theorem 3.1 Hazan)
            eta_t = D / (G * np.sqrt(3000))  # fallback pour t=0
            #eta_t = 100/t**eta if t > 0 else eta
        else:
            eta_t =  D / (G * np.sqrt(self.T))  # fallback pour t=0
        acc_grad_copy += grad_t * eta_t
        z_t = Q2(acc_grad_copy, self.epsilon, c_vector, self.price)

        return z_t, acc_grad_copy

    def SBRD(self, t, a_vector, c_vector, d_vector,  eta, bids, acc_grad, p=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        #p = torch.sum(bids) - bids + self.delta
        z_t = BR_alpha_fair(self.epsilon, c_vector, bids, p,
                                         a_vector, self.delta, self.alpha, self.price, b=0)

        return z_t, acc_grad

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

        self.T = n_iter

        acc_grad = torch.zeros(self.n, dtype=torch.float64)
        matrix_bids = torch.zeros((n_iter + 1, self.n), dtype=torch.float64)
        vec_LSW = torch.zeros(n_iter + 1, dtype=torch.float64)
        vec_SW = torch.zeros(n_iter + 1, dtype=torch.float64)
        potential = vec_SW.clone()
        jain_idx = vec_SW.clone()
        eps_error = vec_SW.clone()
        utiliy = torch.zeros((n_iter + 1, self.n), dtype=torch.float64)
        utiliy_residual = utiliy.clone()
        valuation = utiliy.clone()
        error_NE = torch.zeros(n_iter + 1, dtype=torch.float64)
        error_NE_Hybrid = torch.zeros(( n_iter + 1,2))
        if Hybrid_sets!= None:
            error_NE_Hybrid = torch.zeros((n_iter + 1,len(Hybrid_sets)), dtype=torch.float64) #epsilon_error_Hybid
        matrix_bids[0] = bids.clone()
        jain_idx[0] = self.Jain_index(bids)

        error_NE[0] = self.check_NE(bids, a_vector, c_vector, d_vector)
        eps_error[0] = self.epsilon_error(bids, a_vector, c_vector, d_vector)
        utiliy[0] = Payoff(self.fraction_resource(matrix_bids[0]), matrix_bids[0], a_vector, d_vector, self.alpha, self.price)
        valuation[0] = Valuation(self.fraction_resource(matrix_bids[0]), a_vector, d_vector, self.alpha)
        vec_LSW[0] = LSW_func(self.fraction_resource(matrix_bids[0]), c_vector, a_vector, d_vector, self.alpha)
        vec_SW[0] = SW_func(self.fraction_resource(matrix_bids[0]), c_vector, a_vector, d_vector, self.alpha)
        z_br = BR_alpha_fair(self.epsilon, c_vector, matrix_bids[0], torch.sum(matrix_bids[0]) - matrix_bids[0] + self.delta, a_vector, self.delta, self.alpha, self.price,
                             b=0)
        utiliy_residual[0] = torch.abs(utiliy[0] - Payoff(self.fraction_resource(z_br), z_br, a_vector, d_vector, self.alpha, self.price))
        potential[0] =  log_potential(matrix_bids[0],a_vector,self.price)
        if Hybrid_sets != None:
            error_NE_Hybrid[0] = self.epsilon_error_Hybid(bids, a_vector, c_vector, d_vector, Hybrid_sets)


        k = 0


        for t in range(1, n_iter + 1):

            k = t
            p = torch.sum(matrix_bids[t-1]) - matrix_bids[t-1] + self.delta
            matrix_bids[t], acc_grad = func(t, a_vector, c_vector, d_vector, eta, matrix_bids[t-1], acc_grad, p=p, vary=vary, Hybrid_funcs=Hybrid_funcs, Hybrid_sets=Hybrid_sets)
            jain_idx[t] = self.Jain_index(matrix_bids[t])
            error_NE[t] = self.check_NE(matrix_bids[t], a_vector, c_vector, d_vector)


            eps_error[t] = self.epsilon_error(matrix_bids[t], a_vector, c_vector, d_vector)
            if Hybrid_sets != None:
                error_NE_Hybrid[t] = self.epsilon_error_Hybid(matrix_bids[t], a_vector, c_vector, d_vector, Hybrid_sets)
            vec_LSW[t] = LSW_func(self.fraction_resource(matrix_bids[t]), c_vector, a_vector, d_vector, self.alpha)

            z_br = BR_alpha_fair(self.epsilon, c_vector, matrix_bids[t], p, a_vector, self.delta, self.alpha, self.price, b=0)

            vec_SW[t] = SW_func(self.fraction_resource(matrix_bids[t]), c_vector, a_vector, d_vector, self.alpha)
            utiliy[t] = Payoff(self.fraction_resource(matrix_bids[t]), matrix_bids[t], a_vector, d_vector, self.alpha, self.price)
            utiliy_residual[t] = (utiliy[t] - Payoff(self.fraction_resource(z_br), z_br, a_vector, d_vector, self.alpha, self.price))
            valuation[t] = Valuation(self.fraction_resource(matrix_bids[t]), a_vector, d_vector, self.alpha)
            potential[t] =  log_potential(matrix_bids[t],a_vector,self.price)
            err = torch.min(error_NE[:k])#round(float(torch.min(error_NE[:k])),3)
            #agg_bids[t] = 1/(t+1) * torch.sum(matrix_bids[:t], dim=0)#self.AverageBid(matrix_bids, t)

            if stop and error_NE[t] <= self.tol:
                break
        col = torch.arange(1, k + 1)
        agg_bids = torch.cumsum(matrix_bids[:k, :], dim=0)/ col.unsqueeze(1).expand(-1,self.n)
        Bids = [matrix_bids[:k, :], agg_bids[:k, :], jain_idx[:k]]
        sw = torch.sum(utiliy[:k, :], dim=1); lsw = vec_LSW[:k]
        agg_utility = torch.cumsum(utiliy[:k, :], dim=0)

        agg_utility = agg_utility / col.unsqueeze(1).expand(-1, self.n)

        Utility_set = [utiliy[:k, :], agg_utility[:k, :], utiliy_residual[:k, :], potential[:k], eps_error[:k], error_NE_Hybrid[:k,:]]

        Welfare = [vec_SW[:k], vec_LSW[:k], valuation[:k]]
        #ERRORs = [error_NE_Hybrid]
        return Bids, Welfare, Utility_set, torch.maximum(error_NE[:k], torch.tensor(self.tol))  #matrix_bids[:k, :], vec_LSW[:k], error_NE[:k], Avg_bids[:k, :], utiliy[:k, :]

def plotGame(config,
    x_data, y_data, x_label, y_label, legends, saveFileName,
    ylog_scale, fontsize=40, markersize=40, linewidth=12,
    linestyle="-", pltText=False, step=1, tol=1e-6
):
    plt.figure(figsize=(18, 12))
    y_data = np.array(y_data)
    print(y_data[-1])

    plt.rcParams.update({'font.size': fontsize})
    x_data_copy = x_data.copy()

    if ylog_scale:
        plt.yscale("log")

    # --- Plot curves on main axis ---
    legends = [legend_map.get(l, l) for l in legends]
    for i, legend in enumerate(legends):

        color = "red" if legends[i] == "NE" else colors[i]
        marker = "" if legends[i] == "NE" else markers[i]

        if legends[i] in METHODS:
            color = COLORS_METHODS[legends[i]]
            marker = MARKERS_METHODS[legends[i]]

        y_data_i = y_data[i][:config["T_plot"]]
        x_data_i = x_data[:config["T_plot"]]
        label = ""
        if pltText:
            label = f"{legend}"

        plt.plot(
            x_data_i[::step],
            y_data_i[::step],
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=1 * markersize,
            color=color,
            label=label,
            markeredgecolor="black",
        )

        if pltText:
            last_x = x_data_i[-1]
            last_y = y_data_i[-1]
            lastValue = f"{last_y:.2e}"
            if config.get("metric", "") == "Relative_Efficienty_Loss":
                lastValue = f"{last_y:.3f}%"
            plt.text(
                last_x, last_y,
                lastValue,
                fontweight="bold",
                fontsize=fontsize,
                bbox=dict(facecolor='white', alpha=0.7),
                verticalalignment='bottom', horizontalalignment='right'
            )

    # --- Axis formatting ---
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    if config.get("metric", "") == "Relative_Efficienty_Loss":
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        #ax.yaxis.set_ylim(-1, 1)

    plt.ylabel(str(f"{y_label}"), fontweight="bold", fontsize=2 * fontsize)
    plt.xlabel(str(f"{x_label}"), fontweight="bold", fontsize=2 * fontsize)
    if config["pltLegend"]:
        plt.legend(frameon=False, prop={'weight': 'bold'})
    plt.grid(True)



    # ==========================================================
    # Inset zoom INSIDE the main figure
    # ==========================================================
    # config["x_zoom_interval"] are INDICES in x_data: (i_min, i_max)
    # config["y_zoom_interval"] are VALUE bounds: (y_min, y_max) OR (None, None) for auto
    if config["Add_Zoom"]:
        x_idx_min, x_idx_max = config["x_zoom_interval"]
        y_min, y_max = None,None#config["y_zoom_interval"]

        x1, x2 = x_data[x_idx_min], x_data[x_idx_max - 1]
        use_auto_y = (y_min is None) or (y_max is None)
        #print((y_min is None) or (y_max is None))

        # Position of inset: [left, bottom, width, height] in axes fraction
        inset_rect = config.get("inset_rect", [0.5, 0.5, 0.47, 0.47])
        axins = ax.inset_axes(inset_rect)

        local_y_min, local_y_max = +np.inf, 0

        for i, legend in enumerate(legends):
            color = "red" if legends[i] == "NE" else colors[i]
            marker = "" if legends[i] == "NE" else markers[i]
            if legends[i] in METHODS:
                color = COLORS_METHODS[legends[i]]
                marker = MARKERS_METHODS[legends[i]]

            x_vals = x_data[x_idx_min:x_idx_max]
            y_vals = y_data[i][x_idx_min:x_idx_max]

            axins.plot(
                x_vals[::step//4],
                y_vals[::step//4],
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=markersize,
                color=color,
                markeredgecolor="black",
            )

            if use_auto_y:
                local_y_min = min(local_y_min, np.min(y_vals))
                local_y_max = max(local_y_max, np.max(y_vals))/2


        # Scale + limits in inset
        #if ylog_scale:
        axins.set_yscale("log")
        # x-limits pour la zone de zoom (indices -> valeurs)
        # --- bornes X de la zone de zoom (indices -> valeurs) ---
        x1, x2 = x_data[x_idx_min], x_data[x_idx_max - 1]

        # --- bornes Y brutes à partir des données ---
        if use_auto_y:
            y_min_local = local_y_min
            y_max_local = local_y_max
        else:
            y_min_local, y_max_local = y_min, y_max
            print(y_min_local, y_max_local)

        # sécurité si tout est constant
        dy = y_max_local - y_min_local
        if dy <= 0:
            dy = max(abs(y_max_local), 1e-6)

        # ============ MARGE POUR COUVRIR LES MARKERS ============
        # marge relative (ex: 0.5 -> 50% du range au-dessus et en dessous)
        marker_margin = config.get("zoom_marker_margin", 0.08)

        y1 = y_min_local - marker_margin * dy
        y2 = y_max_local + marker_margin * dy

        # si tu veux être sûr de passer sous 0% (pour metric en %)
        if (not ylog_scale) and config.get("zoom_force_below_zero", False):
            y1 = min(y1, y2)

        # maintenant on fixe les limites de l'inset
        axins.set_xlim(x1, x2)

        if ylog_scale:
            axins.set_yscale("log")
            eps = config.get("zoom_log_eps", 1e-8)
            y1 = max(y1, eps)
            axins.set_ylim(y1, y2)
        else:
            axins.set_ylim(y1, y2)

        # nettoyage visuel de l'inset
        print(y1, y2)
        for label in axins.get_xticklabels() + axins.get_yticklabels():
            label.set_fontweight("bold")
            label.set_fontsize(fontsize *0.6)
        if config.get("metric", "") == "Relative_Efficienty_Loss":
            # Affiche les ticks en pourcentage (0–100%)
            axins.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=3))

        #if config.get("metric","") == "epsilon_error":
        #    axins.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        axins.set_xticklabels([])
        axins.set_ylabel("",fontsize=0.5*fontsize, fontweight="bold")
        #axins.set_yticklabels([])
        axins.tick_params(axis="both", length=0)

        # Rectangle + connecteurs
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset

        mark_inset(
            ax, axins,
            loc1=2, loc2=4,  # coins des connecteurs
            fc="none",
            ec="black",
            lw=config.get("zoom_rect_linewidth", 1),
        )

    plt.tight_layout()

    # --- Save main plot (with inset) ---
    figpath_plot = f"{saveFileName}_plot.pdf"
    plt.savefig(figpath_plot, format="pdf")

    # ==========================================================
    # Legend as separate figure (unchanged)
    # ==========================================================
    legend_handles = [
        Line2D(
            [0], [0],
            color=("red" if legends[k] == "NE"
                   else COLORS_METHODS[legends[k]] if legends[k] in METHODS
                   else colors[k]),
            markeredgecolor="black",
            linestyle=linestyle if linestyle != "" else "-",
            marker=("" if legends[k] == "NE"
                    else MARKERS_METHODS[legends[k]] if legends[k] in METHODS
                    else markers[k]),
            markersize=markersize,
            linewidth=linewidth
        )
        for k in range(len(legends))
    ]

    fig_legend = plt.figure(figsize=(12, 2))
    ax_leg = fig_legend.add_subplot(111)
    ax_leg.axis("off")

    ax_leg.legend(
        legend_handles,
        legends,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        prop={"weight": "bold", "size": fontsize},
        ncol=len(legends),
        loc="center",
        bbox_to_anchor=(0.5, 0.5)
    )

    figpath_legend = f"{saveFileName}_legend.pdf"
    fig_legend.savefig(figpath_legend, format="pdf", bbox_inches="tight")
    plt.close(fig_legend)

    # No separate zoom figure anymore: reuse main plot path
    figpath_zoom = figpath_plot

    return figpath_plot, figpath_legend, figpath_zoom








def plotGame_dim_N(
    config,
    x_data, y_data, y_data_avg, baseline, x_label, y_label, legends, saveFileName,
    ylog_scale, Players2See=[1, 2], fontsize=40, markersize=40, linewidth=12, linestyle="-",
    pltText=False, step=1
):
    """
    Trace, sauvegarde le plot principal + la légende séparée + un zoom (inset) sur un intervalle.
    Hypothèse de shape : y_data[k] a la forme (T, n_players).
    """

    # --- Fallbacks style si non définis globalement ---
    global METHODS, COLORS_METHODS, MARKERS_METHODS, colors, markers, legend_map
    if 'METHODS' not in globals(): METHODS = set(legends)
    if 'COLORS_METHODS' not in globals(): COLORS_METHODS = {}
    if 'MARKERS_METHODS' not in globals(): MARKERS_METHODS = {}
    if 'colors' not in globals():
        colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                  "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    if 'markers' not in globals():
        markers = ["o","s","D","^","v","<",">","P","X","*"]
    if 'legend_map' not in globals():
        legend_map = {}

    legends = [legend_map.get(l, l) for l in legends]

    # --- Préparation données ---
    Tm = min(config.get("T_plot", len(x_data)), len(x_data))
    x_data_i = np.array(x_data[:Tm])


    # --- Figure principale ---
    plt.figure(figsize=(18, 12))
    plt.rcParams.update({'font.size': fontsize})
    if ylog_scale:
        plt.yscale("log")

    legend_handles = []
    legend_labels  = []

    funcNo_NE = [i for i in legends if i != "Non-hybrid" and i != "NE"]

    # =========================
    # PLOT PRINCIPAL
    # =========================
    for i, fc2 in enumerate(legends):

        if fc2 == "NE" or fc2 == "Non-hybrid":
            y_ne = baseline
            plt.axhline(
                y=[y_ne], color="red", linestyle=linestyle, linewidth=linewidth, label="NE"
            )
            if pltText:
                plt.text(
                    x_data_i[-1], y_ne, f"{y_ne:.2e}",
                    fontweight="bold", fontsize=fontsize,
                    bbox=dict(facecolor="white", alpha=0.7),
                    va="bottom", ha="right"
                )
        else:
            k = i
            print(i)
            fc = legends[k]
            if config["num_hybrids"] == 1 and config["num_hybrid_set"] == 1:
                k = len(funcNo_NE) - i - 1
                fc = legends[k]

            color = "red" if fc == "Non-hybrid" else colors[k]
            marker = "" if fc == "Non-hybrid" else markers[k]

            if fc in METHODS:
                color = COLORS_METHODS[fc]
                marker = MARKERS_METHODS[fc]

            # --- Sélection des données (multi-joueurs) ---
            if config["num_hybrids"] == 1 and config["num_hybrid_set"] == 1:
                Yi = y_data  # (T, n_players)
            else:
                Yi = y_data[k]

            lab = fc
            label = ""
            label2 = ""
            Y2i = None

            if config["metric"] == "epsilon_error_Hybrid":
                Yi = y_data

            if config["metric"] in ["Bid", "Payoff"]:
                if config["num_hybrids"] == 1 and config["num_hybrid_set"] == 1:
                    Yi = y_data_avg[0]
                else:
                    Yi = y_data[k]
                lab = f"{fc}--avg."
                lab2 = f"{fc}--inst."
                if config['pltLegend']:
                    label = lab
                    label2 = lab2

                # pour la courbe inst.
                if config["num_hybrids"] == 1 and config["num_hybrid_set"] == 1:
                    Y2i = y_data[0]
                else:
                    Y2i = y_data[k]

            if config["metric"] in ["Avg_Bid", "Avg_Payoff"]:
                lab = f"{fc}."
                if config['pltLegend']:
                    label = lab
                    label2 = lab

            # --- Courbe principale (avg ou autre) ---
            plt.plot(
                x_data_i[::step],
                Yi[:, k][::step],
                linestyle=linestyle,
                linewidth=1.5 * linewidth,
                marker=marker,
                markersize=1.25 * markersize,
                color=color,
                markeredgecolor="black",
                label=label,
            )
            legend_labels.append(lab)
            legend_handles.append(
                Line2D(
                    [0], [0], color=color, marker=marker,
                    markersize=1.25 * markersize, markeredgecolor="black",
                    linestyle=linestyle, linewidth=linewidth
                )
            )

            # --- Courbe inst. si Bid/Payoff & hybrid ---
            if config["metric"] in ["Bid", "Payoff"] and config["num_hybrid_set"] >= 1 and Y2i is not None:
                tt = 0
                if config["num_hybrids"] == 1 and config["num_hybrid_set"] == 1 and k == 1:
                    tt = 0.3

                #print(len(Y2i), Y2i)
                #print(x_data_i.shape, len(Y2i[0]), Y2i[0])

                plt.plot(
                    x_data_i[::step],
                    Y2i[:, k][::step],
                    linestyle=":",
                    linewidth=1.5 * linewidth + tt,
                    color=color,
                    path_effects=[
                        pe.Stroke(linewidth=1.5 * linewidth + 1.5, foreground='black'),
                        pe.Normal()
                    ],
                    label=label2
                )
                legend_labels.append(lab2)
                legend_handles.append(
                    Line2D(
                        [0], [0], color=color, markeredgecolor="black",
                        linestyle=":", linewidth=linewidth
                    )
                )

            if pltText:
                last_x, last_y = x_data_i[-1], Yi[-1, k]
                plt.text(
                    last_x, last_y, f"{last_y:.2e}",
                    fontweight="bold", fontsize=fontsize,
                    bbox=dict(facecolor="white", alpha=0.7),
                    va="bottom", ha="right"
                )

    # --- Axes / labels / mise en forme ---
    ax = plt.gca()
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")

    if config.get("metric", "") == "Relative_Efficienty_Loss":
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))

    plt.ylabel(str(f"{y_label}"), fontweight="bold", fontsize=2 * fontsize)
    plt.xlabel(str(f"{x_label}"), fontweight="bold", fontsize=2 * fontsize)
    plt.legend(frameon=False, prop={'weight': 'bold'})
    plt.grid(True)

    if config["Add_Zoom"]:

        # =========================
        # INSET ZOOM DANS LA MÊME FIGURE
        # =========================
        x_idx_min, x_idx_max = config["x_zoom_interval"]
        inset_rect = config.get("inset_rect", [0.55, 0.55, 0.4, 0.4])
        axins = ax.inset_axes(inset_rect)

        local_y_min = +np.inf
        local_y_max = -np.inf

        # on retrace les courbes, mais restreintes à l'intervalle de zoom
        for i, fc2 in enumerate(legends):
            if fc2 == "NE" or fc2 == "Non-hybrid":
                # NE : ligne horizontale : on n'a besoin que de la valeur
                y_ne = baseline
                axins.axhline(
                    y=[y_ne], color="red", linestyle=linestyle, linewidth=linewidth
                )
                local_y_min = min(local_y_min, y_ne)
                local_y_max = max(local_y_max, y_ne)
            else:
                k = i
                fc = legends[k]
                if config["num_hybrids"] == 1 and config["num_hybrid_set"] == 1:
                    k = len(funcNo_NE) - i - 1
                    fc = legends[k]

                color = "red" if fc == "Non-hybrid" else colors[k]
                marker = "" if fc == "Non-hybrid" else markers[k]
                if fc in METHODS:
                    color = COLORS_METHODS[fc]
                    marker = MARKERS_METHODS[fc]

                # --- Sélection données comme plus haut ---
                if config["num_hybrids"] == 1 and config["num_hybrid_set"] == 1:
                    Yi = y_data
                else:
                    Yi = y_data[k]

                Y2i = None

                if config["metric"] == "epsilon_error_Hybrid":
                    Yi = y_data

                if config["metric"] in ["Bid", "Payoff"]:
                    if config["num_hybrids"] == 1 and config["num_hybrid_set"] == 1:
                        Yi = y_data_avg[0]
                        Y2i = y_data
                    else:
                        Yi = y_data_avg[k]
                        Y2i = y_data[k]

                # zoom sur l'intervalle [x_idx_min:x_idx_max]
                x_zoom = x_data_i[x_idx_min:x_idx_max]
                y_zoom = Yi[x_idx_min:x_idx_max, k]

                axins.plot(
                    x_zoom[::step],
                    y_zoom[::step],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    marker=marker,
                    markersize=markersize,
                    color=color,
                    markeredgecolor="black",
                )

                local_y_min = min(local_y_min, np.min(y_zoom))
                local_y_max = max(local_y_max, np.max(y_zoom))

                # courbe instantanée si nécessaire
                if config["metric"] in ["Bid", "Payoff"] and config["num_hybrid_set"] >= 1 and Y2i is not None:
                    y_zoom2 = Y2i[x_idx_min:x_idx_max, k]
                    axins.plot(
                        x_zoom[::step],
                        y_zoom2[::step],
                        linestyle=":",
                        linewidth=1.5 * linewidth,
                        color=color,
                        markeredgecolor="black",
                        path_effects=[
                            pe.Stroke(linewidth=1.5 * linewidth + 1.5, foreground='black'),
                            pe.Normal()
                        ],
                    )
                    local_y_min = min(local_y_min, np.min(y_zoom2))
                    local_y_max = max(local_y_max, np.max(y_zoom2))

        # --- Limites de l'inset + marge pour englober les gros markers ---
        x1, x2 = x_data_i[x_idx_min], x_data_i[x_idx_max - 1]
        dy = local_y_max - local_y_min if local_y_max > local_y_min else max(abs(local_y_max), 1e-6)
        marker_margin = config.get("zoom_marker_margin", 1.0)  # 100% du range
        y1 = local_y_min - marker_margin * dy
        y2 = local_y_max + marker_margin * dy

        pad_frac_x = config.get("zoom_pad_x", 0.05)
        dx = x2 - x1 if x2 > x1 else 1.0
        x1_pad = x1 - pad_frac_x * dx
        x2_pad = x2 + pad_frac_x * dx

        axins.set_xlim(x1_pad, x2_pad)

        if ylog_scale:
            axins.set_yscale("log")
            eps = config.get("zoom_log_eps", 1e-8)
            y1 = max(y1, eps)
            axins.set_ylim(y1, y2)
        else:
            axins.set_ylim(y1, y2)

        # nettoyage visuel de l'inset (aucun label/ticks)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axins.tick_params(axis="both", which="both", length=0, labelsize=0)
        axins.set_xlabel("")
        axins.set_ylabel("")

        # rectangle + connecteurs
        mark_inset(
            ax, axins,
            loc1=2, loc2=4,
            fc="none",
            ec="black",
            lw=config.get("zoom_rect_linewidth", 3),
        )

    plt.tight_layout()

    # Sauvegarde du plot principal (avec inset)
    figpath_plot = f"{saveFileName}_plot.pdf"
    plt.savefig(figpath_plot, format="pdf")

    # =========================
    # LÉGENDE SÉPARÉE
    # =========================
    rows = max(1, len(Players2See))
    n_items = len(legend_labels)
    ncol = int(np.ceil(n_items / rows)) if n_items else 1

    fig_legend = plt.figure(figsize=(12, 2))
    ax_leg = fig_legend.add_subplot(111)
    ax_leg.axis("off")
    ax_leg.legend(
        legend_handles, legend_labels,
        frameon=True, facecolor="white", edgecolor="black",
        prop={"weight": "bold", "size": fontsize},
        ncol=ncol, loc="center", bbox_to_anchor=(0.5, 0.5)
    )
    figpath_legend = f"{saveFileName}_legend.pdf"
    fig_legend.savefig(figpath_legend, format="pdf", bbox_inches="tight")
    plt.close(fig_legend)

    # plus de figure zoom séparée : on renvoie le même chemin
    figpath_zoom = figpath_plot

    return figpath_plot, figpath_legend, figpath_zoom



def plotGame_Hybrid_last(
        config,
        x_data, y_data,y_data_avg, baseline,
        x_label, y_label, legends, saveFileName, funcs_=["SBRD", "DAE"],
        ylog_scale=False, Players2See=[1, 2], fontsize=40,
        markersize=40, linewidth=12, linestyle="-",
        pltText=False, step=1, tol=1e-3,
        label_fontsize=None,
):
    # --- NEW: label_fontsize softer than 2*fontsize ---
    if label_fontsize is None:
        label_fontsize = int(1.5 * fontsize)

    # --- NEW: use fig, ax directly ---
    fig, ax = plt.subplots(figsize=(18, 12))
    y_data_Hybrid = y_data

    plt.rcParams.update({'font.size': fontsize})
    funcs_ = [legend_map.get(l, l) for l in funcs_]

    if ylog_scale:
        ax.set_yscale("log")

    legend_handles = []
    try:
        curve_NE = baseline
    except Exception:
        curve_NE = baseline

    funcNo_NE = [i for i in funcs_ if i != "Non-hybrid"]
    legend_handles = []
    legend_labels = []
    curves = []
    curves_agg = []
    legend_cache = set()

    # --- Construire les courbes hybrides finales ---
    if config["num_hybrid_set"] == 1:
        for i, fc in enumerate(funcNo_NE):
            curve = []
            curve_agg = []
            if config["metric"] in ["Payoff", "Bid", "Avg_Payoff", "Avg_Bid"]:
                for j in range(config["num_hybrids"]):
                    curve.append(y_data_Hybrid[j][-1][i])
                    curve_agg.append(y_data_avg[j][-1][i])
            else:
                for j in range(config["num_hybrids"]):
                    curve.append(y_data[j][-1])
                    curve_agg.append(y_data_avg[j][-1])
            curves.append(curve)
            curves_agg.append(curve_agg)
    else:
        k = 0
        for i, fc in enumerate(funcNo_NE):
            curve = []
            curve_agg = []
            if config["metric"] in ["Payoff", "Bid", "Avg_Payoff", "Avg_Bid"]:
                for j in range(config["num_hybrids"]):
                    curve.append(y_data_Hybrid[k][-1][i])
                    curve_agg.append(y_data_avg[k][-1][i])
                    k += 1
            else:
                for j in range(config["num_hybrids"]):
                    curve.append(y_data_Hybrid[k][-1])
                    curve_agg.append(y_data_avg[k][-1])
                    k += 1
            curves.append(curve)
            curves_agg.append(curve_agg)

    # --- Plot principal ---
    for j, fc in enumerate(funcs_):
        color = "red" if fc == "Non-hybrid" else colors[j]
        marker = "" if fc == "Non-hybrid" else markers[j]

        if fc in METHODS:
            color = COLORS_METHODS[fc]
            marker = MARKERS_METHODS[fc]

        if fc not in legend_cache and fc!= "Non-hybrid" and fc != "NE":
            legend_cache.add(fc)

            big_marker = Line2D(
                [0], [0],
                color=color,  # line color
                marker=marker,  # same shape as methods
                markersize=markersize *1.25,  # big marker (avg.)
                markerfacecolor=color,
                markeredgecolor="black",
                linestyle="--",
                linewidth=linewidth,
            )

            small_marker = Line2D(
                [0], [0],
                color="black",  # marker inside = black
                marker=marker,  # same shape
                markersize=markersize * 0.7,  # small marker (inst.)
                markerfacecolor=colors[j],
                markeredgecolor="black",
                linestyle="",  # no line for the small one
            )
            if config["metric"] in ["Payoff", "Bid"]:
                legend_handles.append((big_marker, small_marker))
                legend_labels.append(rf"{fc} (avg./inst.)")
            else:
                legend_handles.append((big_marker))
                legend_labels.append(rf"{fc}")


        if fc == "Non-hybrid":
            # ligne horizontale rouge à la valeur baseline
            continue
        else:
            curve = curves[j]  # j aligne avec funcs_ en supposant funcs_[0:len(funcNo_NE)]
            curve_agg = curves_agg[j]
            label = ""
            if config["pltLegend"]:
                label = f"{fc}"
            ax.plot(
                x_data[1:8],
                curve_agg[1:8],
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=1.25 * markersize,
                color=color,
                label=label,
                markeredgecolor="black",
            )

            ax.plot(
                x_data[1:8],
                curve[1:8],
                    linestyle=":",
                    linewidth=1.5 * linewidth,
                    color=color,
                    path_effects=[
                        pe.Stroke(linewidth=1.5 * linewidth + 1.5, foreground='black'),
                        pe.Normal()
                    ],
                    #label=label2
                )

    # ligne horizontale Non-hybrid si présente
    if funcs_[-1] == "Non-hybrid":
        ax.axhline(
            y=curve_NE,
            color="red",
            linestyle=linestyle,
            linewidth=linewidth,
            label=f"Non-hybrid"
        )

        if pltText:
            lastValue = f"{curve_NE:.2e}"
            if config.get("metric", "") == "Relative_Efficienty_Loss":
                lastValue = f"{curve_NE:.3f}%"
            ax.text(
                x_data[-1], curve_NE, lastValue,
                fontweight="bold",
                fontsize=fontsize,
                bbox=dict(facecolor="white", alpha=0.7),
                verticalalignment="bottom",
                horizontalalignment="right"
            )

    # --- Axis formatting ---
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    if config.get("metric", "") == "Relative_Efficienty_Loss":
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=1))

    ax.set_ylabel(f"{y_label}", fontweight="bold", fontsize=2 * fontsize)
    ax.set_xlabel(f"{x_label}", fontweight="bold", fontsize=2 * fontsize)
    ax.grid(True)

    # --- Légende principale ---
    ax.legend(
        frameon=False,
        prop={'weight': 'bold'}
    )

    if config["Add_Zoom"]:
        x_idx_min, x_idx_max = config["x_zoom_interval"]

        local_y_min = +np.inf
        local_y_max = -np.inf

        inset_rect = config.get("inset_rect", [0.5, 0.5, 0.47, 0.47])
        axins = ax.inset_axes(inset_rect, transform=ax.transAxes)

        for j, fc in enumerate(funcs_):
            color = "red" if fc == "Non-hybrid" else colors[j]
            marker = "" if fc == "Non-hybrid" else markers[j % len(markers)]
            if fc in METHODS:
                color = COLORS_METHODS[fc]
                marker = MARKERS_METHODS[fc]

            if fc == "Non-hybrid":
                local_y_min = min(local_y_min, curve_NE)
                local_y_max = max(local_y_max, curve_NE)
                axins.axhline(
                    y=curve_NE,
                    color="red",
                    linestyle=linestyle,
                    linewidth=linewidth,
                )
            else:
                if fc in funcNo_NE:
                    idx = funcNo_NE.index(fc)
                    curve_full = curves[idx]
                    curve_full_agg = curves_agg[idx]
                else:
                    continue

                x_vals = x_data[x_idx_min-1:x_idx_max]
                y_vals = curve_full[x_idx_min-1:x_idx_max]
                y_vals_agg = curve_full_agg[x_idx_min-1:x_idx_max]

                local_y_min = min(local_y_min, min(np.min(y_vals), np.min(y_vals_agg)))
                local_y_max = max(local_y_max, max(np.max(y_vals), np.max(y_vals_agg)))
                print(f"local_y_max:{local_y_max},{local_y_min}")

                axins.plot(
                    x_vals[::step],
                    y_vals_agg[::step],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    marker=marker,
                    markersize=markersize * 0.9,
                    color=color,
                    markeredgecolor="black",
                )
                axins.plot(
                    x_vals[::step],
                    y_vals[::step],
                    linestyle=":",
                    linewidth=1.5 * linewidth,
                    color=color,
                    path_effects=[
                        pe.Stroke(linewidth=1.5 * linewidth + 1.5, foreground='black'),
                        pe.Normal()
                    ],
                    # label=label2
                )
            #axins.set_yscale("log")
           # axins.set_ylabel([])  # Supprime le label


        if "y_zoom_interval" in config and config["y_zoom_interval"] is not None:
            y_min, y_max = local_y_min, local_y_max#config["y_zoom_interval"]
            #print("55555")
            use_auto_y = True
        else:
            y_min, y_max = local_y_min, local_y_max
            use_auto_y = True

        if use_auto_y:
            y1 = local_y_min
            y2 = local_y_max
        else:
            y1, y2 = y_min, y_max

        pad_abs_bottom = config.get("zoom_pad_y_abs_bottom", None)
        pad_abs_top    = config.get("zoom_pad_y_abs_top", None)

        if pad_abs_bottom is not None:
            y1 = pad_abs_bottom
        if pad_abs_top is not None:
            y2 = y2 + pad_abs_top

        pad_frac_x = config.get("zoom_pad_x", 0.02)
        pad_frac_y = config.get("zoom_pad_y", 0.5)

        x1, x2 = x_data[x_idx_min-1], x_data[x_idx_max - 1]
        dx = x2 - x1 if x2 > x1 else 1.0
        dy = y2 - y1 if y2 > y1 else 1.0


        x1_pad = x1 - pad_frac_x * dx
        x2_pad = x2 + pad_frac_x * dx
        y1_pad = y1 - pad_frac_y * dy
        y2_pad = y2 + pad_frac_y * dy
        print(y1_pad, y2_pad, y1, y2)
        axins.set_xlim(x1_pad, x2_pad)

        if ylog_scale:
            axins.set_yscale("log")
            eps = config.get("zoom_log_eps", 1e-8)
            y1_pad = max(y1_pad, eps)
            axins.set_ylim(y1_pad, y2_pad)
        else:
            axins.set_ylim(y1_pad, y2_pad)
        #print(y1_pad, y2_pad, y1, y2)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axins.tick_params(axis="both", length=0)

        mark_inset(
            ax, axins,
            loc1=2, loc2=4,
            fc="none",
            ec="black",
            lw=config.get("zoom_rect_linewidth", 3),
        )

    # --- NEW: adjust margins & tight layout before save ---
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.97)
    fig.tight_layout()

    figpath_plot = f"{saveFileName}_plot.pdf"
    fig.savefig(figpath_plot, format="pdf", bbox_inches="tight")

    # ==========================================================
    # Légende séparée
    # ==========================================================
    fig_legend = plt.figure(figsize=(12, 2))
    ax_leg = fig_legend.add_subplot(111)
    ax_leg.axis("off")

    ax_leg.legend(
        legend_handles,
        funcNo_NE,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        prop={"weight": "bold", "size": fontsize},
        ncol=len(legends),
        loc="center",
        bbox_to_anchor=(0.5, 0.5)
    )

    figpath_legend = f"{saveFileName}_legend.pdf"
    fig_legend.savefig(figpath_legend, format="pdf", bbox_inches="tight")
    plt.close(fig_legend)
    plt.close(fig)

    figpath_zoom = figpath_plot
    return figpath_plot, figpath_zoom, figpath_legend

import matplotlib
def plotGame_dim_N_last(config,
        x_data, y_data, x_label, y_label, legends, saveFileName, funcs_=["SBRD","DAE"],
        ylog_scale=False, Players2See=[1, 2], fontsize=40,
        markersize=40, linewidth=12, linestyle="-",
        pltText=False, step=1, tol=1e-3
):


    plt.figure(figsize=(18, 12))
    #y_data = np.array(y_data, dtype=object)  # s'assurer que les sous-tableaux passent bien
    y_data_Hybrid = y_data
    if config["metric"] in ["Payoff","Bid"]:
        col = torch.arange(1, config["T"] + 1)

        agg_utility = y_data[-1][0]
    else:
        agg_utility = y_data_Hybrid
    plt.rcParams.update({'font.size': fontsize})
    funcs_ = [legend_map.get(l, l) for l in funcs_]
    #print(y_data_Hybrid)
    #print(222)
    if ylog_scale:
        plt.yscale("log")

    legend_handles = []
    legend_labels = []
    curves = []
    curves_agg = []
    funcNo_NE = [i for i in funcs_ if i!="Non-hybrid" and i != "NE"]


    if config["num_hybrids"] != 1 and config["num_hybrid_set"]>1:

        for j, fc in enumerate(funcNo_NE):
            curve = []
            curve_agg =[]

            for i in range(len(y_data_Hybrid)-1):
                #print(f"y_data_Hybrid{y_data_Hybrid[i]}")
                curve.append((y_data_Hybrid[i][j]))
                curve_agg.append((agg_utility[i][j]))
            curves.append(curve)
            curves_agg.append(curve_agg)

    elif config["metric"] not in ["Relative_Efficienty_Loss", "Potential", "Speed"]:
        for j, fc in enumerate(funcNo_NE):
            curve = []
            curve_agg =[]

            for i in range(len(y_data_Hybrid)-1):

                #print(f"y_data_Hybrid{y_data_Hybrid[i]}")
                curve.append((y_data_Hybrid[i][j]))
                #curve_agg.append((agg_utility[i][j]))
            curves.append(curve)
            #curves_agg.append(curve_agg)
        curve_NE = [y_data[-1][0]]


    legend_cache = set()

    for j, fc in enumerate(funcNo_NE):

        color = "red" if fc == "Non-hybrid" or fc == "NE" else colors[j]
        marker = "" if "Non-hybrid" or fc == "NE" else markers[j % len(markers)]
        if fc in METHODS:
            print(fc)
            color = COLORS_METHODS[fc]
            marker = MARKERS_METHODS[fc]

        if fc not in legend_cache and fc!= "Non-hybrid" and fc != "NE":
            legend_cache.add(fc)

            big_marker = Line2D(
                [0], [0],
                color=color,  # line color
                marker=marker,  # same shape as methods
                markersize=markersize *1.25,  # big marker (avg.)
                markerfacecolor=color,
                markeredgecolor="black",
                linestyle="--",
                linewidth=linewidth,
            )

            small_marker = Line2D(
                [0], [0],
                color="black",  # marker inside = black
                marker=marker,  # same shape
                markersize=markersize * 0.7,  # small marker (inst.)
                markerfacecolor=colors[j],
                markeredgecolor="black",
                linestyle="",  # no line for the small one
            )
            if config["metric"] in ["Payoff", "Bid"]:
                legend_handles.append((big_marker, small_marker))
                legend_labels.append(rf"{fc} (avg./inst.)")
            else:
                legend_handles.append((big_marker))
                legend_labels.append(rf"{fc}")

        curve = curves[j]
        #curve_agg = curves_agg[j]
        x_data_i = x_data[:config[("T_plot")]]
        curve = curve[:config["T_plot"]]
        curve2=curve.copy()
        #curve_agg = curve_agg[:config["T_plot"]]

        label = ""
        if config["pltLegend"]:
            label = f"{fc}"

        #if config["metric"] in ["Payoff","Bid"]:
        #    curve2=curve_agg

        if fc not in legend_cache:
            label_inst = rf"{fc} (inst./avg.)"
            legend_cache.add(fc)
        else:
            label_inst = None
        plt.plot(
            x_data_i[::step],
            curve2[::step],
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=1.25 * markersize,
            color=color,
            #label=label,
            markeredgecolor="black",
        )
        if pltText:
            lastValue = f"{curve[-1]:.2e}"
            if config.get("metric", "") == "Relative_Efficienty_Loss":
                lastValue = f"{curve[-1]:.3f}%"
            plt.text(x_data[-1], curve[-1], lastValue,
                fontweight="bold",
                fontsize=fontsize,
                bbox=dict(facecolor="white", alpha=0.7),
                verticalalignment="bottom",
                horizontalalignment="right")

    # légendes et labels
    if funcs_[-1]== "Non-hybrid" or  funcs_[-1] == "NE":

        plt.axhline(
            y=curve_NE,
            color="red",
            linestyle="--",
            linewidth=linewidth,
            label=f"{funcs_[-1]}"
    )
        legend_cache.add(f"{funcs_[-1]}")
        legend_handles.append(
            Line2D(
                [0], [0],
                color="red",
                linestyle="--",
                linewidth=linewidth,
            )
        )
        legend_labels.append(f"{funcs_[-1]}")
        if pltText:
            lastValue = f"{curve_NE[-1]:.2e}"
            if config.get("metric", "") == "Relative_Efficienty_Loss":
                lastValue = f"{curve_NE[-1]:.3f}%"
            plt.text(x_data[-1], curve_NE[-1], lastValue,
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
    if  config["num_hybrid_set"] == 1 and  config["num_hybrids"]>1:
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    if config.get("metric", "") == "Relative_Efficienty_Loss":
        # Affiche les ticks en pourcentage (0–100%)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=3))

    plt.ylabel(str(f"{y_label}"), fontweight="bold", fontsize=2*fontsize)
    plt.xlabel(str(f"{x_label}"), fontweight="bold", fontsize=2*fontsize)
    plt.grid(True)

    # 🔑 Horizontal legend

    plt.legend(legend_handles, legend_labels,
       handler_map={
           tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None)
       },
        frameon=False,
        prop={'weight': 'bold'}
    )
    plt.tight_layout()



    # --- Save plot without legend ---
    figpath_plot = f"{saveFileName}_plot.pdf"
    plt.savefig(figpath_plot, format="pdf")


    # --- Save legend separately ---
    fig_legend = plt.figure(figsize=(12, 2))  # wide & short for horizontal layout
    ax = fig_legend.add_subplot(111)
    ax.axis("off")

    ax.legend(
        legend_handles,
        funcNo_NE,
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


    x_min, x_max = config["x_zoom_interval"]

    fig_zoom = plt.figure(figsize=(18, 12))
    ax_zoom = fig_zoom.add_subplot(111)

    for i, fc in enumerate(funcNo_NE):
        color = "red" if fc == "Non-hybrid" or fc == "NE" else colors[j]
        marker = "" if fc == "Non-hybrid" or fc == "NE" else markers[j % len(markers)]
        if fc in METHODS:
            color = COLORS_METHODS[fc]
            marker = MARKERS_METHODS[fc]
        #n = y_data[i].shape[1]


        x_vals = x_data[x_min-1:x_max]  # 5 derniers x

        y_vals = curves[i][x_min-1:x_max]  # 5 derniers y
        #x_vals_avg = x_data[x_min-1:x_max]  # 5 derniers x

        #y_vals_agg = curves_agg[i][x_min-1:x_max]  # 5 derniers y

        if fc not in legend_cache:
            label_inst = rf"{fc} (inst./avg.)"
            legend_cache.add(fc)
        else:
            label_inst = None

        ax_zoom.plot(
            x_vals,
            y_vals,
            #alpha=1, lw=5.2,
            # label=label,
            marker=marker,
            linestyle="--",
            markersize=2 * markersize,
            color=color,
            markerfacecolor=colors,  # FILL of marker = BLACK
            markeredgecolor="black",  # BORDER = BLACK
        )



        if pltText:
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
    #ax_zoom.set_xlim(x_data[-2], x_data[-1])
    #ax_zoom.set_ylim(plt.ylim())  # reprendre les bornes du plot principal
    if funcs_[-1]=="Non-hybrid" or funcs_[-1] == "NE":

        ax_zoom.axhline(
            y=curve_NE,
            color="red",
            linestyle=linestyle,
            linewidth=linewidth,
            label=f"Non-hybrid"
        )
    for label in ax_zoom.get_xticklabels() + ax_zoom.get_yticklabels():
        label.set_fontweight("bold")

    if config.get("metric", "") == "Relative_Efficienty_Loss":
        # Affiche les ticks en pourcentage (0–100%)
        ax_zoom.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))

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
    #print(2)
    #fig_zoom.savefig(figpath_zoom, format="pdf")

    plt.close(fig_zoom)





    return figpath_plot, figpath_zoom, figpath_legend


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



def plotGame_Jain(config,results, list_gamma,
    ylog_scale=False, fontsize=40, markersize=40, linewidth=12,
    linestyle="-", pltText=False, step=1,tol=1e-6
):
    plt.figure(figsize=(18, 12))

    plt.rcParams.update({'font.size': fontsize})
    x_data = list_gamma.copy()

    if ylog_scale:
        plt.yscale("log")

    # --- Plot curves ---
    l=0
    for i, (n, gamma_map) in enumerate(sorted(results.items(), key=lambda kv: kv[0])):
        y_data_i = [gamma_map.get(g, None) for g in list_gamma]

        plt.plot(
            x_data[::step],
            y_data_i[::step],
            linestyle=linestyle,
            linewidth=linewidth,
            marker=markers[i],
            markersize=1 * markersize,
            color=colors[i],
            label=rf"$n={n}$",
            markeredgecolor="black",
        )
    # --- Axis formatting ---
    ax = plt.gca()

    # --- Axis formatting ---
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")



    # --- Save plot without legend ---
    #figpath_plot = f"{saveFileName}_plot.pdf"
    #plt.savefig(figpath_plot, format="pdf")

    plt.ylabel(str(rf"Jain Index"), fontweight="bold", fontsize=2*fontsize)
    plt.xlabel(str(rf"$\gamma$"), fontweight="bold", fontsize=2*fontsize)
    if config["pltLegend"]:
        plt.legend(frameon=False, prop={'weight': 'bold'})
    plt.grid(True)
    plt.tight_layout()

    # --- Save plot without legend ---
    figpath_plot = f"JainIndex_plot.pdf"
    plt.savefig(figpath_plot, format="pdf")


    return figpath_plot