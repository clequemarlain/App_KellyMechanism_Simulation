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

#from main_by_learning import config

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
    "DA": rf"DAQ$_F$",
    "OGD_V": rf"OGD$_V$",
    "OGD": rf"OGD$_F$",
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

import torch


def solve_br_player_multi_resource(
    A_i, p_i, c_i, alpha, price, Y_i, epsilon, tol=1e-10, max_iter=200
):
    """
    Solve
        argmax_{z_i in R_i} sum_k Y_i[k] [ A_i[k] V_alpha(z_k/(z_k+p_k)) - price[k] z_k ]
    subject to
        z_k >= epsilon * Y_i[k],   sum_k z_k <= c_i.

    Parameters
    ----------
    A_i, p_i, alpha, price, Y_i : 1D tensors of shape (d,)
    c_i : scalar
    epsilon : scalar
    """

    device = A_i.device
    dtype = A_i.dtype
    d = A_i.numel()

    if not torch.is_tensor(price):
        price = torch.tensor(price, dtype=dtype, device=device)
    if price.ndim == 0:
        price = price.repeat(d)

    lower = epsilon * Y_i
    active = Y_i > 0

    def z_k_given_lambda(k, lam):
        if not active[k]:
            return torch.tensor(0.0, dtype=dtype, device=device)

        A = A_i[k]
        p = p_i[k]
        a = alpha[k]
        q = price[k]
        zmin = lower[k]

        # alpha = 0
        if a ==0:
            z = torch.sqrt(A * p / (q + lam)) - p
            return torch.maximum(z, zmin)

        # alpha = 1
        if a==1:
            disc = p**2 + 4.0 * A * p / (q + lam)
            z = (-p + torch.sqrt(disc)) / 2.0
            return torch.maximum(z, zmin)

        # alpha = 2
        if a==2:
            z = torch.sqrt(A * p / (q + lam))
            return torch.maximum(z, zmin)

        # general alpha: scalar bisection on z
        zmax = c_i - lower.sum() + zmin
        lo = zmin.clone()
        hi = torch.tensor(zmax, dtype=dtype, device=device)

        def g(z):
            return A * p * z**(-a) * (z + p)**(a - 2.0) - q - lam

        # enlarge hi if needed
        for _ in range(50):
            if g(hi) <= 0:
                break
            hi = 2.0 * hi

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if g(mid) > 0:
                lo = mid
            else:
                hi = mid
            if torch.abs(hi - lo) <= tol:
                break

        z = 0.5 * (lo + hi)
        return torch.maximum(z, zmin)

    def z_of_lambda(lam):
        z = torch.zeros(d, dtype=dtype, device=device)
        for k in range(d):
            z[k] = z_k_given_lambda(k, lam)
        return z

    # budget inactive?
    z0 = z_of_lambda(torch.tensor(0.0, dtype=dtype, device=device))
    if z0.sum() <= c_i + tol:
        return z0

    # otherwise outer bisection on lambda
    lam_lo = torch.tensor(0.0, dtype=dtype, device=device)
    lam_hi = torch.tensor(1.0, dtype=dtype, device=device)

    while z_of_lambda(lam_hi).sum() > c_i:
        lam_hi = 2.0 * lam_hi

    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        z_mid = z_of_lambda(lam_mid)
        if z_mid.sum() > c_i:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
        if torch.abs(lam_hi - lam_lo) <= tol:
            #print(_)
            break
    return z_of_lambda(0.5 * (lam_lo + lam_hi))

def optimal_x_multiresource_matrix(
    A_matrix: torch.Tensor,          # (n,d)
    alpha_k: torch.Tensor,      # (d,)
    D_mask: torch.Tensor,       # (n,d) bool
    c_vector: torch.Tensor,     # (n,d)
    eps: torch.Tensor,
    Y: torch.Tensor,
    S_k=None,                   # (d,) totals, default 1 each
    x_min: float = 0.0,
    delta = 0.01
):
    """
    Returns x_opt (n,d) solving the welfare-optimal ALLOCATION benchmark per resource.
    """
    device, dtype = A_matrix.device, A_matrix.dtype
    n, d = D_mask.shape
    if S_k is None:
        S_k = torch.ones(d, device=device, dtype=dtype)
    else:
        S_k = torch.as_tensor(S_k, device=device, dtype=dtype)

    x_opt = torch.zeros((n, d), device=device, dtype=dtype)

    for k in range(d):
        idx = torch.where(D_mask[:, k])[0]
        if idx.numel() == 0:
            continue

        a_eff = A_matrix[idx,k]#a_i[idx] * beta_k[k]
        Y_eff = Y[idx,k]
        eps_eff = eps[idx,k]
        c_eff = c_vector[idx]
        ak = float(alpha_k[k])

        x_k = compute_optimal_x_alpha(c_eff * Y_eff , a_eff, eps_eff, delta, 0, ak )
        #x_k = optimal_x_one_resource_kkt_bisect(a_k=a_eff,alpha=ak,S=float(S_k[k].item()),
        #    x_min=torch.max(x_min[:,k]), x_max=torch.max(x_max[:,k]),)
        x_opt[idx, k] = torch.tensor(x_k, dtype=dtype, device=device)

    val_opt = Valuation_matrix(x_opt, A_matrix, 0, alpha_k, D_mask)
    sw_opt = val_opt.sum()
    return x_opt, sw_opt




def alpha_fair_inverse_utility(y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Inverse of the standard alpha-fair utility above:
      alpha=1: exp(y)
      alpha!=1: ((1-alpha)*y)^(1/(1-alpha))
    """
    y = np.asarray(y, dtype=float)
    if np.isclose(alpha, 1.0):
        return np.exp(y)
    else:
        base = (1.0 - alpha) * y
        # In typical use, base should be >= 0 when alpha < 1.
        # For alpha > 1, base can be negative and still yield positive x
        # only if y is negative and the exponent is rational with odd denom etc.
        # Practically, your model parameters should make gamma_x well-defined.
        if np.any(base <= 0):
            # If you want strict feasibility, you can raise instead.
            base = np.maximum(base, 1e-18)
        return np.power(base, 1.0 / (1.0 - alpha))

def compute_optimal_x_alpha(
    c_vector,
    a_vector,
    eps,
    delta: float,
    d_vector,
    alpha: float,
):
    """
    Solve:
      maximize  sum_i [ a_i * V_alpha(x_i) + d_i ]
      s.t.      sum_i x_i = S
                eps_x_i <= x_i <= gamma_x_i

    where:
      eps_x_i = eps_i / (eps_i + sum(c) - c_i + delta)   (your min_fraction)
      S = sum(c)/(sum(c)+delta)
      gamma_x_i = V_alpha^{-1}((c_i - d_i)/a_i)          (generalized cap)

    For alpha>0: KKT gives a_i * V'(x_i)=lambda -> x_i=(a_i/lambda)^(1/alpha), then clip.
    For alpha=0: linear objective -> greedy fill by descending a_i.
    """
    a = np.asarray(a_vector, dtype=float)
    c = np.asarray(c_vector, dtype=float)
    eps = np.asarray(eps, dtype=float)
    d = np.asarray(d_vector, dtype=float)

    def min_fraction(eps: np.ndarray, budgets: np.ndarray, delta: float):
        return eps / (eps + np.sum(budgets) - budgets + delta)

    # lower bounds
    eps_x = min_fraction(eps, c, delta)

    # total available resource
    C = np.sum(c)
    S = C / (C + delta)

    # upper bounds via generalized inverse utility:
    # a_i * V(x_i) + d_i <= c_i  =>  V(x_i) <= (c_i - d_i)/a_i
   # print(c,a)
    y_cap = 0 * (c - d)
    if a.all()!=0:
        y_cap = (c - d) / a
    gamma_x = alpha_fair_inverse_utility(y_cap, alpha)

    # Edge case: if S is outside feasible sum interval, KKT solution saturates
    sum_min = np.sum(eps_x)
    sum_max = np.sum(gamma_x)

    if S <= sum_min:
        return eps_x.copy()
    if S >= sum_max:
        return gamma_x.copy()

    # --- alpha = 0: linear case, maximize sum_i a_i x_i over a box + sum constraint
    if np.isclose(alpha, 0.0):
        x = eps_x.copy()
        remaining = S - np.sum(x)

        order = np.argsort(-a)  # descending a_i
        for i in order:
            if remaining <= 0:
                break
            add = min(remaining, gamma_x[i] - x[i])
            if add > 0:
                x[i] += add
                remaining -= add
        return x

    # --- alpha > 0: bisection on lambda with clipped KKT form
    alpha_val = float(alpha)

    def x_from_lambda(lmbda: float) -> np.ndarray:
        lmbda = max(lmbda, 1e-18)
        x_unclipped = np.power(a / lmbda, 1.0 / alpha_val)
        return np.minimum(np.maximum(x_unclipped, eps_x), gamma_x)

    def total_x_minus_S(lmbda: float) -> float:
        return float(np.sum(x_from_lambda(lmbda)) - S)

    # Lambda bounds induced by box:
    # x_i <= gamma_i  => (a_i/lambda)^(1/alpha) <= gamma_i  => lambda >= a_i / gamma_i^alpha
    # x_i >= eps_i    => lambda <= a_i / eps_i^alpha
    lmbda_min = np.min(a / np.power(gamma_x, alpha_val))
    lmbda_max = np.max(a / np.power(eps_x, alpha_val))

    # Safety in case of numerical weirdness
    lmbda_min = max(lmbda_min, 1e-18)
    lmbda_max = max(lmbda_max, lmbda_min * 1.000001)

    # Bisection
    lmbda = optimize.bisect(total_x_minus_S, lmbda_min, lmbda_max, xtol=1e-12, maxiter=200)

    return x_from_lambda(lmbda)


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


def V_func(x, alpha):#,a_vector_k,beta_k):
    if alpha == 1:
        V = torch.zeros_like(x)
        mask = x > 0
        V = torch.log(x)
    else:
        V = 1 / (1 - alpha) * (x) ** (1 - alpha)
    return V

def Q1(acc_gradient, eps, c, price):
    #price = price.view(1, -1)  # [1, d]
    price_s = float(price) if torch.is_tensor(price) and price.numel() == 1 else price
    z = torch.clamp(acc_gradient, min=eps / price_s, max=c / price_s)

    return z


def Q_simplex_i(y: torch.Tensor, eps, c):
    """
    Euclidean projection onto:
        { z : z >= eps, sum_k z_k <= c }

    Water-filling (your Lemma):
        v = y - eps
        z = eps + (v - tau)_+

    Parameters
    ----------
    y   : (d,)
    eps : scalar or (d,)
    c   : scalar (budget for this player)

    Returns
    -------
    z   : (d,)
    """
    y = y.to(dtype=torch.float64)

    # eps as tensor, broadcast to (d,)
    if not torch.is_tensor(eps):
        eps = torch.tensor(eps, device=y.device, dtype=y.dtype)
    eps = eps.to(device=y.device, dtype=y.dtype)
    if eps.ndim == 0:
        eps_vec = eps.expand_as(y)
    else:
        eps_vec = eps
        if eps_vec.shape != y.shape:
            raise ValueError(f"eps must be scalar or shape {y.shape}, got {eps_vec.shape}")

    # budget after shift
    C = (c - eps_vec.sum()).item()  # python float

    if C <= 0.0:
        return eps_vec.clone()

    v = y - eps_vec
    #print(f"y:{y}")
    # already feasible
    if torch.clamp(v, min=0.0).sum().item() <= C:
        return eps_vec + torch.clamp(v, min=0.0)

    # --- water-filling ---
    v_sorted, _ = torch.sort(v, descending=True)
    v_sorted_pos = torch.clamp(v_sorted, min=0.0)
    cssv = torch.cumsum(v_sorted_pos, dim=0)

    d = y.numel()
    # find rho = max m such that v^(m) > (sum_{k<=m} v^(k) - C)/m
    # keep it readable
    rho = 1
    for m in range(d):
        if v_sorted_pos[m].item() > (cssv[m].item() - C) / (m + 1):
            rho = m + 1

    tau = (cssv[rho - 1].item() - C) / rho
    tau = max(tau, 0.0)

    z = eps_vec + torch.clamp(v - tau, min=0.0)
    return z


def Q_simplex(y: torch.Tensor, eps, c: torch.Tensor, Y: torch.Tensor):
    """
    Apply Q_simplex_i row-wise.

    Parameters
    ----------
    y : (n,d)
    eps : scalar or (d,)
    c : (n,) budgets

    Returns
    -------
    z : (n,d)
    """
    z = torch.empty_like(y, dtype=torch.float64)
    c = c.to(device=y.device, dtype=torch.float64).view(-1)

    for i in range(y.shape[0]):
        z[i,:] = Q_simplex_i(y[i,:], eps * Y[i,:], c[i].item())
    return z




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

    return  br#Q1(br, eps, c_vector, price)

def Valuation(x, a_vector, d_vector, alpha, beta, Y):
    V = sum([beta[k]* Y[:,k] * V_func(x[:,k], alpha[k]) for k,beta_k in enumerate(alpha)])
    return a_vector * V + d_vector

def Valuation_matrix(x, A_matrix, d_vector, alpha, Y):
    V = sum([A_matrix[:, k]* Y[:,k] * V_func(x[:,k], alpha[k]) for k,beta_k in enumerate(alpha)])
    return V + d_vector
def Payoff_matrix(x, z, A_matrix, d_vector, alpha, price,Y, k=None):
    if k is None:
        U = Valuation_matrix(x, A_matrix, d_vector, alpha, Y) - price * torch.sum(Y * z, dim=1)
    else:
        U = A_matrix* Y[:,k] * V_func(x, alpha) - price*Y[:,k] * z
    return U
def Payoff(x, z, a_vector, d_vector, alpha, price, beta,Y, k=None):
    #print(alpha)
    if k is None:
        U = Valuation(x, a_vector, d_vector, alpha, beta, Y) - price * torch.sum(Y * z, dim=1)
    else:
        U = beta[k]* Y[:,k] * a_vector * V_func(x, alpha[k]) - price*Y[:,k] * z
    return U


# =========================================================
# Fairness metrics
# =========================================================
# Used to (i) fill in the Jain's-index trace that was previously
# always zero, and (ii) provide a generalized alpha-fair fairness
# index consistent with the alpha-fair utility family (V_func)
# already used throughout this file. Work on any nonnegative vector
# (resource shares, bids, or per-agent utilities) along an arbitrary
# tensor dimension, so they can be used both inside the per-iteration
# learning loop and on final results.

def jain_index(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """
    Jain's fairness index: J(x) = (sum_i x_i)^2 / (n * sum_i x_i^2).
    Range: [1/n, 1]; 1 = perfectly fair (all entries equal),
    1/n = maximally unfair (all mass on a single agent).
    x must be nonnegative; negative entries are clamped to 0.
    """
    x = torch.clamp(x, min=0.0)
    n = x.shape[dim]
    s1 = x.sum(dim=dim)
    s2 = (x ** 2).sum(dim=dim)
    return (s1 ** 2) / (n * s2 + eps)


def alpha_fairness_index(x: torch.Tensor, alpha: float, dim: int = -1,
                          eps: float = 1e-12) -> torch.Tensor:
    """
    Generalized alpha-fairness index of Lan, Kao, Chiang & Sabharwal
    (2010), normalized so it always returns a value in (0, 1] with
    1 = perfectly fair, matching Jain's index exactly at alpha=2:

        f_alpha(x) = | (sum_i x_i)^alpha / (n^(alpha-1) * sum_i x_i^alpha) |

    At alpha=1 the family is singular (0/0 in the limit); we use the
    continuous limit instead, i.e. normalized Shannon entropy of the
    normalized allocation -- the natural fairness notion for the
    log-utility (proportional fairness) case used in this paper:

        f_1(x) = H(x / sum x) / log(n)   in [0, 1]

    At alpha=0 the raw formula degenerates to 1 for any strictly
    positive x (a known artifact of this family at alpha=0), so for
    alpha=0 (linear utility) we fall back to Jain's index instead,
    the standard choice for linear/proportional shares.
    """
    x = torch.clamp(x, min=eps)
    n = x.shape[dim]

    if abs(alpha - 1.0) < 1e-9:
        p = x / (x.sum(dim=dim, keepdim=True) + eps)
        H = -(p * torch.log(p + eps)).sum(dim=dim)
        return H / torch.log(torch.tensor(float(n), dtype=x.dtype, device=x.device))

    if abs(alpha) < 1e-9:
        return jain_index(x, dim=dim, eps=eps)

    s1 = x.sum(dim=dim) ** alpha
    s2 = (x ** alpha).sum(dim=dim)
    f = s1 / (float(n) ** (alpha - 1) * s2 + eps)
    return torch.abs(f)


def per_resource_jain_index(x_alloc: torch.Tensor, Y: torch.Tensor,
                             eps: float = 1e-12) -> torch.Tensor:
    """
    Jain's index computed per resource, restricted to the agents that
    actually consume that resource (Y[:,k] == 1).
    x_alloc: (n,d) allocation shares. Y: (n,d) participation mask.
    Returns: (d,) tensor, one Jain index per resource.
    """
    x_masked = x_alloc * Y
    s1 = x_masked.sum(dim=0)
    s2 = (x_masked ** 2).sum(dim=0)
    n_active = Y.sum(dim=0).clamp(min=1.0)
    return (s1 ** 2) / (n_active * s2 + eps)


def per_resource_alpha_fair_index(x_alloc: torch.Tensor, Y: torch.Tensor,
                                   alpha, eps: float = 1e-12) -> torch.Tensor:
    """
    Generalized alpha-fair index per resource. `alpha` may be a scalar
    (same for every resource) or a (d,) sequence (one alpha per
    resource, as used elsewhere in this file for alphas_vec).
    Returns: (d,) tensor.
    """
    d = x_alloc.shape[1]
    if not hasattr(alpha, "__len__"):
        alpha = [alpha] * d
    out = torch.zeros(d, dtype=x_alloc.dtype, device=x_alloc.device)
    for k in range(d):
        mask = Y[:, k] > 0
        n_active = int(mask.sum().item())
        if n_active == 0:
            out[k] = float("nan")
            continue
        xk = x_alloc[mask, k]
        out[k] = alpha_fairness_index(xk, float(alpha[k]), dim=0, eps=eps)
    return out


def aggregate_fairness(per_resource_values: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Aggregate a (d,) per-resource fairness vector into a single scalar,
    weighted by the number of agents active on each resource.
    """
    weights = Y.sum(dim=0).clamp(min=0.0)
    valid = ~torch.isnan(per_resource_values)
    if valid.sum() == 0:
        return torch.tensor(float("nan"))
    w = weights[valid]
    v = per_resource_values[valid]
    if w.sum() <= 0:
        return v.mean()
    return (w * v).sum() / w.sum()


import torch

def compute_G_i_matrix(A_i, i, delta, epsilon, c, alpha, price, Y):
    """
    A_i: (d,) valuations for player i over resources
    alpha: (d,) alpha per resource (or scalar broadcast)
    Y: (n,d) mask (0/1) indicating which player uses which resource
    """

    n, d = Y.shape
    Yi = Y[i].to(dtype=torch.bool)
    active = Yi.nonzero(as_tuple=False).flatten()

    if active.numel() == 0:
        return 0.0

    # number of active resources per player
    d_per_player = Y.sum(dim=1)  # (n,)

    # per-player minimal total bid due to eps on active resources
    E = epsilon * d_per_player  # (n,)

    # per-player max bid on ONE active coordinate when budget is tight:
    # z_{j,max} = c - (d_j - 1)*epsilon
    zmax_per_player = c - (d_per_player - 1) * epsilon  # (n,)

    # per-resource smin^k and smax^k (others only)
    # smin^k: others bid epsilon if active on k
    smin_k = (epsilon * Y[:, :].sum(dim=0)) - (epsilon * Y[i, :]) + delta  # (d,)

    # smax^k: others put their whole remaining budget on k if they are active on k
    # i.e., player j contributes zmax_per_player[j] on k if Y[j,k]=1
    smax_k = (zmax_per_player[:, None] * Y).sum(dim=0) - (zmax_per_player[i] * Y[i]) + delta  # (d,)

    def grad_k(k, z, s):
        # matches your formula; assumes alpha is per resource
        return A_i[k] * s / (z + s) ** (2 - alpha[k]) * (z ** (-alpha[k])) - price

    # z bounds for player i on each active coordinate
    zmax_i = c - (active.numel() - 1) * epsilon

    # Evaluate gradient vector norm at “corners”
    vals = []
    for z in (epsilon, zmax_i):
        g_smin = torch.stack([grad_k(k, z, smin_k[k]) for k in active]).abs()
        g_smax = torch.stack([grad_k(k, z, smax_k[k]) for k in active]).abs()#.sum()
        vals.append(torch.max(g_smin))
        vals.append(torch.max(g_smax))
    G = torch.max(torch.stack(vals)).item()
    return G

import torch


def compute_G_k(a_i, delta, epsilon, c, n, alpha, beta, price, Y):
    """
    Compute a bound G_k for each resource k, then return max_k G_k.

    Parameters
    ----------
    a_i : float
        Player coefficient.
    delta : float
        Regularization parameter.
    epsilon : float
        Minimum bid.
    c : float
        Budget.
    n : int
        Number of players.
    alpha : list or 1D tensor of length d
        Alpha values per resource.
    beta : list or 1D tensor of length d
        Beta values per resource.
    price : float
        Unit price.
    Y : list or 1D tensor of length d
        Demand indicator/weight for this player.

    Returns
    -------
    G_max : torch.Tensor
        max_k G_k
    G_vec : torch.Tensor
        vector (G_1, ..., G_d)
    """
    alpha = torch.as_tensor(alpha, dtype=torch.float64)
    beta = torch.as_tensor(beta, dtype=torch.float64)
    Y = torch.as_tensor(Y, dtype=torch.float64)

    d = len(beta)

    smin = (n - 1) * epsilon + delta
    smax = (n - 1) * (c - (d - 1) * epsilon) + delta

    if smin <= 0:
        raise ValueError("smin must be strictly positive.")
    if c - (d - 1) * epsilon <= 0:
        raise ValueError("Need c > (d-1)*epsilon for feasible bids.")

    def grad_k(k, z, s):
        return Y[k] * (
            a_i * beta[k] * s / ((z + s) ** (2 - alpha[k])) * (z ** (-alpha[k]))
            - price
        )

    G_vec = torch.zeros(d, dtype=torch.float64)

    for k in range(d):
        # candidates at extreme points
        g1 = abs(grad_k(k, epsilon, smin))
        g2 = abs(grad_k(k, epsilon, smax))
        g3 = abs(grad_k(k, c - (d - 1) * epsilon, smin))
        g4 = abs(grad_k(k, c - (d - 1) * epsilon, smax))

        G_vec[k] = torch.max(torch.stack([g1, g2, g3, g4]))

    G_max = torch.max(G_vec)
    return  G_vec

def compute_G(a_i, delta, epsilon,c, n, alpha, beta, price, Y):
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
    d = len(beta)
    smin = (n - 1) * epsilon + delta
    smax = (n - 1) * (c - (d-1) * epsilon) + delta

    #smin = (n - 1) * epsilon + delta
    #smax = (n - 1) * c + delta


    def grad(k, z, s):
        return Y[k] * (a_i * beta[k] *  s / (z + s) ** (2 - alpha[k]) * z ** (-alpha[k]) - price)

    Norm = []
    for k in range(d):
        Norm.append(abs(grad(k,epsilon,smax)))
        for l in range(d-1):
            Norm[k]+=abs(grad(k,epsilon,smin))
    Norm2 = []
    for k in range(d):
        Norm2.append(abs(grad(k,epsilon,smin)))
        for l in range(d-1):
            Norm2[k]+=abs(grad(k,epsilon,smax))


    G = max(max(Norm), max(Norm2)) #abs(a_i*smax/(epsilon *(epsilon+smax)) - 1), abs(a_i*smin/(epsilon *(epsilon+smin)) - 1))
    #G = 0.10*sup_grad_components_phi(phi, epsilon, c, n_samples=5000)
    #G = max(abs(a_i*smax/(epsilon *(epsilon+smax)) - 1), abs(a_i*smin/(epsilon *(epsilon+smin)) - 1))
    return G

import torch
import numpy as np

class GameKelly:
    def __init__(self, n: int, d: int, beta, price,
                 epsilon, delta, alpha, tol, Y,
                 payoff_min=None, payoff_max=None):
        """
        alpha can be:
          - scalar (same alpha for all resources), or
          - tensor/list shape (d,) (different alpha per resource)
        price can be:
          - scalar, or
          - tensor/list shape (d,) (different price per resource)
        """
        self.n = n
        self.d = d
        self.beta = beta

        # normalize alpha -> (d,)
        #if not torch.is_tensor(alpha):
        #    alpha = torch.tensor(alpha, dtype=torch.float64)
        #if alpha.ndim == 0:
        #    alpha = alpha.repeat(d)
        self.alpha = alpha  # (d,)

        # normalize price -> (d,)
        #if not torch.is_tensor(price):
        #    price = torch.tensor(price, dtype=torch.float64)
        #if price.ndim == 0:
        #    price = price.repeat(d)
        self.price = price  # (d,)
        self.Y = Y

        # normalize epsilon -> (d,) or (n,d) is allowed externally
        if not torch.is_tensor(epsilon):
            epsilon = torch.tensor(epsilon, dtype=torch.float64)
        self.epsilon = epsilon

        self.delta = delta
        self.tol = tol
        self.payoff_min = payoff_min
        self.payoff_max = payoff_max

    # =========================
    # Utilities (DO NOT rename)
    # =========================
    def fraction_resource_k(self, z_k):
        return z_k / (torch.sum(z_k) + self.delta)

    def grad_phi(self, phi, z_vec):
        """
        z_vec: (n,) vector
        returns diag of Jacobian: (n,)
        """

        z = z_vec.clone().detach().requires_grad_(True)
        J = torch.autograd.functional.jacobian(phi, z)

        #noise_std = 0.1  # tune this
        #noise = noise_std * torch.randn_like(J.diag())
        #grad_noisy = grad + noise
        return J.diag() #+ noise

    # projected-gradient rest-point residual
    def proj_residual(self, Z_new, A_matrix, c_vector):
        Z_br = torch.zeros_like(Z_new)
        for i in range(self.n):
            p_i = Z_new.sum(dim=0) - Z_new[i, :] + self.delta
            Z_br[i, :] = solve_br_player_multi_resource(A_i=A_matrix[i, :],
                                                        p_i=p_i,
                                                        c_i=c_vector[i],
                                                        alpha=self.alpha,
                                                        price=self.price,
                                                        Y_i=self.Y[i, :],
                                                        epsilon=self.epsilon,
                                                        tol=1e-6,
                                                        )
        return Z_br

    def check_NE(self, t, A_matrix, c_vector, d_vector, eta,
            Z_prev, acc_grad, D, G, p, vary, alpha,
            Hybrid_funcs, Hybrid_sets, update):
        """
        Nash residual aggregated over resources:
          err = sum_r || BR_r(z[:,r]) - z[:,r] ||_2

        Notes:
        - fixes the bug `err += err` (doubling itself)
        - supports alpha[r] and (optionally) price per resource if self.price is (d,)
        - numerically robust: compares alpha as float
        """
        p = Z_prev.sum(dim=0, keepdim=True) - Z_prev + self.delta  # (n,d)
        eta_t = D / (G * np.sqrt(self.T))
        #eta_t = self.epsilon * c_vector / (
        #        A_matrix.amax(dim=1) * torch.sqrt(torch.tensor(self.T, device=A_matrix.device))
        #)
        is_hybrid = getattr(update, "__name__", "") == "Hybrid"
        if is_hybrid:
            Z_new, acc_grad = update(A_matrix, c_vector, d_vector, eta_t,
                Z_prev, acc_grad, self.Y,
                Hybrid_funcs=Hybrid_funcs, Hybrid_sets=Hybrid_sets,
            )
        else:
            Z_new, acc_grad = update(A_matrix, c_vector, d_vector, eta_t,
                Z_prev, acc_grad, self.Y
            )
        Z_br = self.proj_residual(Z_new, A_matrix, c_vector)
        # optional algorithmic step residual
        err_step = torch.norm(Z_new - Z_br, p=2)
        #err_step = torch.norm(Z_prev - Z_new, p=2)

        return Z_new, acc_grad, err_step#torch.norm(Z_prev - Z_new, p=2)

    def _phi_k(self, z_k, A_matrix_k, d_vector, k, Y):
        """
        z_k: (n,)
        returns vector payoff-gradient map (n,)
        """
        #print(self.alpha[k])
        alpha_k = self.alpha[k]#float(self.alpha[alpha_fair_values].item())
        #price_k = self.price[alpha_fair_values]

        x_k = self.fraction_resource_k(z_k)
        V_k = V_func(x_k, alpha_k)  # MUST accept scalar alpha_k
        return Y[:, k] * (A_matrix_k*V_k - self.price * z_k + d_vector) #self.Y[:,k] * (a_vector*self.beta[k]*V - self.price * z_k + d_vector)

    def best_response_k(self, z_k, p_k, a_vector, c_vector, d_vector, k):
        alpha_k =  self.alpha[k]#float(self.alpha[k].item())
        #price_k = self.price[k]

        if alpha_k not in [0, 1, 2]:
            return solve_nonlinear_eq(
                a_vector, p_k, alpha_k, self.epsilon, c_vector, self.price,
                max_iter=1000, tol=self.tol
            )
        return BR_alpha_fair(
            self.epsilon, c_vector, z_k, p_k,
            a_vector, self.delta, alpha_k, self.price, b=0
        )

    # ============================================================
    # Learning rules for d resources (KEEP NAMES, CHANGE SIGNATURE)
    # All rules now take bids Z:(n,d), p:(n,d), acc_grad:(n,d)
    # and return z_t:(n,d), acc_grad:(n,d).
    # ============================================================

    def Hybrid(self, A_matrix, c_vector, d_vector, eta_t, bids, acc_grad, Y,
               Hybrid_funcs=None, Hybrid_sets=None):
        """
        Mixed-population update for heterogeneous dynamics (e.g. a
        fraction of agents playing BR while the rest play DA/OGD, as
        in Fig. 5/6). Each subset of agents follows its own learning
        rule, but every rule is evaluated against the FULL current
        joint bid vector `bids` (never sliced before the call), so
        each agent's gradient / best response correctly sees the true
        aggregate bid of ALL competitors, regardless of which rule
        those competitors use. Only the rows owned by a given rule are
        then written into the merged result.

        Hybrid_funcs: list of method names, e.g. ["BR", "DA"].
        Hybrid_sets:  list of index lists, same length/order as
                      Hybrid_funcs, partitioning range(n).
        """
        Z_new = bids.clone()
        acc_new = acc_grad.clone()

        for idx_set, func_name in enumerate(Hybrid_funcs):
            func_ = getattr(self, func_name)
            I = Hybrid_sets[idx_set]  # player indices following this rule

            Z_candidate, acc_candidate = func_(
                A_matrix, c_vector, d_vector, eta_t, bids, acc_grad, Y
            )
            Z_new[I] = Z_candidate[I]
            acc_new[I] = acc_candidate[I]

        return Z_new, acc_new
#    def OGD(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, D, G, alpha,
#              p=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
    def OGD(self, A_matrix, c_vector, d_vector, eta_t, bids, acc_grad,Y):

        Z = bids

        z_candidate = torch.zeros_like(Z)
        for k in range(self.d):
            #def phi_k(z_k):
            #    return self._phi_k(z_k, A_matrix[:,k], d_vector,  k)

            s = Z[:, k].sum() -  Z[:, k] +  self.delta

            mask = Z[:, k] > 0
            s = s[mask]
            grad_k = torch.zeros_like(Z[:, k])
            grad_k[mask] = Y[:, k][mask] * (
                        A_matrix[:, k][mask] * s / (Z[:, k][mask] + s) ** (2 - self.alpha[k]) * Z[:, k][mask] ** (
                    -self.alpha[k]) - self.price)  # self.grad_phi(phi_k, Z[:, k])
            #self.grad_phi(phi_k, Z[:, k])
            z_candidate[:, k] = Z[:, k] + eta_t * grad_k
        Z_new = Q_simplex(z_candidate, self.epsilon, c_vector, Y)
        return Z_new, acc_grad

    def DA(self,  A_matrix, c_vector, d_vector, eta_t, bids, acc_grad, Y):

        Z = bids
        acc = acc_grad.clone()

        for k in range(self.d):
            #def phi_k(z_k):
            #    return self._phi_k(z_k, A_matrix[:, k], d_vector, k)

            s = Z[:, k].sum() - Z[:, k] + self.delta
            mask =  Z[:, k]>0
            s = s[mask]
            grad_k = torch.zeros_like(Z[:, k])
            grad_k[mask] = Y[:,k][mask] * (A_matrix[:, k][mask] *  s / ( Z[:, k][mask] + s) ** (2 - self.alpha[k]) *  Z[:, k][mask] ** (-self.alpha[k]) - self.price)#self.grad_phi(phi_k, Z[:, k])
            acc[:, k] = acc[:, k] + grad_k
        Z_candidate = torch.zeros_like(acc_grad)
        for k in range(self.d):
            Z_candidate[:,k] = acc[:,k] * eta_t
        Z_new = Q_simplex(Z_candidate, self.epsilon, c_vector, Y)

        return Z_new, acc


    def BR(self, A_matrix, c_vector, d_vector, eta_t, bids, acc_grad, Y,
           inner_iters: int = 200, inner_lr: float = 1e-3, inner_tol: float = 1e-8):
        """
        Best-response update: each agent i moves to (an inner-loop
        approximation of) its best response given every other agent's
        CURRENT bid held fixed.

        Fixes vs. the previous version:
        - Z_new used to start from `torch.empty_like(Z)` (uninitialized
          memory) instead of the agent's actual current bid.
        - the inner-loop gradient was evaluated at the static original
          bid `Z[i,:]` on every step instead of the evolving iterate,
          so it was not actually ascending toward a best response.
        - a fixed 1000-step inner loop with a print() on every step
          made this update rule too slow to use in practice; replaced
          with early stopping on the step size.
        - inactive resources (Y[i,k]==0) are masked out exactly like
          OGD/DA do, instead of relying on Z[i,:]>0 as a proxy.
        """
        Z = bids
        Z_new = Z.clone()
        S = torch.zeros_like(Z)
        for k in range(self.d):
            S[:, k] = Z[:, k].sum() - Z[:, k] + self.delta

        alpha_t = torch.as_tensor(self.alpha, dtype=Z.dtype, device=Z.device)

        for i in range(self.n):
            active = Y[i, :] > 0
            if not torch.any(active):
                continue

            s_i = S[i, :].clone()
            z_i = Z_new[i, :].clone()

            for _ in range(inner_iters):
                grad_i = torch.zeros_like(z_i)
                grad_i[active] = (
                    A_matrix[i, active] * s_i[active]
                    / (z_i[active] + s_i[active]) ** (2 - alpha_t[active])
                    * z_i[active] ** (-alpha_t[active])
                    - self.price
                )
                z_candidate = z_i + inner_lr * grad_i
                z_candidate = Q_simplex_i(z_candidate, self.epsilon * Y[i, :], c_vector[i])
                step = torch.norm(z_candidate - z_i)
                z_i = z_candidate
                if step < inner_tol:
                    break

            Z_new[i, :] = z_i

        return Z_new, acc_grad


    # ---------- learning loop (works for alpha per resource) ----------
    def learning(self, func, A_matrix, c_vector, d_vector, n_iter: int, eta, bids,
                 vary: bool = False, stop: bool = False, Hybrid_funcs=None, Hybrid_sets=None):

        update = getattr(self, func)
        self.T = 3000
        device, dtype = bids.device, bids.dtype

        matrix_bids = torch.zeros((n_iter + 1, self.n, self.d), device=device, dtype=dtype)
        matrix_alloc = matrix_bids.clone()
        matrix_bids[0] = bids.clone()


        vec_SW = torch.zeros(n_iter + 1, device=device, dtype=dtype)
        vec_LSW = torch.zeros(n_iter + 1, device=device, dtype=dtype)
        jain_idx = torch.zeros(n_iter + 1, device=device, dtype=dtype)
        alpha_fair_idx = torch.zeros(n_iter + 1, device=device, dtype=dtype)
        eps_error = torch.zeros(n_iter + 1, device=device, dtype=dtype)
        error_NE = torch.zeros(n_iter + 1, device=device, dtype=dtype)

        acc_grad = torch.zeros((self.n, self.d), device=device, dtype=dtype)

        # init
        Z0 = matrix_bids[0]
        k_last = n_iter

        G = torch.zeros_like(A_matrix[:,0])
        D = torch.zeros_like(A_matrix[:,0])
        for i in range(self.n):
            D[i] = (c_vector[i] - torch.sum(self.Y[i,:]) * self.epsilon)
            G[i] = compute_G_i_matrix(A_matrix[i,:], i,self.delta, self.epsilon, c_vector[i],  self.alpha, self.price, self.Y)
          #  if c_vector[i] == 50:
           #     G[i] = 0.56 * compute_G(a_vector[i], self.epsilon, c_vector[i], self.epsilon, self.n,self.alpha, self.beta, self.price, self.Y[i,:])
        print(f"G: {G}")
        p = Z0.sum(dim=0, keepdim=True) - Z0 + self.delta  # (n,d)
        print(f"eta: {D/(G*self.T**(0.5))}")

        Z_new, acc_grad,error_NE[0] = self.check_NE(0, A_matrix, c_vector, d_vector, eta,  Z0, acc_grad, D, G, p, vary, self.alpha,
                                        Hybrid_funcs, Hybrid_sets, update)
        x_tr = Z0 / (Z0.sum(dim=0, keepdim=True) + self.delta)
        vec_SW[0]  = Valuation_matrix(x_tr, A_matrix, d_vector, self.alpha, self.Y).sum()
        matrix_alloc[0] = x_tr
        jain_idx[0] = aggregate_fairness(per_resource_jain_index(x_tr, self.Y), self.Y)
        alpha_fair_idx[0] = aggregate_fairness(
            per_resource_alpha_fair_index(x_tr, self.Y, self.alpha), self.Y
        )
        for t in range(1, n_iter + 1):
            Z_prev = matrix_bids[t - 1]  # (n,d)
            p = Z_prev.sum(dim=0, keepdim=True) - Z_prev + self.delta  # (n,d)
            Z_new, acc_grad, error_NE[t] = self.check_NE(t, A_matrix, c_vector, d_vector, eta,  Z_prev, acc_grad, D, G, p, vary, self.alpha,
                                        Hybrid_funcs, Hybrid_sets, update)

            matrix_bids[t] = Z_new
            matrix_alloc[t] = Z_new / (Z_new.sum(dim=0, keepdim=True) + self.delta)
            vec_SW[t] = Valuation_matrix(matrix_alloc[t], A_matrix, d_vector, self.alpha, self.Y).sum()
            jain_idx[t] = aggregate_fairness(
                per_resource_jain_index(matrix_alloc[t], self.Y), self.Y
            )
            alpha_fair_idx[t] = aggregate_fairness(
                per_resource_alpha_fair_index(matrix_alloc[t], self.Y, self.alpha), self.Y
            )
            if stop and error_NE[t] <= self.tol:
                k_last = t
                break

        T_used = k_last
        traj = matrix_bids[:T_used]  # (T,n,d)

        col = torch.arange(1, T_used + 1, device=device, dtype=dtype).view(-1, 1, 1)
        agg_bids = torch.cumsum(traj, dim=0) / col

        Bids = [traj, agg_bids, jain_idx[:T_used], alpha_fair_idx[:T_used]]
        Welfare = [vec_SW[:T_used], vec_LSW[:T_used]]
        Utility_set = [eps_error[:T_used], eps_error[:T_used]]

        return Bids, Welfare, Utility_set, torch.maximum(
            error_NE[:T_used], torch.tensor(self.tol, device=device, dtype=dtype)
        )


import torch


def vprime_vsecond(x: torch.Tensor, utility: str):
    """
    Returns V'(x), V''(x) for:
      - 'log'    : V(x)=log(x)
      - 'linear' : V(x)=x
    """
    if utility == "log":
        vp = 1.0 / x
        vpp = -1.0 / (x ** 2)
    elif utility == "linear":
        vp = torch.ones_like(x)
        vpp = torch.zeros_like(x)
    else:
        raise ValueError("utility must be 'log' or 'linear'")
    return vp, vpp


def pseudo_hessian_block_torch(
    z_k: torch.Tensor,      # shape (n,)
    a_k: torch.Tensor,      # shape (n,)
    r: torch.Tensor,        # shape (n,)
    utility: str = "log",
    delta_k: float = 0.0,
):
    """
    Weighted pseudo-Hessian block H_r^(k) for one resource k.
    """
    if torch.any(z_k <= 0):
        raise ValueError("All bids must be strictly positive.")
    if torch.any(a_k <= 0):
        raise ValueError("All entries of a_k must be positive.")
    if torch.any(r <= 0):
        raise ValueError("All entries of r must be positive.")

    S = z_k.sum() + delta_k
    x = z_k / S

    vp, vpp = vprime_vsecond(x, utility)

    # Your formulas:
    # f_i = (1-x_i)^2 V''(x_i) - 2(1-x_i) V'(x_i)
    # g_i = -x_i(1-x_i) V''(x_i) + (2x_i-1)V'(x_i)
    f = (1.0 - x) ** 2 * vpp - 2.0 * (1.0 - x) * vp
    g = -x * (1.0 - x) * vpp + (2.0 * x - 1.0) * vp

    # Multiply by a_i^(k), divide by S^2
    f = a_k * f / (S ** 2)
    g = a_k * g / (S ** 2)

    n = z_k.numel()
    Hk = torch.zeros((n, n), dtype=z_k.dtype, device=z_k.device)

    # diagonal
    Hk[torch.arange(n), torch.arange(n)] = 2.0 * r * f

    # off-diagonal
    for i in range(n):
        for j in range(i + 1, n):
            hij = r[i] * g[i] + r[j] * g[j]
            Hk[i, j] = hij
            Hk[j, i] = hij

    return Hk


def full_pseudo_hessian_torch(
    z: torch.Tensor,            # shape (n,d)
    A_matrix: torch.Tensor,     # shape (n,d)
    r: torch.Tensor = None,     # shape (n,)
    utility: str = "log",
    delta=0.0,                  # scalar or shape (d,)
):
    """
    Full block-diagonal weighted pseudo-Hessian.
    """
    n, d = z.shape
    if A_matrix.shape != (n, d):
        raise ValueError("A_matrix must have shape (n,d)")

    if r is None:
        r = torch.ones(n, dtype=z.dtype, device=z.device)

    if isinstance(delta, (int, float)):
        delta = torch.full((d,), float(delta), dtype=z.dtype, device=z.device)
    else:
        delta = delta.to(dtype=z.dtype, device=z.device)
        if delta.shape != (d,):
            raise ValueError("delta must be scalar or shape (d,)")

    H = torch.zeros((n * d, n * d), dtype=z.dtype, device=z.device)

    for k in range(d):
        Hk = pseudo_hessian_block_torch(
            z_k=z[:, k],
            a_k=A_matrix[:, k],
            r=r,
            utility=utility,
            delta_k=float(delta[k].item()),
        )
        sl = slice(k * n, (k + 1) * n)
        H[sl, sl] = Hk

    return H


def is_negative_definite_torch(H: torch.Tensor, tol: float = 1e-10):
    """
    Returns (is_negative_definite, max_eigenvalue)
    """
    eigvals = torch.linalg.eigvalsh(H)
    maxeig = eigvals.max().item()
    return maxeig < -tol, maxeig


def sample_action_space_torch(
    n: int,
    d: int,
    budgets,                       # scalar or tensor (n,)
    eps: float = 1e-2,
    n_samples: int = 100,
    demand_mask: torch.Tensor = None,   # shape (n,d), bool
    device: str = "cpu",
    dtype=torch.float64,
):
    """
    Samples z in the action space:
      z_i^(k) >= eps on demanded resources,
      z_i^(k) = 0 otherwise,
      sum_k z_i^(k) <= c_i
    """
    if isinstance(budgets, (int, float)):
        budgets = torch.full((n,), float(budgets), dtype=dtype, device=device)
    else:
        budgets = budgets.to(dtype=dtype, device=device)

    if demand_mask is None:
        demand_mask = torch.ones((n, d), dtype=torch.bool, device=device)

    samples = []

    for _ in range(n_samples):
        z = torch.zeros((n, d), dtype=dtype, device=device)

        for i in range(n):
            active = torch.where(demand_mask[i])[0]
            m = active.numel()
            if m == 0:
                continue

            min_required = m * eps
            if budgets[i].item() < min_required:
                raise ValueError(f"Budget too small for player {i}")

            remaining = budgets[i] - min_required
            extra_total = torch.rand(1, dtype=dtype, device=device).item() * remaining.item()

            # Dirichlet via Gamma
            gamma = torch.distributions.Gamma(
                torch.ones(m, dtype=dtype, device=device),
                torch.ones(m, dtype=dtype, device=device)
            ).sample()
            w = gamma / gamma.sum()

            z[i, active] = eps + extra_total * w

        samples.append(z)

    return samples


def test_negative_definiteness_torch(
    A_matrix: torch.Tensor,
    utility: str = "log",
    budgets=100.0,
    eps: float = 1e-2,
    r: torch.Tensor = None,
    delta=0.0,
    n_samples: int = 200,
    demand_mask: torch.Tensor = None,
    tol: float = 1e-10,
    device: str = "cpu",
    dtype=torch.float64,
):
    """
    Tests ND on random samples of the action space.
    Returns a dict with worst eigenvalues and a violating sample if found.
    """
    A_matrix = A_matrix.to(dtype=dtype, device=device)
    n, d = A_matrix.shape

    if r is None:
        r = torch.ones(n, dtype=dtype, device=device)
    else:
        r = r.to(dtype=dtype, device=device)

    if isinstance(delta, (int, float)):
        delta = torch.full((d,), float(delta), dtype=dtype, device=device)
    else:
        delta = delta.to(dtype=dtype, device=device)

    samples = sample_action_space_torch(
        n=n,
        d=d,
        budgets=budgets,
        eps=eps,
        n_samples=n_samples,
        demand_mask=demand_mask,
        device=device,
        dtype=dtype,
    )

    all_nd = True
    worst_full = -1e18
    worst_block = -1e18
    bad_sample = None
    bad_resource = None
    bad_block = None

    for idx, z in enumerate(samples):
        H = full_pseudo_hessian_torch(z, A_matrix, r=r, utility=utility, delta=delta)
        ok_full, maxeig_full = is_negative_definite_torch(H, tol=tol)
        worst_full = max(worst_full, maxeig_full)

        for k in range(d):
            Hk = pseudo_hessian_block_torch(
                z[:, k], A_matrix[:, k], r=r, utility=utility, delta_k=float(delta[k].item())
            )
            ok_block, maxeig_block = is_negative_definite_torch(Hk, tol=tol)
            worst_block = max(worst_block, maxeig_block)

            if not ok_block and bad_sample is None:
                all_nd = False
                bad_sample = z.detach().cpu()
                bad_resource = k
                bad_block = Hk.detach().cpu()

        if not ok_full:
            all_nd = False
            if bad_sample is None:
                bad_sample = z.detach().cpu()

    return {
        "all_negative_definite": all_nd,
        "worst_max_eig_full": worst_full,
        "worst_max_eig_block": worst_block,
        "violating_sample": bad_sample,
        "violating_resource": bad_resource,
        "violating_block": bad_block,
    }


# -------------------------
# Helpers
# -------------------------
import os
LINESTYLES = [
    "-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))
]

COLORS = [
    "tab:blue", "tab:orange", "tab:green",
    "tab:red", "tab:purple", "tab:brown",
    "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
]

def build_curve_styles(curve_keys):
    styles = {}
    for i, key in enumerate(curve_keys):
        styles[key] = dict(
            color=COLORS[i % len(COLORS)],
            linestyle=LINESTYLES[(i // len(COLORS)) % len(LINESTYLES)],
        )
    return styles

def _auto_y_limits(y_slices, ylog=False, q=(0.02, 0.98), margin=0.20, eps=1e-12):

    yy = np.concatenate([np.asarray(v).ravel() for v in y_slices if len(v) > 0])
    yy = yy[np.isfinite(yy)]
    if yy.size == 0:
        return (eps if ylog else 0.0, 1.0)

    if ylog:
        yy = yy[yy > 0]
        if yy.size == 0:
            return (eps, 1.0)
        lo = np.quantile(yy, q[0])
        hi = np.quantile(yy, q[1])
        lo = max(lo, eps)
        if hi <= lo:
            hi = lo * 10.0
        return (max(lo / (1.0 + margin), eps), hi * (1.0 + margin))

    lo = np.quantile(yy, q[0])
    hi = np.quantile(yy, q[1])
    if hi <= lo:
        hi = lo + max(abs(lo), eps)
    dy = hi - lo
    y1 = max(0.0, lo - margin * dy)
    y2 = hi + margin * dy
    return y1, y2

def add_zoom_inset(ax, it, curves, *, cfg, colors, styles, ylog=False, pct_xmax=None):
    """
    curves: dict name -> y array
    colors: list or dict of colors (indexed consistently with curves you plot)
    styles: dict name -> (linestyle, marker)
    """
    if not cfg.get("Add_Zoom", False):
        return

    # ---- zoom window (indices) ----
    x_idx_min, x_idx_max = cfg.get("x_zoom_interval", (0, min(len(it), 200)))
    x_idx_min = max(0, int(x_idx_min))
    x_idx_max = min(int(x_idx_max), len(it))
    assert x_idx_max > x_idx_min, "Invalid zoom interval"

    # ---- y bounds ----
    y_min, y_max = cfg.get("y_zoom_interval", (None, None))
    use_auto = (y_min is None) or (y_max is None)

    # ---- inset position ----
    inset_rect = cfg.get("inset_rect", [0.58, 0.52, 0.40, 0.42])
    axins = ax.inset_axes(inset_rect)

    y_slices = []
    k = 0  # color index if colors is a list
    curves_res = {}
    curves_eff = {}
    colors_6 = {}
    styles_6 = {}
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # ---- Styles: algo colors fixed; gamma markers fixed ----
    algo_colors = {"OGD": "tab:blue", "DA": "tab:orange"}
    algo_styles = {"OGD": ("-", None), "DA": ("--", None)}
    i = 0
    def key(algo, g): return f"{algo},γ={g}"
    gamma_markers_cycle = ["s", "o","^", "D", "v", "P", "X"]
    gamma_markers = {g: gamma_markers_cycle[i % len(gamma_markers_cycle)] for i, g in enumerate(gammas)}

    for g in gammas:
        for algo in ["OGD", "DA"]:
            k = key(algo, g)
            colors_6[k] = algo_colors[algo]
            styles_6[k] = (algo_styles[algo][0], gamma_markers[g])  # marker encodes gamma
            i =+1
    # ---- Resource colors (panel 3) ----
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markersize =30

    # ---- User markers (panel 3) ----
    user_markers_cycle = ["s","o", "^", "D", "X","*" ,"h" , "v", "+"]
    # ---- plot selected curves in inset ----
    k  =0
    for name, y in curves.items():
        print(name)
        if name in ["OGD,γ=10", "DA,γ=10", "OGD,γ=50", "DA,γ=50", "OGD,γ=100", "DA,γ=100"]:   # <-- your selection rule
            y = np.asarray(y, float)
            L = min(len(it), len(y))
            y = y[:L]

            j2 = min(x_idx_max, L)
            xz = it[x_idx_min:j2]
            yz = y[x_idx_min:j2]
            y_slices.append(yz)

            step = 10
            axins.plot(
                xz[::step], yz[::step],
                linewidth=2,
                linestyle="-",
                color=colors_6[name],
                marker=user_markers_cycle[k],
                markersize=markersize,
                markerfacecolor=colors_6[name],  # fill color
                markeredgecolor="black",  # 🔥 black contour
                markeredgewidth=1.0,  # thickness of contour
            )
            k += 1

    # ---- x limits ----
    axins.set_xlim(it[x_idx_min], it[x_idx_max - 1])

    # ---- y limits ----
    if use_auto:
        y1, y2 = _auto_y_limits(
            y_slices,
            ylog=ylog,
            q=cfg.get("zoom_quantiles", (0.02, 0.98)),
            margin=cfg.get("zoom_margin", 0.20),
            eps=cfg.get("zoom_log_eps", 1e-12),
        )
    else:
        y1, y2 = y_min, y_max

    if ylog:
        axins.set_yscale("log")
        y1 = max(y1, cfg.get("zoom_log_eps", 1e-12))

    axins.set_ylim(y1, y2)



    # ---- percent formatter if needed ----
    if pct_xmax is not None:
        axins.yaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=pct_xmax, decimals=2)
        )

    # ---- styling ----

    axins.grid(alpha=0.25)
    axins.tick_params(axis="both", labelsize=30, length=0)
    axins.set_xticklabels([])
    for lab in axins.get_yticklabels():
        lab.set_fontweight("bold")


    # ---- draw rectangle + connectors ----
    mark_inset(
        ax, axins,
        loc1=cfg.get("loc1", 2),
        loc2=cfg.get("loc2", 4),
        fc="none",
        ec="black",
        lw=cfg.get("zoom_rect_linewidth", 1),
    )


def _auto_y_limits_percent(y_slices, *, q=(0.02, 0.98), margin=0.15, eps=1e-12):
    """
    y_slices: list of arrays (already sliced to zoom window)
    Returns (y1,y2) in same units as y (NOT multiplied by 100).
    Uses quantiles to avoid spikes + adds margin.
    """
    yy = np.concatenate([np.asarray(v).ravel() for v in y_slices if len(v) > 0])
    yy = yy[np.isfinite(yy)]
    if yy.size == 0:
        return 0.0, 1.0

    lo = np.quantile(yy, q[0])
    hi = np.quantile(yy, q[1])
    if hi <= lo:
        hi = lo + max(abs(lo), eps)

    dy = hi - lo
    y1 = lo - margin * dy
    y2 = hi + margin * dy
    # keep nonnegative for a loss metric
    y1 = max(0.0, y1)
    return y1, y2

def save_legend_only(path, handles, labels, ncol=3, figsize=(10, 1.6), fontsize=11):
    fig = plt.figure(figsize=figsize)
    fig.legend(handles, labels, loc="center", ncol=ncol, frameon=True,
               prop={"weight": "bold"}, fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def save_legend_only2(path, handles, labels, ncol=3, figsize=(12, 2.0), fontsize=40):
    fig = plt.figure(figsize=figsize)

    legend = fig.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=True,
        prop={"size": fontsize, "weight": "bold"},
        handlelength=2.5,
        handletextpad=0.8,
        columnspacing=1.2,
        labelspacing=0.6,
    )

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)



# -------------------------
# Main plotting function
# -------------------------
def plot_metric_and_bids_vs_gamma(
    case_name,
    results_case,          # dict with keys: residuals, effloss, z_star
    gammas,                # list like [0,20,40]
    D_mask,                # (n,d) bool
    resource_names=None,   # list length d
    user_names=None,       # list length n
    out_dir="figures",
    file_prefix=None,
    # truncation (different cuts per panel)
    cut_panel1=100,
    cut_panel2=200,
    # zoom configs per panel (can differ)
    zoom1=None,
    zoom2=None,
    # saving z*
    save_z=True,
):
    os.makedirs(out_dir, exist_ok=True)
    if file_prefix is None:
        file_prefix = case_name.replace(" ", "_").replace("/", "_")

    # infer sizes
    z0 = results_case[gammas[0]]["z_star"]
    n, d = z0.shape

    D_mask = np.asarray(D_mask, bool)
    assert D_mask.shape == (n, d)

    if resource_names is None:
        resource_names = [str(k+1) for k in range(d)]
    if user_names is None:
        user_names = [str(i+1) for i in range(n)]

    # ---- Styles: algo colors fixed; gamma markers fixed ----
    algo_colors = {"OGD": "tab:blue", "DA": "tab:orange", "BR": "tab:purple"}
    algo_styles = {"OGD": ("-", None), "DA": ("--", None)}

    gamma_markers_cycle = ["s", "o","^", "D", "v", "P", "X"]
    gamma_markers = {g: gamma_markers_cycle[i % len(gamma_markers_cycle)] for i, g in enumerate(gammas)}

    # ---- Resource colors (panel 3) ----
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    resource_colors = [cycle[(k + 2) % len(cycle)] for k in range(d)]

    # ---- User markers (panel 3) ----
    user_markers_cycle = ["s","o", "^", "X","D", "*" ,"h" , "v", "+"]
    user_markers = [user_markers_cycle[i % len(user_markers_cycle)] for i in range(n)]

    # ---- Panel 1 + 2 curves: build 6 series (algo x gamma) ----
    def key(algo, g): return f"{algo},γ={g}"

    curves_res = {}
    curves_eff = {}
    colors_6 = {}
    styles_6 = {}
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    resource_colors = [cycle[(k + 2) % len(cycle)] for k in range(d*2)]
    i = 0
    for g in gammas:
        for algo in ["OGD", "DA"]:
            k = key(algo, g)
            curves_res[k] = np.asarray(results_case[g]["residuals"][algo], float)
            curves_eff[k] = np.asarray(results_case[g]["eff_loss"][algo], float)
            colors_6[k] = algo_colors[algo]
            styles_6[k] = (algo_styles[algo][0], gamma_markers[g])  # marker encodes gamma
            i =+1

    # ---- Figure ----
    figsize = (20, 12)
    fontsize = 40
    markersize = 30
    plt.rcParams.update({
        "font.size": 40,
        "axes.titlesize": 14,
        "axes.labelsize": 2 * fontsize,
        "xtick.labelsize":  1.25*fontsize,
        "ytick.labelsize":  1.25 * fontsize,
    })

    curve_keys = [
        "OGD, γ=0", "DA, γ=0",
        "OGD, γ=10", "DA, γ=10",
        "OGD, γ=50", "DA, γ=50",
    ]
    curve_styles = build_curve_styles(curve_keys)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    linewidth = 12
    # =======================
    # Panel 1: Residual
    # =======================
    ax = axes
    T1 = min(cut_panel1, max(len(v) for v in curves_res.values()))
    it1 = np.arange(1, T1 + 1)

    clr = 0

    for gamma in gammas:
        for algo in ["OGD", "DA"]:
            #key = f"{algo}, γ={gamma}"
            y = np.asarray(results_case[gamma]["residuals"][algo], float)
            L = min(T1, len(y))
            k = key(algo, gamma)
            step = 400
            ax.plot(
                it1[:L][::step], y[:L][::step],
                linewidth=2,
                linestyle="-",
                color=colors_6[k],
                marker=user_markers_cycle[clr],
                markersize=markersize,
                markerfacecolor=colors_6[k],  # fill color
                markeredgecolor="black",  # 🔥 black contour
                markeredgewidth=1.0,  # thickness of contour
            )

            clr = clr + 1

    ax.set_xlabel(r"Time step $(t)$", fontweight="bold")
    ax.set_ylabel("Residual", fontweight="bold")
    ax.set_yscale("log")

    # --- Scientific notation, shown once ---
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # always scientific
    formatter.set_useOffset(True)  # show ×10^k only once

    ax.xaxis.set_major_formatter(formatter)

    # --- Force integer ticks (no decimals) ---
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax.grid(alpha=0.3)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    if zoom1 is None:
        zoom1 = dict(Add_Zoom=False)
    # ============================
    # Zoom inset for Panel 2
    # ============================

    add_zoom_inset(ax, it1, curves={k: v[:T1] for k, v in curves_res.items()},
                   cfg=zoom1, colors=colors_6, styles="--", ylog=True)


    # =======================
    # Save figure
    # =======================
    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"{file_prefix}_residual.pdf")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


    # =======================
    # Save ONE global legend (separate file)
    # =======================
    # Build legend handles: algorithms (line), gammas (marker), resources (color), users (marker)
    handles, labels = [], []



    # gammas (markers)
    clr = 0
    for g in gammas:

        for algo in ["OGD", "DA"]:
            k = key(algo, g)
            h = plt.Line2D([0], [0], color=colors_6[k], linestyle="-", linewidth=2,
                           marker=user_markers_cycle[clr], markersize=markersize,
                           markerfacecolor=colors_6[k], markeredgecolor="black",markeredgewidth=1.0)
            handles.append(h);
            labels.append(fr"{algo}-{g}")
            clr += 1
            # algo

    legend_path = os.path.join(out_dir, f"{file_prefix}_legend.pdf")
    save_legend_only(legend_path, handles, labels, ncol=len(gammas)*2, figsize=(12, 2.0), fontsize=11)

    # =======================
    # Save z_star values
    # =======================
    if save_z:
        # store z_star[gamma] stacked into (G,n,d) + gamma list
        Z = np.stack([np.asarray(results_case[g]["z_star"], float) for g in gammas], axis=0)
        z_path = os.path.join(out_dir, f"{file_prefix}_zstar.npz")
        np.savez(z_path, gammas=np.asarray(gammas), z_star=Z)

    print(f"[{case_name}] saved:")
    print("  figure :", fig_path)
    print("  legend :", legend_path)
    if save_z:
        print("  z_star :", z_path)



import numpy as np
import matplotlib.pyplot as plt

def plot_user_budget_stacked_by_resource(
    z,                          # (n,d) bids/spending
    c_i,                        # (n,) budgets (scalar ok)
    D_mask=None,                # (n,d) bool; non-demanded => 0
    user_names=None,            # list length n
    resource_names=None,        # list length d (for legend)
    resource_colors=None,       # list length d
    show_unused=True,
    unused_color="white",
    edgecolor="black",
    linewidth=1.2,
    ax=None,
    title=None,
    ylim_pad=0.05,
):
    z = np.asarray(z, float)
    n, d = z.shape

    c_i = np.asarray(c_i, float)
    if c_i.ndim == 0:
        c_i = np.full(n, float(c_i))
    assert c_i.shape == (n,)

    if D_mask is None:
        D_mask = np.ones((n, d), dtype=bool)
    else:
        D_mask = np.asarray(D_mask, bool)
        assert D_mask.shape == (n, d)

    # enforce non-demanded = 0
    z_plot = np.where(D_mask, z, 0.0)

    if user_names is None:
        user_names = [f"User {i+1}" for i in range(n)]
    if resource_names is None:
        resource_names = [f"{k+1}" for k in range(d)]

    # resource colors (consistent mapping)
    if resource_colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        resource_colors = [cycle[(k + 2) % len(cycle)] for k in range(d)]
    else:
        assert len(resource_colors) == d

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.8))

    x = np.arange(n)
    bottom = np.zeros(n)

    handles = []
    labels = []

    # stacked bars: each layer is a resource k
    for k in range(d):
        h = ax.bar(
            x,
            z_plot[:, k],
            bottom=bottom,
            color=resource_colors[k],
            edgecolor=edgecolor,
            linewidth=linewidth,
            label=fr"Resource $k={resource_names[k]}$",
        )
        bottom += z_plot[:, k]
        handles.append(h[0])
        labels.append(fr"Resource $k={resource_names[k]}$")

    # unused budget on top
    if show_unused:
        unused = np.maximum(0.0, c_i - bottom)
        if np.any(unused > 1e-12):
            h = ax.bar(
                x,
                unused,
                bottom=bottom,
                color=unused_color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                label="Unused",
            )
            handles.append(h[0])
            labels.append("Unused")

    # axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(user_names, fontweight="bold")
    ax.set_ylabel(r"Budget spent $z_i=\sum_k z_i^{(k)}$", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    if title:
        ax.set_title(title, fontweight="bold")

    ymax = max(np.max(c_i), np.max(bottom))
    ax.set_ylim(0, (1.0 + ylim_pad) * ymax)

    for t in ax.get_yticklabels():
        t.set_fontweight("bold")

    return ax, handles, labels


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


def plot_player_bids_resourcewise(
    bids_by_algo,
    save_dir,
    player_labels=None,
    resource_labels=None,
    algo_colors=None,
    step=1,
    cut=None,
    fontsize=26,
    figsize_per_panel=(7.5, 6.0),
    markersize=10,
    linewidth=3,
    logy=False,
    sharey=False,
    show_final_star=False,
    zstars_by_algo=None,
    figure_format="pdf",
):
    """
    Plot bids resource-wise:
      - one figure per algorithm
      - one subplot per resource
      - one curve per player

    Parameters
    ----------
    bids_by_algo : dict
        Example:
            bids_by_algo["OGD"] = array of shape (T, n, d) or (n, T, d)

    save_dir : str
        Directory where figures are saved.

    player_labels : list[str] or None
        Labels for players. Default: ["Player 1", ..., "Player n"]

    resource_labels : list[str] or None
        Labels for resources. Default: ["Resource 1", ..., "Resource d"]

    algo_colors : dict or None
        Optional map algo -> color, only used for equilibrium markers or titles if desired.
        Player colors are chosen automatically.

    step : int
        Plot every `step` points.

    cut : int or None
        Maximum number of iterations to plot.

    fontsize : int
        Base fontsize.

    figsize_per_panel : tuple
        Size per subplot panel.

    markersize : int
        Marker size if markers are used.

    linewidth : int or float
        Line width.

    logy : bool
        Whether to use log-scale on y-axis.

    sharey : bool
        Whether subplots share the same y-axis.

    show_final_star : bool
        Whether to show equilibrium/final bid as horizontal dashed line.

    zstars_by_algo : dict or None
        Optional final bids per algo, shape (n, d), used if show_final_star=True.

    figure_format : str
        "pdf", "png", etc.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Default player styles
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    player_markers = ["o", "s", "^", "D", "X", "v", "P", "*", "<", ">"]

    for algo, Z in bids_by_algo.items():
        Z = np.asarray(Z, dtype=float)

        # Accept (T, n, d) or (n, T, d)
        if Z.ndim != 3:
            raise ValueError(f"{algo}: expected 3D array, got shape {Z.shape}")

        # Heuristic: if first dim is small and second dim is large, probably (n, T, d)
        if Z.shape[0] < Z.shape[1]:
            # assume (n, T, d) -> convert to (T, n, d)
            Z = np.transpose(Z, (1, 0, 2))

        T, n, d = Z.shape

        if cut is not None:
            T = min(T, cut)
            Z = Z[:T]

        it = np.arange(1, T + 1)

        if player_labels is None:
            this_player_labels = [fr"Player {i+1}" for i in range(n)]
        else:
            this_player_labels = player_labels

        if resource_labels is None:
            this_resource_labels = [fr"Resource {k+1}" for k in range(d)]
        else:
            this_resource_labels = resource_labels

        fig_w = figsize_per_panel[0] * d
        fig_h = figsize_per_panel[1]
        fig, axes = plt.subplots(1, d, figsize=(fig_w, fig_h), sharey=sharey)

        if d == 1:
            axes = [axes]

        for k, ax in enumerate(axes):
            for i in range(n):
                color = cycle[i % len(cycle)]
                marker = player_markers[i % len(player_markers)]

                ax.plot(
                    it[::step],
                    Z[:T:step, i, k],
                    linewidth=linewidth,
                    color=color,
                    marker=marker if len(it[::step]) <= 40 else None,
                    markersize=markersize,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    label=this_player_labels[i],
                )

                if show_final_star and zstars_by_algo is not None and algo in zstars_by_algo:
                    zstar = np.asarray(zstars_by_algo[algo], dtype=float)
                    if zstar.shape == (n, d):
                        ax.axhline(
                            y=zstar[i, k],
                            linestyle="--",
                            linewidth=2,
                            color=color,
                            alpha=0.8,
                        )

            ax.set_title(this_resource_labels[k], fontsize=fontsize, fontweight="bold")
            ax.set_xlabel(r"Time step $(t)$", fontsize=fontsize, fontweight="bold")
            ax.grid(alpha=0.3)

            if logy:
                ax.set_yscale("log")

            ax.tick_params(
                axis="both",
                which="major",
                labelsize=int(0.9 * fontsize),
                width=1.5,
                length=6
            )

            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")

        axes[0].set_ylabel(r"Bid $z_i^{(k)}(t)$", fontsize=fontsize, fontweight="bold")

        # Global legend
        handles = []
        for i in range(n):
            color = cycle[i % len(cycle)]
            marker = player_markers[i % len(player_markers)]
            handles.append(
                Line2D(
                    [0], [0],
                    color=color,
                    marker=marker,
                    linewidth=linewidth,
                    markersize=markersize,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    label=this_player_labels[i]
                )
            )

        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=min(n, 5),
            frameon=True,
            prop={"size": max(12, fontsize - 4), "weight": "bold"}
        )

        fig.suptitle(
            fr"Player bids resource-wise ({algo})",
            fontsize=int(1.15 * fontsize),
            fontweight="bold",
            y=1.15
        )

        fig.tight_layout()
        save_path = os.path.join(save_dir, f"bids_resourcewise_{algo}.{figure_format}")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {save_path}")

import numpy as np

def aggregate_trajectories(trajs, T=None, mode="mean", q_low=0.1, q_high=0.9):
    """
    trajs: list of 1D arrays
    Returns:
        x      : time indices
        center : mean or median trajectory
        lower  : lower quantile
        upper  : upper quantile
    """
    if len(trajs) == 0:
        raise ValueError("Empty trajectory list.")

    lengths = [len(y) for y in trajs]
    if T is None:
        T = min(lengths)   # safest choice: keep common prefix
    else:
        T = min(T, min(lengths))

    A = np.stack([np.asarray(y[:T], dtype=float) for y in trajs], axis=0)  # shape (M, T)

    if mode == "mean":
        center = A.mean(axis=0)
    elif mode == "median":
        center = np.median(A, axis=0)
    else:
        raise ValueError("mode must be 'mean' or 'median'")

    lower = np.quantile(A, q_low, axis=0)
    upper = np.quantile(A, q_high, axis=0)
    x = np.arange(1, T + 1)

    return x, center, lower, upper

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

def plot_residuals_with_noise_band(
    results_by_d,
    d_values,
    c_values,
    algos,
    algo_colors,
    save_path,
    step=1,
    cut=None,
    fontsize=40,
    linewidth=4,
    markersize=20,
    c_markers_cycle=None,
    use_logy=True,
    band_alpha=0.18,
    center_mode="mean",     # "mean" or "median"
    q_low=0.1,
    q_high=0.9,
):
    if c_markers_cycle is None:
        c_markers_cycle = ["s", "o", "^", "X", "D", "*", "h", "v", "+"]

    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    # Find common T
    max_len = 0
    for d in d_values:
        for c in c_values:
            for algo in algos:
                trajs = results_by_d[d][c]["residuals"][algo]
                if len(trajs) > 0:
                    max_len = max(max_len, min(len(y) for y in trajs))

    T = max_len if cut is None else min(cut, max_len)

    l = 0
    for d in d_values:
        for c in c_values:
            for algo in algos:
                trajs = results_by_d[d][c]["residuals"][algo]
                if len(trajs) == 0:
                    continue

                x, center, lower, upper = aggregate_trajectories(
                    trajs, T=T, mode=center_mode, q_low=q_low, q_high=q_high
                )

                color = algo_colors[algo]
                marker = c_markers_cycle[l % len(c_markers_cycle)]

                # Important for log scale
                if use_logy:
                    eps = 1e-16
                    center = np.maximum(center, eps)
                    lower = np.maximum(lower, eps)
                    upper = np.maximum(upper, eps)

                # Noise band
                ax.fill_between(
                    x[::step],
                    lower[::step],
                    upper[::step],
                    color=color,
                    alpha=band_alpha,
                    linewidth=0
                )

                # Central curve
                ax.plot(
                    x[::step],
                    center[::step],
                    linestyle="--",
                    linewidth=linewidth,
                    color=color,
                    marker=marker,
                    markersize=markersize,
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                )

            l += 1

    # Legend for algorithms
    legend_handles = [
        Line2D(
            [0], [0],
            color=algo_colors[algo],
            linewidth=10,
            linestyle="--",
            label=algo
        )
        for algo in algos
    ]

    ax.legend(
        handles=legend_handles,
        frameon=True,
        loc="best",
        prop={"size": 40, "weight": "bold"}
    )

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(True)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_fontsize(40)
    ax.xaxis.get_offset_text().set_fontweight("bold")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.set_xlabel(r"Time step $(t)$", fontsize=2 * fontsize, fontweight="bold")
    ax.set_ylabel("Residual", fontsize=2 * fontsize, fontweight="bold")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=int(1.25 * fontsize),
        width=2,
        length=8
    )

    if use_logy:
        ax.set_yscale("log")

    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")



import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


def _prepare_curve(y):
    """
    Convert possible formats into a 1D numpy array.

    Accepted:
    - 1D array
    - list containing one 1D array
    - list of several 1D arrays -> returns median trajectory on common horizon
    """
    if isinstance(y, list):
        if len(y) == 0:
            return np.array([], dtype=float)

        # list of runs
        if np.ndim(y[0]) >= 1:
            lengths = [len(np.asarray(v).ravel()) for v in y]
            T = min(lengths)
            A = np.stack([np.asarray(v, dtype=float).ravel()[:T] for v in y], axis=0)
            return np.median(A, axis=0)

        return np.asarray(y, dtype=float).ravel()

    return np.asarray(y, dtype=float).ravel()


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


def _prepare_curve(y):
    """
    Convert possible formats into a 1D numpy array.

    Accepted:
    - 1D array
    - list containing one 1D array
    - list of several 1D arrays -> returns median trajectory on common horizon
    """
    if isinstance(y, list):
        if len(y) == 0:
            return np.array([], dtype=float)

        if np.ndim(y[0]) >= 1:
            lengths = [len(np.asarray(v).ravel()) for v in y]
            T = min(lengths)
            A = np.stack([np.asarray(v, dtype=float).ravel()[:T] for v in y], axis=0)
            return np.median(A, axis=0)

        return np.asarray(y, dtype=float).ravel()

    return np.asarray(y, dtype=float).ravel()


def plot_residuals_vs_d(
    results_by_d,
    algos,
    c_values,
    save_path,
    legend_path=None,
    alpha=None,
    n=None,
    epsilon=None,
    noise=None,
    algo_colors=None,
    c_markers_cycle=None,
    step=1,
    cut=100000,
    fontsize=40,
    markersize=30,
    linewidth=3,
    ylabel="Residual",
    xlabel=r"Time step $(t)$",
    ylog=True,
):
    """
    Plot residual trajectories.

    Parameters
    ----------
    results_by_d : dict
        Expected structure:
        results_by_d[d][c]["residuals"][algo] = 1D array
        or a list of 1D arrays (multiple runs).

    algos : list[str]
        Algorithms to plot.

    c_values : list
        Budget values.

    save_path : str
        Output path, e.g. 'figures/myplot.pdf'

    legend_path : str or None
        Separate legend file path.

    algo_colors : dict or None
        Example: {"OGD":"tab:blue", "DA":"tab:orange", "BR":"tab:purple"}

    c_markers_cycle : list or None
        Markers used for d-values.

    step : int
        Plot every `step` points.

    cut : int
        Maximum time horizon.

    ylog : bool
        Use log scale on y-axis.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    if legend_path is not None:
        os.makedirs(os.path.dirname(legend_path) or ".", exist_ok=True)

    if algo_colors is None:
        algo_colors = {"OGD": "tab:blue", "DA": "tab:orange", "BR": "tab:purple"}

    if c_markers_cycle is None:
        c_markers_cycle = ["s", "o", "^", "X", "D", "*", "h", "v", "+"]

    d_values = sorted(results_by_d.keys())
    d_to_marker = {d: c_markers_cycle[i % len(c_markers_cycle)] for i, d in enumerate(d_values)}

    # -------- build common horizon safely --------
    max_len = 0
    for d in d_values:
        for c in c_values:
            for algo in algos:
                y = _prepare_curve(results_by_d[d][c]["residuals"][algo])
                if len(y) > 0:
                    max_len = max(max_len, len(y))

    if max_len == 0:
        raise ValueError("No data found to plot.")

    T = min(cut, max_len)
    it = np.arange(1, T + 1)

    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    for d in d_values:
        marker = d_to_marker[d]
        for c in c_values:
            for algo in algos:
                y = _prepare_curve(results_by_d[d][c]["residuals"][algo])
                if len(y) == 0:
                    continue

                L = min(T, len(y))
                yy = y[:L]

                if ylog:
                    yy = np.maximum(yy, 1e-16)

                ax.plot(
                    it[:L][::step],
                    yy[::step],
                    linestyle="--",
                    linewidth=linewidth,
                    color=algo_colors.get(algo, None),
                    marker=marker,
                    markersize=markersize,
                    markerfacecolor=algo_colors.get(algo, None),
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                )

    # -------- x scientific notation --------
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(True)

    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_fontsize(fontsize)
    ax.xaxis.get_offset_text().set_fontweight("bold")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.set_xlabel(xlabel, fontsize=2 * fontsize)
    ax.set_ylabel(ylabel, fontsize=2 * fontsize)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=int(1.25 * fontsize),
        width=2,
        length=8
    )

    if ylog:
        ax.set_yscale("log")

    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")

    # =========================
    # Separate legend file
    # =========================
    # -------- légende séparée : 2 lignes DA / OGD --------
    if legend_path is not None:
        handles = []
        labels = []

        # 🔥 IMPORTANT : ordre imposé
        ordered_algos = ["DA", "OGD"]

        # construction des lignes
        for algo in ordered_algos:
            for d in d_values:
                handles.append(
                    Line2D(
                        [0], [0],
                        color=algo_colors.get(algo, "black"),
                        linestyle="--",
                        linewidth=linewidth,
                        marker=d_to_marker[d],
                        markersize=markersize,
                        markerfacecolor=algo_colors.get(algo, "black"),
                        markeredgecolor="black",
                        markeredgewidth=1.2,
                    )
                )
                labels.append(fr"$d={d}$")

        ncol = len(d_values)

        fig_leg = plt.figure(figsize=(3.2 * ncol, 3.6))
        fig_leg.legend(
            handles,
            labels,
            loc="center",
            ncol=ncol,
            frameon=True,
            prop={"size": int(1.1 * fontsize), "weight": "bold"},
            handlelength=2.4,
            columnspacing=1.5,
            handletextpad=0.8,
        )

        fig_leg.savefig(legend_path, bbox_inches="tight")
        plt.close(fig_leg)

        print(f"Saved legend: {legend_path}")

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker


def plot_algo_vs_budget(
    res_by_c,
    algos,
    budgets,
    save_path,
    tol = 1e-5,
    legend_path=None,
    algo_colors=None,
    budget_markers=None,
    step=1,
    cut=None,
    fontsize=40,
    markersize=20,
    linewidth=3,
    ylog=True,
    ylabel="Residual",
):
    """
    Plot:
        color = algorithm
        marker = budget

    res_by_c[c]["residuals"][algo] = 1D array
    """

    if algo_colors is None:
        algo_colors = {"OGD": "tab:blue", "DA": "tab:orange"}

    #if budget_markers is None:
    markers_cycle = ["s", "o", "^", "X", "D", "*", "h", "v", "+"]

    budget_markers = {}
    k = 0

    for c in budgets:
        for algo in algos:
            budget_markers[(algo, c)] = markers_cycle[k % len(markers_cycle)]
            k += 1


    # ---- Build common horizon ----
    max_len = 0
    for c in budgets:
        for algo in algos:
            y = np.asarray(res_by_c[c]["residuals"][algo])
            max_len = max(max_len, len(y))

    T = max_len if cut is None else min(cut, max_len)
    it = np.arange(1, T + 1)

    fig, ax = plt.subplots(1, 1, figsize=(20, 12))


    for algo in algos:
        for c in budgets:
            marker = budget_markers[(algo, c)]
            y = np.asarray(res_by_c[c]["residuals"][algo])
            L = min(T, len(y))

            yy = y[:L]
            if ylog:
                yy = np.maximum(yy, tol)

            ax.plot(
                it[:L][::step],
                yy[::step],
                color=algo_colors[algo],
                linestyle="--",
                linewidth=linewidth,
                marker=marker,
                markersize=markersize,
                markerfacecolor=algo_colors[algo],
                markeredgecolor="black",
                markeredgewidth=1.5,
            )

    # ---------- AXIS ----------
    ax.set_xlabel(r"Time step $(t)$", fontsize=2*fontsize)#, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=2*fontsize)#, fontweight="bold")

    if ylog:
        ax.set_yscale("log")

    # -------- x scientific notation --------
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(True)

    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_fontsize(fontsize)
    ax.xaxis.get_offset_text().set_fontweight("bold")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=int(1.2 * fontsize),
        width=2,
        length=8,
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")

    # =========================
    # SEPARATE LEGEND
    # =========================
    if legend_path is not None:
        handles = []
        labels = []

        # --- Algorithms (colors) ---
        for algo in algos:
            handles.append(
                Line2D(
                    [0], [0],
                    color=algo_colors[algo],
                    linewidth=8,
                    linestyle="--"
                )
            )
            labels.append(algo)

        # --- Budgets (markers) ---
        for c in budgets:
            handles.append(
                Line2D(
                    [0], [0],
                    color="black",
                    marker=budget_markers[(algo, c)],
                    linestyle="None",
                    markersize=markersize,
                )
            )
            labels.append(fr"$c={c}$")

        fig_leg = plt.figure(figsize=(14, 2))
        fig_leg.legend(
            handles,
            labels,
            loc="center",
            ncol=len(handles),
            frameon=True,
            prop={"size": fontsize, "weight": "bold"},
        )

        fig_leg.savefig(legend_path, bbox_inches="tight")
        plt.close(fig_leg)

        print(f"Saved legend: {legend_path}")


from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


def save_combined_legend(
    algos,
    budgets,
    algo_colors,
    budget_markers,
    save_path,
    fontsize=40,
    linewidth=4,
    markersize=20,
):
    """
    Legend entries like:
        OGD-100, DA-100, OGD-50, DA-50
    """

    handles = []
    labels = []

    if algo_colors is None:
        algo_colors = {"OGD": "tab:blue", "DA": "tab:orange"}

    #if budget_markers is None:
    markers_cycle = ["s", "o", "^", "X", "D", "*", "h", "v", "+"]

    budget_markers = {}
    k = 0

    for c in budgets:
        for algo in algos:
            budget_markers[(algo, c)] = markers_cycle[k % len(markers_cycle)]
            k += 1

    for c in budgets:
        for algo in algos:
            handles.append(
                Line2D(
                    [0], [0],
                    color=algo_colors[algo],
                    linestyle="--",
                    linewidth=linewidth,
                    marker=budget_markers[(algo, c)],
                    markersize=markersize,
                    markerfacecolor=algo_colors[algo],
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                )
            )
            labels.append(f"{algo}-{c}")

    fig = plt.figure(figsize=(14, 2))

    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=len(labels),   # all in one row like your figure
        frameon=True,
        prop={"size": fontsize, "weight": "bold"},
    )

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved legend: {save_path}")


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker


def plot_final_res_vs_d_with_quantiles(
    results_alpha0,
    results_alpha1,
    ds,
    budget,
    algo,
    save_path,
    legend_path=None,
    color_lin="tab:blue",
    color_log="tab:orange",
    d_markers=None,
    q_low=0.1,
    q_high=0.9,
    center_mode="median",
    fontsize=40,
    markersize=20,
    linewidth=4,
    band_alpha=0.15,
    ylog=True,
    ylabel="Final residual",
    xlabel=r"$(d)$",
):
    """
    results_alphaX[d][budget]["final_res"][algo] = list of runs
    each run can be scalar or size-1 array
    """
    "^", "X", "D", "*", "h", "v", "+"

    if d_markers is None:
        d_markers = {2: "s", 3: "o", 5: "^",7:"*", 10: "X"}

    def summarize(vals):
        arr = np.asarray(vals, dtype=float).reshape(-1)
        if center_mode == "mean":
            center = arr.mean()
        else:
            center = np.median(arr)
        low = np.quantile(arr, q_low)
        high = np.quantile(arr, q_high)
        return center, low, high

    x = np.asarray(ds, dtype=float)

    y0, y0_low, y0_high = [], [], []
    y1, y1_low, y1_high = [], [], []

    for d in ds:
        vals0 = results_alpha0[d][budget]["final_res"][algo]
        vals1 = results_alpha1[d][budget]["final_res"][algo]

        c0, l0, h0 = summarize(vals0)
        c1, l1, h1 = summarize(vals1)

        y0.append(c0); y0_low.append(l0); y0_high.append(h0)
        y1.append(c1); y1_low.append(l1); y1_high.append(h1)

    y0 = np.asarray(y0, dtype=float)
    y0_low = np.asarray(y0_low, dtype=float)
    y0_high = np.asarray(y0_high, dtype=float)

    y1 = np.asarray(y1, dtype=float)
    y1_low = np.asarray(y1_low, dtype=float)
    y1_high = np.asarray(y1_high, dtype=float)

    if ylog:
        eps = 1e-16
        y0 = np.maximum(y0, eps)
        y0_low = np.maximum(y0_low, eps)
        y0_high = np.maximum(y0_high, eps)

        y1 = np.maximum(y1, eps)
        y1_low = np.maximum(y1_low, eps)
        y1_high = np.maximum(y1_high, eps)

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # quantile bands
    ax.fill_between(x, y0_low, y0_high, color=color_lin, alpha=band_alpha, linewidth=0)
    ax.fill_between(x, y1_low, y1_high, color=color_log, alpha=band_alpha, linewidth=0)

    # central lines
    ax.plot(x, y0, color=color_lin, linestyle="-", linewidth=linewidth)
    ax.plot(x, y1, color=color_log, linestyle="-", linewidth=linewidth)

    # markers on central lines
    for i, d in enumerate(ds):
        mk = d_markers[d]

        ax.plot(
            x[i], y0[i],
            linestyle="None",
            marker=mk,
            markersize=markersize,
            color=color_lin,
            markerfacecolor=color_lin,
            markeredgecolor="black",
            markeredgewidth=1.2,
        )

        ax.plot(
            x[i], y1[i],
            linestyle="None",
            marker=mk,
            markersize=markersize,
            color=color_log,
            markerfacecolor=color_log,
            markeredgecolor="black",
            markeredgewidth=1.2,
        )

    ax.set_xlabel(xlabel, fontsize=2 * fontsize, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=2 * fontsize, fontweight="bold")

    ax.set_xticks(ds)
    ax.xaxis.set_major_locator(mticker.FixedLocator(ds))

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=int(1.1 * fontsize),
        width=2,
        length=8
    )

    if ylog:
        ax.set_yscale("log")

    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")

    if legend_path is not None:
        handles, labels = [], []

        # family colors
        handles.append(Line2D([0], [0], color=color_lin, linewidth=linewidth, linestyle="-"))
        labels.append(r"$\alpha=0$")

        handles.append(Line2D([0], [0], color=color_log, linewidth=linewidth, linestyle="-"))
        labels.append(r"$\alpha=1$")

        # d markers
        for d in ds:
            handles.append(
                Line2D(
                    [0], [0],
                    color="black",
                    linestyle="None",
                    marker=d_markers[d],
                    markersize=markersize,
                )
            )
            labels.append(fr"$d={d}$")

        fig_leg = plt.figure(figsize=(16, 2.2))
        fig_leg.legend(
            handles,
            labels,
            loc="center",
            ncol=len(labels),
            frameon=True,
            prop={"size": fontsize - 4, "weight": "bold"},
        )
        fig_leg.savefig(legend_path, bbox_inches="tight")
        plt.close(fig_leg)

        print(f"Saved legend: {legend_path}")