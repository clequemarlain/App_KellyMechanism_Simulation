import numpy as np
import torch
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import root_scalar

# ---- Plotting palette (kept but not required by Streamlit; user can override) ----
colors = [
    "darkorange","royalblue","green","purple","gold","teal","magenta","brown","black",
    "crimson","darkcyan","indigo","salmon","lime","navy","coral","darkgreen","orchid","slategray","darkkhaki"
]
markers = ["s","^","v","D","*","p","x","+","|","s","^","v","D","*","p","x","+","|","s","^","v","D","*","p","x","+","|"]

def V_func(x, alpha: float):
    if alpha == 1:
        return torch.log(x)
    return 1.0 / (1.0 - alpha) * (x ** (1.0 - alpha))

def Q1(acc_gradient, eps, c, price):
    return torch.minimum(torch.maximum(eps/price, acc_gradient), c/price)

def Q2(acc_gradient, eps, c, price):
    return torch.maximum(eps/price, torch.minimum(torch.exp(acc_gradient - 1), c/price))

def solve_nonlinear_eq(a, s, alpha, eps, c_vector, price=1.0, max_iter=100, tol=1e-5):
    """
    Solves for z in: price * (z + s_i)^(2 - alpha) * z^alpha = a_i * s_i
    for each i, using bisection. Falls back to Q1 when the bracket fails.
    """
    a = a.detach().cpu().numpy() if torch.is_tensor(a) else np.array(a, dtype=float)
    s = s.detach().cpu().numpy() if torch.is_tensor(s) else np.array(s, dtype=float)
    c_vec = c_vector.detach().cpu().numpy() if torch.is_tensor(c_vector) else np.array(c_vector, dtype=float)

    n = len(a)
    z_list = []
    for i in range(n):
        def f(z):
            return price * (z + s[i]) ** (2 - alpha) * (z ** alpha) - a[i] * s[i]
        lower_bound = max(tol, 1e-12)
        upper_bound = c_vec[i] / price if c_vec[i] > 0 else 1.0
        try:
            # Ensure sign change; expand upper bound if needed
            fl = f(lower_bound)
            fu = f(upper_bound)
            k = 0
            while fl * fu > 0 and k < 30:
                upper_bound *= 2.0
                fu = f(upper_bound)
                k += 1
            if fl * fu > 0:
                raise RuntimeError("No sign change")
            sol = root_scalar(f, bracket=[lower_bound, upper_bound], method='bisect', xtol=tol, maxiter=max_iter)
            if sol.converged:
                z_list.append(sol.root)
            else:
                z_list.append(lower_bound)
        except Exception:
            # fallback
            z_list.append(lower_bound)
    z_tensor = torch.tensor(z_list, dtype=torch.float64)
    return Q1(z_tensor, eps, c_vector, price)

def Utility(x, a_vector, d_vector, alpha):
    V = V_func(x, alpha)
    return a_vector * V + d_vector

def LSW_func(x, budgets, a_vector, d_vector, alpha):
    utility = Utility(x, a_vector, d_vector, alpha)
    utility_bugeted = torch.minimum(utility, budgets)
    lsw = torch.sum(utility_bugeted)
    return lsw

class GameKelly:
    def __init__(self, n: int, price: float, epsilon, delta, alpha, tol):
        self.n = n
        self.price = price
        self.epsilon = epsilon
        self.delta = delta
        self.alpha = alpha
        self.tol = tol

    def fraction_resource(self, z):
        return z / (torch.sum(z) + self.delta)

    def grad_phi(self, phi, bids):
        z = bids.clone().detach().requires_grad_(True)
        jacobi = torch.autograd.functional.jacobian(phi, z)
        return jacobi.diag()

    def check_NE(self, z: torch.Tensor, a_vector, c_vector, d_vector):
        p = torch.sum(z) - z + self.delta
        if self.alpha not in [0, 1, 2]:
            err = torch.maximum(torch.norm(solve_nonlinear_eq(a_vector, p, self.alpha, self.epsilon, c_vector, self.price, max_iter=1000, tol=self.tol) - z),
                                self.tol * torch.ones(1))
        else:
            br = BR_alpha_fair(self.epsilon, c_vector, z, p, a_vector, self.delta, self.alpha, self.price, b=0)
            err = torch.maximum(torch.norm(br - z), self.tol * torch.ones(1))
        return err

    def AverageBid(self, z, t):
        return (1.0/t) * torch.sum(z, dim=0)

    def XL(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector
        acc_grad_copy = acc_grad.clone()
        grad_t = self.grad_phi(phi, bids)
        acc_grad_copy += grad_t / (t ** eta) if vary else grad_t * eta
        z_t = torch.maximum(self.epsilon / self.price, c_vector / (1 + torch.exp(-acc_grad_copy)))
        return z_t, acc_grad_copy

    def Hybrid(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        acc_grad_copy = acc_grad.clone()
        z_t = torch.zeros_like(bids)
        for idx_set, func in enumerate(Hybrid_funcs):
            func = getattr(self, func)
            z_t[Hybrid_sets[idx_set]], acc_grad_copy[Hybrid_sets[idx_set]] = func(
                t, a_vector[Hybrid_sets[idx_set]], c_vector[Hybrid_sets[idx_set]],
                d_vector[Hybrid_sets[idx_set]], eta, bids[Hybrid_sets[idx_set]], acc_grad[Hybrid_sets[idx_set]], vary=vary
            )
        return z_t, acc_grad_copy

    def AsynXL(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, vary=False):
        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector
        acc_grad_copy = acc_grad.clone()
        z_t = bids.clone()
        n = bids.shape[0]
        for i in range(n):
            grad_t = self.grad_phi(phi, z_t)
            acc_grad_copy[i] += grad_t[i] / (t ** eta) if vary else grad_t[i] * eta
            z_t[i] = torch.maximum(self.epsilon / self.price, c_vector[i] / (1 + torch.exp(-acc_grad_copy[i])))
        return z_t, acc_grad_copy

    def OGD(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector
        grad_t = self.grad_phi(phi, bids)
        eta_t = (1 / (t ** eta)) if (vary and t > 0) else eta
        z_candidate = bids + eta_t * grad_t
        z_t = Q1(z_candidate, self.epsilon, c_vector, self.price)
        return z_t, acc_grad

    def DAQ(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector
        acc_grad_copy = acc_grad.clone()
        grad_t = self.grad_phi(phi, bids)
        acc_grad_copy += (grad_t / (t ** eta) if (vary and t > 0) else grad_t * eta)
        z_t = Q1(acc_grad_copy, self.epsilon, c_vector, self.price)
        return z_t, acc_grad_copy

    def DAH(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector
        acc_grad_copy = acc_grad.clone()
        grad_t = self.grad_phi(phi, bids)
        acc_grad_copy += (grad_t / (t ** eta) if (vary and t > 0) else grad_t * eta)
        z_t = Q2(acc_grad_copy, self.epsilon, c_vector, self.price)
        return z_t, acc_grad_copy

    def AsynDAQ(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        def phi(z):
            x = self.fraction_resource(z)
            V = V_func(x, self.alpha)
            return a_vector * V - self.price * z + d_vector
        acc_grad_copy = acc_grad.clone()
        z_t = bids.clone()
        n = bids.shape[0]
        for i in range(n):
            grad_t = self.grad_phi(phi, z_t)
            acc_grad_copy[i] += grad_t[i] / (t ** eta) if vary else grad_t[i] * eta
            z_t[i] = Q1(acc_grad_copy[i], self.epsilon, c_vector, self.price)
        return z_t, acc_grad_copy

    def SBRD(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, b=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        p = torch.sum(bids) - bids + self.delta
        z_t = BR_alpha_fair(self.epsilon, c_vector, bids, p, a_vector, self.delta, self.alpha, self.price, b=b)
        return z_t, acc_grad

    def NumSBRD(self, t, a_vector, c_vector, d_vector, eta, bids, acc_grad, b=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        p = torch.sum(bids) - bids + self.delta
        z_t = solve_nonlinear_eq(a_vector, p, self.alpha, self.epsilon, c_vector, self.price, max_iter=100, tol=self.tol)
        z_t = Q1(z_t, self.epsilon, c_vector, self.price)
        return z_t, acc_grad

    def AsynBRD(self, a_vector, c_vector, d_vector, t, eta, bids, acc_grad, b=0, vary=False, Hybrid_funcs=None, Hybrid_sets=None):
        n = bids.shape[0]
        z_t = bids.clone()
        for i in range(n):
            p = torch.sum(z_t) - z_t[i] + self.delta
            z_t[i] = BR_alpha_fair(self.epsilon, c_vector[i], z_t[i], p, a_vector[i], self.delta, self.alpha, self.price, b=b)
            z_t[i] = Q1(z_t[i], self.epsilon, c_vector[i], self.price)
        return z_t, acc_grad

    def learning(self, func, a_vector, c_vector, d_vector, n_iter: int, eta, bids, vary: bool = False, stop=False, Hybrid_funcs=None, Hybrid_sets=None):
        func = getattr(self, func)
        acc_grad = torch.zeros(self.n, dtype=torch.float64)
        matrix_bids = torch.zeros((n_iter + 1, self.n), dtype=torch.float64)
        Avg_bids = matrix_bids.clone()
        vec_LSW = torch.zeros(n_iter + 1, dtype=torch.float64)
        utiliy = torch.zeros((n_iter + 1, self.n), dtype=torch.float64)
        error_NE = torch.zeros(n_iter + 1, dtype=torch.float64)
        matrix_bids[0] = bids.clone().to(torch.float64)
        Avg_bids[0] = bids.clone().to(torch.float64)
        error_NE[0] = self.check_NE(bids, a_vector, c_vector, d_vector)
        utiliy[0] = Utility(self.fraction_resource(matrix_bids[0]), a_vector, d_vector, self.alpha)

        k = 0
        for t in range(1, n_iter + 1):
            k = t
            matrix_bids[t], acc_grad = func(t, a_vector, c_vector, d_vector, eta, matrix_bids[t-1], acc_grad, vary=vary, Hybrid_funcs=Hybrid_funcs, Hybrid_sets=Hybrid_sets)
            error_NE[t] = self.check_NE(matrix_bids[t], a_vector, c_vector, d_vector)
            vec_LSW[t] = LSW_func(self.fraction_resource(matrix_bids[t]), c_vector, a_vector, d_vector, self.alpha)
            utiliy[t] = Utility(self.fraction_resource(matrix_bids[t]), a_vector, d_vector, self.alpha)
            err = torch.min(error_NE[:k])
            Avg_bids[t] = self.AverageBid(matrix_bids, t)
            if stop and err <= self.tol:
                break
        return matrix_bids[:k, :], utiliy[:k, :], error_NE[:k]

def BR_alpha_fair(eps, c_vector, z: torch.Tensor, p, a_vector: torch.Tensor, delta, alpha, price: float, b=0):
    a_vector = a_vector.to(dtype=torch.float64)
    if alpha == 0:
        br = -p + torch.sqrt(a_vector * p / price)
    elif alpha == 1:
        if b == 0:
            br = (-p + torch.sqrt(p ** 2 + 4 * a_vector * p / price)) / 2
        else:
            discriminant = p ** 2 + 4 * a_vector * p * (1 + b) / price
            br = (-p * (2 * b + 1) + torch.sqrt(discriminant)) / (2 * (1 + b))
    elif alpha == 2:
        br = torch.sqrt(a_vector * p / price)
    else:
        # For general alpha, use solve_nonlinear_eq via check_NE/NumSBRD; here fallback:
        br = torch.sqrt(torch.clamp(a_vector * p / price, min=0.0))
    return Q1(br, eps, c_vector, price)

def plotGame(x_data, y_data, x_label, y_label, legends, saveFileName, ylog_scale, fontsize=16, markersize=6, linewidth=2, linestyle="-", pltText=False, step=1):
    plt.figure()
    y_data = np.array(y_data, dtype=object)
    if ylog_scale:
        plt.yscale("log")
    legend_handles = [
        Line2D([0], [0], color=colors[i], linestyle=linestyle, linewidth=linewidth)
        for i in range(len(legends))
    ]
    for i in range(len(legends)):
        if linestyle == "":
            mask = y_data[i] > 0
            x_vals = [x_data[i]] * y_data[i][mask].shape[0]
            plt.plot(x_vals[::step], (y_data[i][mask])[::step], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=markers[i], markersize=markersize, label=f"{legends[i]}")
        else:
            plt.plot(x_data[::step], (y_data[i])[::step], color=colors[i], linestyle=linestyle, linewidth=linewidth, marker=markers[i], markersize=markersize, label=f"{legends[i]}")
        if pltText:
            last_y = y_data[i][-1]
            plt.text(len(y_data[i]) - 1, last_y, f"{last_y:.2e}", bbox=dict(facecolor='white', alpha=0.7))
        plt.legend(legend_handles, legends, frameon=True, facecolor="white", edgecolor="black",
                   prop={"weight": "bold"})

    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight("bold")  # ✅ Graduation des axes en gras

    plt.ylabel(f"{y_label}", fontweight="bold")
    plt.xlabel(f"{x_label}", fontweight="bold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{saveFileName}.pdf", format="pdf")
    return f"{saveFileName}.pdf"

def plotGame_dim_N(x_data, y_data, x_label, y_label, legends, saveFileName, ylog_scale, fontsize=16, markersize=6, linewidth=2, linestyle="-", pltText=False, step=1):
    plt.figure()
    y_data = np.array(y_data, dtype=object)
    linestyles = ["-", "--", ":", "-."]
    if ylog_scale:
        plt.yscale("log")
        # --- Custom legend: linestyle + color only (no marker) ---
    legend_handles = [
        Line2D([0], [0], color=colors[i], linestyle=linestyles[i], linewidth=linewidth)
        for i in range(len(legends))
    ]
    for i in range(len(legends)):
        n = len((y_data[i, 0]))
        for j in range(n):
            plt.plot(x_data[::step], (y_data[i])[:, j][::step], color=colors[i], linestyle=linestyles[i % len(linestyles)], linewidth=linewidth, marker=markers[j], markersize=markersize)
    plt.legend(legend_handles, legends, frameon=True, facecolor="white", edgecolor="black",
               prop={"weight": "bold"})

    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight("bold")  # ✅ Graduation des axes en gras

    plt.ylabel(f"{y_label}", fontweight="bold")
    plt.xlabel(f"{x_label}", fontweight="bold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{saveFileName}.pdf", format="pdf")
    return f"{saveFileName}.pdf"