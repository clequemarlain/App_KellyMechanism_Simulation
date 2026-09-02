import ast
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from scipy.optimize import minimize

from Journal2025.style import apply_journal_style
from main import SimulationRunner

from src.game.utils import *
from src.game.config import SIMULATION_CONFIG as DEFAULT_CONFIG
from src.game.config import SIMULATION_CONFIG_table as DEFAULT_CONFIG_TABLE
from src.game.description import ALGO_DESCRIPTIONS
from main_table_simulation import run_simulation_table_avg, display_results_streamlit_dict
from src.game.Jain_index import run_jain_vs_gamma, plot_jain_vs_gamma
from linear_VS_log import run_main_gamma_curvature
#from simulation_param_n_gamma import *

JOURNAL_FIGURE_DIR = apply_journal_style()
PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def build_intro_animation() -> bytes:
    """Build the introductory animation once from the bundled PNG diagrams."""
    diagram_paths = [
        PROJECT_ROOT / "src" / "game" / f"kellyMechanism-Journal-Page-{page}.drawio.png"
        for page in range(1, 5)
    ]
    frames = []
    for diagram_path in diagram_paths:
        with Image.open(diagram_path) as image:
            frames.append(image.convert("RGB"))

    animation = io.BytesIO()
    frames[0].save(
        animation,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=3000,
        loop=0,
    )
    return animation.getvalue()


def figure_prefix(config, metric=None, suffix="n"):
    """Build a stable path prefix for generated journal figures."""
    metric_name = metric or config["metric"]
    if suffix == "player":
        filename = f"{metric_name}_alpha{config['alpha']}_gamma{config['gamma']}_player"
    else:
        filename = f"{metric_name}_alpha{config['alpha']}_gamma{config['gamma']}_n_{config['n']}"
    return str(JOURNAL_FIGURE_DIR / filename)


def _project_bid_budget(values, budget, minimum, required=None):
    """Project one player's resource bids onto {z >= minimum, sum(z) <= budget}."""
    values = np.asarray(values, dtype=float)
    if required is None:
        required = np.ones(values.shape, dtype=bool)
    required = np.asarray(required, dtype=bool)
    result = np.zeros_like(values)
    if not np.any(required):
        return result
    active_values = values[required]
    active_count = int(required.sum())
    active_values = np.maximum(active_values, minimum)
    if active_values.sum() <= budget:
        result[required] = active_values
        return result

    shifted = active_values - minimum
    radius = max(float(budget) - active_count * minimum, 0.0)
    ordered = np.sort(shifted)[::-1]
    cumulative = np.cumsum(ordered)
    active = np.nonzero(ordered * np.arange(1, len(ordered) + 1) > cumulative - radius)[0]
    theta = (cumulative[active[-1]] - radius) / (active[-1] + 1) if active.size else 0.0
    result[required] = np.maximum(shifted - theta, 0.0) + minimum
    return result


def _multiresource_alpha_fair_utility(x, alpha):
    """Standard alpha-fair utility used by the multi-resource experiment."""
    x = np.maximum(np.asarray(x, dtype=float), 1e-12)
    alpha = np.asarray(alpha, dtype=float)
    alpha = np.broadcast_to(alpha, x.shape)
    result = np.empty_like(x)
    linear = np.isclose(alpha, 0.0)
    logarithmic = np.isclose(alpha, 1.0)
    other = ~(linear | logarithmic)
    result[linear] = x[linear]
    result[logarithmic] = np.log(x[logarithmic])
    result[other] = (
        np.power(x[other], 1.0 - alpha[other]) / (1.0 - alpha[other])
    )
    return result


def _multiresource_alpha_fair_marginal(x, alpha):
    """Derivative of the standard alpha-fair utility."""
    x = np.maximum(np.asarray(x, dtype=float), 1e-12)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), x.shape)
    return np.power(x, -alpha)


def _multiresource_best_response(
        bids, capacities, valuations, budgets, alpha_by_player,
        minimum_bid, delta, requirements=None):
    r"""Return every player's joint best response with opponents fixed.

    Player ``i`` solves

        max sum_k a_i^k V(x_i^k(z_i^k; z_-i^k)) - z_i^k
        s.t. z_i^k >= minimum_bid and sum_k z_i^k <= c_i.

    The payoff is concave and separable across resources once opponents are
    fixed.  A scalar KKT multiplier therefore couples the resource bids.
    """
    bids = np.asarray(bids, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    valuations = np.asarray(valuations, dtype=float)
    budgets = np.asarray(budgets, dtype=float)
    alpha_by_player = np.asarray(alpha_by_player, dtype=float)
    player_count, resource_count = bids.shape
    if requirements is None:
        requirements = np.ones_like(bids, dtype=bool)
    requirements = np.asarray(requirements, dtype=bool)
    opponents = bids.sum(axis=0)[None, :] - bids + float(delta)

    best_response = np.empty_like(bids)
    for player in range(player_count):
        active = requirements[player]
        if not np.any(active):
            best_response[player] = 0.0
            continue
        competition = opponents[player]

        def negative_payoff(values):
            totals = np.maximum(values + competition, 1e-12)
            allocations = capacities * values / totals
            utility = np.sum(
                valuations[player, active]
                * _multiresource_alpha_fair_utility(
                    allocations[active], alpha_by_player[player]
                )
                - values[active]
            )
            return -float(utility)

        def negative_payoff_gradient(values):
            totals = np.maximum(values + competition, 1e-12)
            allocations = capacities * values / totals
            marginal = _multiresource_alpha_fair_marginal(
                allocations, alpha_by_player[player]
            )
            gradient = np.zeros(resource_count, dtype=float)
            gradient[active] = -(
                valuations[player, active] * marginal[active]
                * capacities[active] * competition[active] / totals[active] ** 2
                - 1.0
            )
            return gradient

        initial = _project_bid_budget(
            bids[player], budgets[player], minimum_bid, active
        )
        result = minimize(
            negative_payoff,
            initial,
            jac=negative_payoff_gradient,
            method="SLSQP",
            bounds=[(minimum_bid, None) if flag else (0.0, 0.0) for flag in active],
            constraints=[{
                "type": "ineq",
                "fun": lambda values, budget=budgets[player]: (
                    budget - np.sum(values)
                ),
                "jac": lambda values: -np.ones_like(values),
            }],
            options={"ftol": 1e-10, "maxiter": 100},
        )
        if not result.success:
            raise RuntimeError(
                f"Best-response solver failed for player {player + 1}: "
                f"{result.message}"
            )
        best_response[player] = result.x

    return best_response


def _multiresource_exact_best_response(
        bids, capacities, valuations, budgets, alpha_by_player,
        minimum_bid, delta, requirements=None, tolerance=1e-12):
    """Joint playerwise BR using the closed forms for alpha in {0, 1, 2}.

    A nonnegative KKT multiplier enforces each player's shared bid budget.
    Consequently only a scalar bisection is needed when that budget binds.
    """
    bids = np.asarray(bids, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    valuations = np.asarray(valuations, dtype=float)
    budgets = np.asarray(budgets, dtype=float)
    alpha_by_player = np.asarray(alpha_by_player, dtype=float)
    if requirements is None:
        requirements = np.ones_like(bids, dtype=bool)
    requirements = np.asarray(requirements, dtype=bool)
    opponents = bids.sum(axis=0)[None, :] - bids + float(delta)
    response = np.zeros_like(bids)

    for player in range(bids.shape[0]):
        active = requirements[player]
        alpha = float(alpha_by_player[player])
        if not any(np.isclose(alpha, supported) for supported in (0.0, 1.0, 2.0)):
            raise ValueError("Exact BR supports only alpha = 0, 1, or 2.")
        competition = opponents[player, active]
        weighted_capacity = (
            valuations[player, active]
            * np.power(capacities[active], 1.0 - alpha)
        )

        def bids_at(multiplier):
            price = 1.0 + multiplier
            if np.isclose(alpha, 0.0):
                values = np.sqrt(weighted_capacity * competition / price) - competition
            elif np.isclose(alpha, 1.0):
                values = 0.5 * (
                    -competition
                    + np.sqrt(competition ** 2 + 4.0 * weighted_capacity * competition / price)
                )
            else:
                values = np.sqrt(weighted_capacity * competition / price)
            return np.maximum(values, minimum_bid)

        values = bids_at(0.0)
        if values.sum() > budgets[player] + tolerance:
            low, high = 0.0, 1.0
            while bids_at(high).sum() > budgets[player]:
                high *= 2.0
            for _ in range(100):
                middle = 0.5 * (low + high)
                if bids_at(middle).sum() > budgets[player]:
                    low = middle
                else:
                    high = middle
                if high - low <= tolerance * max(1.0, high):
                    break
            values = bids_at(high)
        response[player, active] = values
    return response


def _multiresource_ogd_constants(
        capacities, valuations, budgets, alpha, minimum_bid, delta,
        gradient_bound_mode="practical", requirements=None):
    """Return playerwise action diameters and selected gradient bounds.

    ``practical`` uses

        G_i = max_k(a_i^k / epsilon + 1),

    ``infinity`` maximizes the absolute full gradient component over the
    feasible domain and then takes the infinity norm across resources.
    ``legacy`` reproduces the original Run Simulation bound for one resource.
    """
    n, resource_count = valuations.shape
    if requirements is None:
        requirements = np.ones_like(valuations, dtype=bool)
    requirements = np.asarray(requirements, dtype=bool)
    active_counts = requirements.sum(axis=1)
    free_budget = np.maximum(budgets - active_counts * minimum_bid, 0.0)
    diameter = np.where(active_counts == 1, free_budget, np.sqrt(2.0) * free_budget)
    if gradient_bound_mode == "practical":
        gradient_bound = np.max(
            np.where(requirements, valuations / float(minimum_bid) + 1.0, 0.0),
            axis=1,
        )
        return diameter, gradient_bound
    if gradient_bound_mode == "legacy":
        if resource_count != 1:
            raise ValueError(
                "The legacy Run Simulation bound is available only for one resource."
            )
        # Reproduce the original positional call:
        # compute_G(a_i, epsilon, c_i, epsilon, n, alpha).
        legacy_epsilon = budgets
        legacy_delta = float(minimum_bid)
        legacy_upper_bid = float(minimum_bid)
        competition_min = (n - 1) * legacy_epsilon + legacy_delta
        competition_max = (n - 1) * legacy_upper_bid + legacy_delta
        player_interest = valuations[:, 0]
        gradient_bound = np.maximum(
            np.abs(
                player_interest * competition_max
                / (
                    legacy_epsilon
                    * (legacy_epsilon + competition_max)
                )
                - 1.0
            ),
            np.abs(
                player_interest * competition_min
                / (
                    legacy_epsilon
                    * (legacy_epsilon + competition_min)
                )
                - 1.0
            ),
        )
        return diameter, np.maximum(gradient_bound, 1e-12)
    if gradient_bound_mode != "infinity":
        raise ValueError(
            "Gradient bound mode must be 'practical', 'infinity', or 'legacy'."
        )

    capacities = np.asarray(capacities, dtype=float)
    player_alphas = np.asarray(alpha, dtype=float)
    if player_alphas.ndim == 0:
        player_alphas = np.full(n, float(player_alphas))
    player_alphas = np.broadcast_to(player_alphas, (n,))
    maximum_bid = minimum_bid + free_budget
    competition_min = (n - 1) * minimum_bid + float(delta)
    gradient_bound = np.empty(n, dtype=float)

    for player in range(n):
        competition_max = (
            np.sum(maximum_bid) - maximum_bid[player] + float(delta)
        )
        player_alpha = player_alphas[player]
        component_bounds = np.empty(resource_count, dtype=float)
        for resource in range(resource_count):
            z_candidates = [minimum_bid, maximum_bid[player]]
            candidates = []
            for bid in z_candidates:
                competition_candidates = [
                    competition_min, competition_max
                ]
                if player_alpha < 1.0:
                    stationary = bid / (1.0 - player_alpha)
                    if competition_min <= stationary <= competition_max:
                        competition_candidates.append(stationary)
                for competition in competition_candidates:
                    positive_term = (
                        valuations[player, resource]
                        * capacities[resource] ** (1.0 - player_alpha)
                        * competition
                        * (bid + competition) ** (player_alpha - 2.0)
                        * bid ** (-player_alpha)
                    )
                    candidates.append(abs(positive_term - 1.0))
            component_bounds[resource] = max(candidates)
        gradient_bound[player] = np.max(component_bounds)

    return diameter, np.maximum(gradient_bound, 1e-12)


def _multiresource_best_response_residual(
        bids, capacities, valuations, budgets, alpha_by_player,
        minimum_bid, delta, requirements=None, per_resource=False):
    r"""Return ||z^BR(z^t) - z^t||_2 for the joint bid vector."""
    aggregate, by_resource = _multiresource_best_response_residuals(
        bids, capacities, valuations, budgets, alpha_by_player,
        minimum_bid, delta, requirements,
    )
    return by_resource if per_resource else aggregate


def _multiresource_best_response_residuals(
        bids, capacities, valuations, budgets, alpha_by_player,
        minimum_bid, delta, requirements=None):
    """Return aggregate and per-resource BR residuals from one BR solve."""
    exact_alphas = all(
        any(np.isclose(alpha, supported) for supported in (0.0, 1.0, 2.0))
        for alpha in np.asarray(alpha_by_player, dtype=float)
    )
    solver = (
        _multiresource_exact_best_response
        if exact_alphas else _multiresource_best_response
    )
    best_response = solver(
        bids, capacities, valuations, budgets, alpha_by_player,
        minimum_bid, delta, requirements,
    )
    difference = best_response - bids
    by_resource = np.linalg.norm(difference, axis=0)
    return float(np.linalg.norm(by_resource)), by_resource


def run_multiresource_alpha_experiment(
        n, capacities, valuations, budgets, alphas, algorithm, iterations,
        minimum_bid, repetitions, seed=7, convergence_tolerance=1e-4,
        delta=0.0, player_alphas=None, step_scale=1.0,
        gradient_bound_mode="practical", residual_mode="iterate_difference",
        requirements=None, residual_evaluation_interval=1):
    """Run a learning rule on a parallel-resource alpha-fair Kelly game."""
    supported_algorithms = {"BR", "OGD_F", "OGD_V", "RRM_V", "DAQ_F"}
    if algorithm not in supported_algorithms:
        raise ValueError(
            "The multi-resource experiment supports "
            + ", ".join(sorted(supported_algorithms))
            + "."
        )
    if residual_mode not in {"iterate_difference", "best_response"}:
        raise ValueError(
            "Residual mode must be 'iterate_difference' or 'best_response'."
        )
    residual_evaluation_interval = int(residual_evaluation_interval)
    if residual_evaluation_interval < 1:
        raise ValueError("The residual evaluation interval must be at least one.")
    capacities = np.asarray(capacities, dtype=float)
    valuations = np.asarray(valuations, dtype=float)
    budgets = np.asarray(budgets, dtype=float)
    if requirements is None:
        requirements = np.ones_like(valuations, dtype=bool)
    requirements = np.asarray(requirements, dtype=bool)
    if requirements.shape != valuations.shape or np.any(requirements.sum(axis=1) == 0):
        raise ValueError("Every player must require at least one resource.")
    delta = float(delta)
    step_scale = float(step_scale)
    if delta < 0.0:
        raise ValueError("Delta must be nonnegative.")
    if step_scale <= 0.0:
        raise ValueError("The OGD step-size scale must be positive.")
    if valuations.shape != (n, len(capacities)):
        raise ValueError("The interest-factor matrix must have shape (players, resources).")
    if np.any(valuations[requirements] <= 0.0):
        raise ValueError("Every interest factor a_i^k must be positive.")
    if player_alphas is not None:
        player_alphas = np.asarray(player_alphas, dtype=float)
        if player_alphas.shape != (n,) or np.any(player_alphas < 0.0):
            raise ValueError("Enter exactly one nonnegative α value per device.")
        alpha_configurations = [(float(np.mean(player_alphas)), player_alphas)]
    else:
        alpha_configurations = [
            (float(alpha), np.full(n, float(alpha))) for alpha in alphas
        ]
    if algorithm == "BR" and any(
        not any(np.isclose(alpha, supported) for supported in (0.0, 1.0, 2.0))
        for _, values in alpha_configurations for alpha in values
    ):
        raise ValueError("Exact BR can be used only with alpha values 0, 1, and 2.")
    records, allocations, convergence_histories = [], {}, {}
    resource_convergence_histories = {}

    for alpha_axis_value, alpha_by_player in alpha_configurations:
        alpha_matrix = alpha_by_player[:, None]
        if algorithm == "BR":
            diameters = gradient_bounds = np.ones(n, dtype=float)
        else:
            diameters, gradient_bounds = _multiresource_ogd_constants(
                capacities, valuations, budgets, alpha_by_player, minimum_bid,
                delta, gradient_bound_mode=gradient_bound_mode,
                requirements=requirements,
            )
        run_metrics, final_x, run_histories, run_resource_histories = [], [], [], []
        for repetition in range(repetitions):
            rng = np.random.default_rng(seed + repetition)
            # Sample over the full feasible action set. For one resource this
            # is exactly Uniform[minimum_bid, budget], as in Run Simulation.
            resource_count = len(capacities)
            active_counts = requirements.sum(axis=1)
            free_budget = np.maximum(budgets - active_counts * minimum_bid, 0.0)
            if resource_count == 1:
                bids = rng.uniform(
                    minimum_bid, budgets[:, None], size=(n, 1)
                )
            else:
                directions = rng.dirichlet(
                    np.ones(resource_count), size=n
                )
                utilization = rng.uniform(0.0, 1.0, size=n)
                bids = requirements * (
                    minimum_bid
                    + utilization[:, None]
                    * free_budget[:, None]
                    * directions
                )
                bids = np.vstack([
                    _project_bid_budget(bids[i], budgets[i], minimum_bid, requirements[i])
                    for i in range(n)
                ])
            convergence_history = []
            resource_histories = []
            accumulated_gradient = np.zeros_like(bids)
            if residual_mode == "best_response":
                initial_residual, initial_resource_residual = (
                    _multiresource_best_response_residuals(
                        bids, capacities, valuations, budgets, alpha_by_player,
                        minimum_bid, delta, requirements,
                    )
                )
                convergence_history.append(initial_residual)
                resource_histories.append(initial_resource_residual)
            for t in range(1, iterations + 1):
                previous_bids = bids.copy()
                if algorithm == "BR":
                    bids = _multiresource_exact_best_response(
                        previous_bids, capacities, valuations, budgets,
                        alpha_by_player, minimum_bid, delta, requirements,
                    )
                else:
                    totals = np.maximum(bids.sum(axis=0) + delta, 1e-12)
                    x = capacities[None, :] * bids / totals[None, :]
                    marginal_utility = _multiresource_alpha_fair_marginal(x, alpha_matrix)
                    dx_db = capacities[None, :] * (totals[None, :] - bids) / totals[None, :] ** 2
                    gradient = (valuations * marginal_utility * dx_db - 1.0) * requirements
                    fixed_step = (
                        step_scale * diameters
                        / (gradient_bounds * np.sqrt(iterations))
                    )
                    varying_step = (
                        step_scale * diameters
                        / (gradient_bounds * np.sqrt(t))
                    )
                    if algorithm == "OGD_F":
                        candidate = bids + fixed_step[:, None] * gradient
                    elif algorithm == "OGD_V":
                        candidate = bids + varying_step[:, None] * gradient
                    elif algorithm == "RRM_V":
                        accumulated_gradient += varying_step[:, None] * gradient
                        candidate = accumulated_gradient
                    else:  # DAQ_F: fixed-horizon quadratic dual averaging
                        accumulated_gradient += gradient
                        candidate = fixed_step[:, None] * accumulated_gradient
                    bids = np.vstack([
                        _project_bid_budget(row, budgets[i], minimum_bid, requirements[i])
                        for i, row in enumerate(candidate)
                    ])
                if residual_mode == "iterate_difference":
                    resource_residual = np.linalg.norm(previous_bids - bids, axis=0)
                    residual = float(np.linalg.norm(resource_residual))
                elif (
                    t % residual_evaluation_interval == 0
                    or t == iterations
                ):
                    residual, resource_residual = (
                        _multiresource_best_response_residuals(
                            bids, capacities, valuations, budgets, alpha_by_player,
                            minimum_bid, delta, requirements,
                        )
                    )
                    residual_was_evaluated = True
                else:
                    # Exact BR residuals are substantially more expensive than
                    # a learning update. Hold the latest sampled value between
                    # evaluations so every trajectory retains T+1 entries.
                    residual = convergence_history[-1]
                    resource_residual = np.asarray(resource_histories[-1]).copy()
                    residual_was_evaluated = False
                convergence_history.append(residual)
                resource_histories.append(resource_residual)
                if (
                    residual <= convergence_tolerance
                    and (
                        residual_mode == "iterate_difference"
                        or residual_was_evaluated
                    )
                ):
                    # Preserve fixed-length arrays for averaging and plotting,
                    # without spending time recomputing an already converged run.
                    remaining = iterations - t
                    convergence_history.extend([residual] * remaining)
                    resource_histories.extend(
                        [np.asarray(resource_residual).copy() for _ in range(remaining)]
                    )
                    break

            totals = np.maximum(bids.sum(axis=0) + delta, 1e-12)
            x = capacities[None, :] * bids / totals[None, :]
            convergence_residual = convergence_history[-1]
            weighted_throughput = float(np.sum(valuations * x * requirements))
            eligible_values = np.where(requirements, valuations, -np.inf)
            resource_maxima = np.max(eligible_values, axis=0)
            resource_maxima[~np.isfinite(resource_maxima)] = 0.0
            max_weighted = float(np.sum(capacities * resource_maxima))
            total_utility = float(np.sum(
                valuations
                * _multiresource_alpha_fair_utility(x, alpha_matrix)
                * requirements
            ))
            jain_by_resource = []
            for resource in range(resource_count):
                active_x = x[requirements[:, resource], resource]
                denominator = active_x.size * np.square(active_x).sum()
                jain_by_resource.append(
                    float(active_x.sum() ** 2 / denominator) if denominator > 0 else np.nan
                )
            jain = float(np.nanmean(jain_by_resource))
            run_metrics.append((
                float(x.sum()), weighted_throughput / max_weighted, jain,
                total_utility,
                convergence_residual,
                float(convergence_residual <= convergence_tolerance),
            ))
            final_x.append(x)
            run_histories.append(convergence_history)
            run_resource_histories.append(resource_histories)
            run_metrics[-1] = run_metrics[-1] + (
                np.asarray(jain_by_resource), np.asarray(resource_histories[-1])
            )

        scalar_metrics = np.asarray([metrics[:6] for metrics in run_metrics], dtype=float)
        mean_metrics = np.mean(scalar_metrics, axis=0)
        records.append({
            "alpha": alpha_axis_value,
            "alpha_values": alpha_by_player.tolist(),
            "allocated_capacity": mean_metrics[0],
            "efficiency": mean_metrics[1],
            "jain": mean_metrics[2],
            "jain_by_resource": np.mean([m[6] for m in run_metrics], axis=0).tolist(),
            "total_utility": mean_metrics[3],
            "residual_mode": residual_mode,
            "convergence_residual": mean_metrics[4],
            "residual_by_resource": np.mean([m[7] for m in run_metrics], axis=0).tolist(),
            "converged_fraction": mean_metrics[5],
            "converged": bool(mean_metrics[5] == 1.0),
        })
        allocations[alpha_axis_value] = np.mean(final_x, axis=0)
        convergence_histories[alpha_axis_value] = np.mean(
            np.asarray(run_histories, dtype=float), axis=0
        )
        resource_convergence_histories[alpha_axis_value] = np.mean(
            np.asarray(run_resource_histories, dtype=float), axis=0
        )

    return (
        records, allocations, convergence_histories,
        resource_convergence_histories,
    )

# -----------------------
# PAGE CONFIG & HEADER
# -----------------------
st.set_page_config(
    page_title="EdgeKelly Lab",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {max-width: 1500px; padding-top: 1.5rem; padding-bottom: 4rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #e5e7eb;}
    [data-testid="stSidebar"] .block-container {padding-top: 1.25rem;}
    h1, h2, h3 {letter-spacing: -0.025em;}
    h2 {margin-top: 2.5rem; padding-top: .5rem; border-top: 1px solid #e5e7eb;}
    div[data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e5e7eb;
        border-radius: 12px; padding: .85rem 1rem;}
    div.stButton > button, div.stDownloadButton > button {border-radius: 9px; font-weight: 650;}
    .ek-hero {padding: 1.35rem 1.5rem; border-radius: 16px;
        background: linear-gradient(120deg, #0f172a, #164e63); color: white;
        margin: .5rem 0 1.25rem 0;}
    .ek-hero h1 {color: white; margin: 0 0 .35rem 0;}
    .ek-hero p {color: #dbeafe; margin: 0; font-size: 1.05rem;}
    .ek-note {border-left: 4px solid #0891b2; padding: .65rem 1rem;
        background: #ecfeff; border-radius: 0 8px 8px 0; color: #164e63;}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 5])
with col1:
    st.image("src/game/logo_avgn.png")  # Avignon Univ logo
with col2:
    st.image("src/game/cognetslogo1.png")  # Project logo

# --- Product header and navigation ---
st.markdown("""
<div class="ek-hero">
  <h1>EdgeKelly Lab</h1>
  <p>Interactive α-fair resource-allocation experiments for learning dynamics in Kelly games.</p>
</div>
""", unsafe_allow_html=True)

nav_cols = st.columns(3)
nav_cols[0].page_link("app.py", label="Simulator", icon="🧪")
nav_cols[1].page_link("pages/1_Experiment_Guide.py", label="Experiment guide", icon="🧭")
nav_cols[2].page_link("pages/2_Methodology.py", label="Methodology", icon="📘")

st.markdown(
    '<div class="ek-note"><strong>Start here:</strong> choose global parameters and learning methods '
    'in the sidebar, then run only the experiment you need below. Results remain available while '
    'you adjust presentation controls.</div>',
    unsafe_allow_html=True,
)


# -----------------------
# INTRO ANIMATION
# -----------------------
st.image(build_intro_animation())

# -----------------------
# SIDEBAR CONFIG
# -----------------------


# =========================================================
# ⚙️ SIDEBAR CONFIGURATION
# =========================================================
with st.sidebar:
    st.caption("EDGEKELLY LAB")
    st.header("Simulation setup")
    st.page_link("pages/1_Experiment_Guide.py", label="How to use the experiments", icon="🧭")
    st.page_link("pages/2_Methodology.py", label="Algorithms and metrics", icon="📘")

    # --- Copy default config ---
    cfg = dict(DEFAULT_CONFIG)

    # ------------------------
    # 📌 Basic parameters
    # ------------------------
    cfg["n"] = st.number_input("Players (n)", 2, 100, cfg["n"], step=1)

    cfg["T"] = st.number_input("Iterations (T)", 10, 100000, cfg["T"], step=10)
    cfg["T_plot"] = st.number_input(
        "Iterations to display (T_plot)",
        10,
        100000,
        min(int(cfg["T"]), int(cfg["T"])),
        step=10,
    )
    cfg["Nb_random_sim"] = st.number_input("Number of simulations", 1, 50, int(cfg["Nb_random_sim"]), step=1)
    cfg["alpha"] = st.selectbox("α (fairness)", [0, 1, 2], index=[0, 1, 2].index(cfg["alpha"]))
    cfg["eta"] = st.number_input(
        "Learning-rate scale (η)",
        1e-7,
        100.0,
        float(cfg["eta"]),
        step=0.1,
        format="%.7f",
        help="Applied automatically to every selected method that uses a learning rate.",
    )
    cfg["lr_vary"] = st.checkbox("Vary learning rate over time?", value=cfg["lr_vary"])
    cfg["Add_Zoom"] = st.checkbox("Add a Zoom, please select X_axis Zoom", value=False)
    if cfg["Add_Zoom"]:
        # Ensure config exists
        if "config" not in st.session_state:
            st.session_state.cfg = {
                "inset_rect": [0.55, 0.55, 0.40, 0.40],
            }



        st.write("### Zoom inset rectangle (in axes coordinates 0–1)")
        left = st.number_input(
            "Left",
            min_value=0.0, max_value=1.0,
            value=0.68, step=0.01,
            help="Horizontal position of the zoom box (0 = far left, 1 = far right)."
        )

        bottom = st.number_input(
            "Bottom",
            min_value=0.0, max_value=1.0,
            value=0.68, step=0.01,
            help="Vertical position of the zoom box (0 = bottom, 1 = top)."
        )

        width = st.number_input(
            "Width",
            min_value=0.0, max_value=1.0,
            value=0.30, step=0.01,
            help="Width of the zoom box relative to the full plot (0 to 1 scale)."
        )

        height = st.number_input(
            "Height",
            min_value=0.0, max_value=1.0,
            value=0.30, step=0.01,
            help="Height of the zoom box relative to the full plot (0 to 1 scale)."
        )

        cfg["inset_rect"] = [left, bottom, width, height]


    cfg["keep_initial_bid"] = st.checkbox("Keep same initial bid for all simulations?", value=False)

    # ------------------------
    # 💰 Game parameters
    # ------------------------
    cfg["price"] = st.number_input("Price (λ)", 1e-4, 1000.0, float(cfg["price"]), step=0.1, format="%.4f")
    cfg["a"] = st.number_input("a (utility scale)", 0.1, 1e6, float(cfg["a"]), step=10.0)
    cfg["gamma"] = st.number_input("γ (heterogeneity)", 0.0, cfg["a"], float(cfg["gamma"]), step=1.0)
    cfg["d_vector"] = st.number_input("d_i (service offset)", 0.0, 1e6, 0.0, step=10.0)

    # ------------------------
    # 🎯 Metric to visualize
    # ------------------------
    metrics_all = [
        "Relative_Efficienty_Loss","Avg_Payoff","Inst.Payoff", "epsilon_error",   "Speed", "epsilon_error_Hybrid",  "Bid", "Potential", "Pareto",
        "SW","Jain_Index", "LSW", "Dist_To_Optimum_SW", "Avg_Bid",  "Res_Payoff"
    ]
    cfg["metric"] = st.selectbox("Metric to plot", metrics_all, index=metrics_all.index(cfg["metric"]))

    cfg["Track"] = st.checkbox("Track the metric over time?", value=True)
    cfg["pltLegend"] = st.checkbox("Show plot legend", value=False)
    cfg["Random_set"] = st.checkbox("Random players' sets?", value=True)
    cfg["show_y_axis"] = st.checkbox("Show y-axis", value=True)


    # ------------------------
    # 🎲 Initial bids
    # ------------------------
    cfg["Random_Initial_Bid"] = st.checkbox("Random initial bids?", value=True)
    if not cfg["Random_Initial_Bid"]:
        cfg["var_init"] = st.number_input("Variance of initial bids", 0.0, 1e6, float(cfg["var_init"]), step=1.0)

    # ------------------------
    # ⚙️ Advanced parameters
    # ------------------------
    with st.expander("Advanced Parameters"):
        cfg["a_min"] = st.number_input("Minimum a_i", 0.1, 1e6, float(cfg["a_min"]), step=1.0)
        cfg["mu"] = st.number_input("μ (budget heterogeneity)", 0.0, float(cfg["c"]) / cfg["n"], float(cfg["mu"]), step=1.0)
        cfg["c"] = st.number_input("c (base budget)", 10.0, 1e6, float(cfg["c"]), step=10.0)
        cfg["delta"] = st.number_input("δ (slack)", 0.0, 10.0, float(cfg["delta"]), step=0.1)
        cfg["epsilon"] = st.number_input("ε (min bid)", 0.0, 100.0, float(cfg["epsilon"]), step=0.05)



        # Heterogeneity vectors
        cfg["a_vector"] = st.text_area(
            "List of heterogeneous a_i",
            value=str([max(cfg["a"] - cfg["gamma"] * i, cfg['a_min']) for i in range(cfg["n"])])
        )
        try:
            cfg["a_vector"] = ast.literal_eval(cfg["a_vector"])
        except Exception:
            st.error("Invalid format for a_vector.")

        # Ranges for tables
        cfg["list_n"] = [int(x) for x in st.text_area(
            "List of n values",
            value=", ".join(str(x) for x in DEFAULT_CONFIG_TABLE["list_n"])
        ).split(",") if x.strip()]
        cfg["list_gamma"] = [float(x) for x in st.text_area(
            "List of γ values",
            value=", ".join(str(x) for x in DEFAULT_CONFIG_TABLE["list_gamma"])
        ).split(",") if x.strip()]


        s_min = (cfg["n"] - 1) * cfg["epsilon"] + cfg["delta"]
        s_max = (cfg["n"] - 1) * cfg["c"] * torch.ones(1) + cfg["delta"]
        z_max = BR_alpha_fair(cfg["epsilon"] * torch.ones(1), cfg["c"] * torch.ones(1), cfg["c"] * torch.ones(1), s_min, torch.tensor(cfg["a_vector"]), cfg["delta"], cfg["alpha"], cfg["price"], b=0)
        x_max = z_max / (z_max + s_min)
        x_min = cfg["epsilon"] * torch.ones(1) / (cfg["epsilon"] * torch.ones(1) + s_max)
        x_min_2 = cfg["c"] * torch.ones(1) / (cfg["c"] * torch.ones(1) + s_max)

        Payoff_min = torch.min(Payoff(x_min, cfg["epsilon"] * torch.ones(1), torch.tensor(cfg["a_vector"]) , cfg["d_vector"], cfg["alpha"], cfg["price"]),
                               Payoff(x_min_2, cfg["c"] * torch.ones(1), torch.tensor(cfg["a_vector"]), cfg["d_vector"], cfg["alpha"], cfg["price"]))
        Payoff_max = torch.max(Payoff(x_max, z_max, torch.tensor(cfg["a_vector"]), cfg["d_vector"], cfg["alpha"], cfg["price"]))
       # print(Payoff_max , Payoff_min,float(cfg["tol"])/ float(Payoff_max - Payoff_min[0])) #/ float(Payoff_max - Payoff_min[0])
        cfg["tol"] = st.number_input("Tolerance", 1e-12, 1e-2, float(cfg["tol"]), step=1e-6)

    # ------------------------
    # 🧠 Learning methods
    # ------------------------
    lr_methods_all = ["DAQ_F", "DAQ_V", "OGD_F", "OGD_V", "BR", "DAE", "RRM_V", "Hybrid", "XL", "NumBR"]
    selected_methods = st.multiselect(
        "Select learning methods",
        lr_methods_all,
        default=[ "DAQ_F", "DAQ_V",  "RRM_V", "OGD_F", "OGD_V","BR"]
    )
    # ✅ If "Hybrid" is selected, keep only "Hybrid"
    if "Hybrid" in selected_methods:
        selected_methods = ["Hybrid"]
    cfg["lrMethods"] = selected_methods
    cfg["selected_methods"] = selected_methods

    # Use the single configured learning rate consistently. Comparing several
    # manual rates is an expert workflow and made the standard form ambiguous.
    cfg["Learning_rates"] = [cfg["eta"]] * len(selected_methods)
    DEFAULT_CONFIG["Learning_rates"] = list(cfg["Learning_rates"])
    LEGENDS = list(selected_methods)
    LEGENDS_Hybrid = []
    LEGENDS_Hybrid_full = []
    cfg["num_lrmethod"] = 0
    cfg["num_hybrids"] = 0
    cfg["num_hybrid_set"] = 0


    cfg["y_zoom_interval"] = [0.0,1.0]


    if "Hybrid" in selected_methods :
        st.info("You selected Hybrid. You can configure multiple hybrid algorithms below.")
        func_group  = []
        # Number of hybrids
        num_hybrids = st.number_input(
            "How many Hybrid algorithms do you want to configure?",
            min_value=1,
            max_value=cfg["n"]-1,
            value=1,
            step=1
        )
        if num_hybrids == 1:
            percent_A1 = st.number_input(
                "Select percentage of players in first subset (A₁)",
                min_value=1,
                max_value=99,
                value=50,
                step=1,
                # %format="%d%%",
                help="Defines the percentage of players assigned to A₁ in the hybrid group."
            )
            # Convert percentage to number of players
            cfg["Nb_A1"] = [max(1, int(cfg["n"] * percent_A1 / 100))]

        x_zoom_interval = st.slider(
            label="🔍 Select X-axis zoom interval",
            min_value=1,
            max_value=cfg["T_plot"],
            value=(1, cfg["T_plot"]+1),  # default: full x range
            step=1
        )
        cfg["x_zoom_interval"] = x_zoom_interval





        cfg["num_hybrids"] = num_hybrids
        hybrid_options = [m for m in lr_methods_all if m != "Hybrid"]
        num_hybrid_set = st.number_input(
            "How many Group Hybrid algorithms do you want to configure?",
            min_value=1,
            max_value=cfg["n"]-1,
            value=1,
            step=1
        )
        if num_hybrids>1:
            cfg["lrMethods"] = cfg["lrMethods"] + ["Hybrid"]*(num_hybrid_set*num_hybrids - 1)
        cfg["Hybrid_funcs_"] = []
        cfg["num_hybrid_set"] = num_hybrid_set

        if cfg["num_hybrids"] ==1  and cfg["num_hybrid_set"] >=1:
            x_zoom_interval = st.slider(
                label="🔍 Select X-axis zoom interval",
                min_value=0,
                max_value=cfg["T_plot"],
                value=(0, cfg["T_plot"]),  # default: full x range
                step=1
            )
            cfg["x_zoom_interval"] = x_zoom_interval

        #print(f"num_hybrid_set{num_hybrid_set}")
        for i in range(num_hybrid_set):
            method = st.multiselect(
                f"Select Hybrid funcs ",
                hybrid_options,
                default=["BR","DAQ_V"],
                key=f"hybrid_method_{i}"
            )
            #print(method)
            if method[1] not in func_group:
                func_group.append(method[1])
            cfg["Hybrid_funcs_"].append(method)

        #selected_methods =  [m for m in selected_methods if m != "Hybrid"]

        h_idx = 1
        # Initialise la liste des k si pas déjà définie
        if "Nb_A1" not in cfg:
            cfg["Nb_A1"] = []
        else:
            cfg["Nb_A1"] = cfg["Nb_A1"]

        cfg["Hybrid_sets"] =[]
        cfg["Hybrid_funcs"] = []

        for secMeth in range(num_hybrid_set):
            cfg["Learning_rates"] = cfg["Learning_rates"] + [cfg["eta"]] * num_hybrids

            sets = []
            if num_hybrids > 1:
                cfg["Nb_A1"] += list(range(1, num_hybrids + 1))
            LEGENDS_Hybrid.append(cfg["Hybrid_funcs_"][secMeth][1])#+rf" -- $\eta={cfg["eta"]}$")
            #LEGENDS_Hybrid.append(cfg["Hybrid_funcs_"][secMeth][1] + rf" -- $\eta={cfg["eta"]}$")
            sets = []  # contiendra la liste finale de [subset, remaining]
            kk = 0
            h_idx = 0

            for h in cfg["Nb_A1"][:num_hybrids]:

                cfg["Hybrid_funcs"].append(cfg["Hybrid_funcs_"][secMeth])
                h_idx += 1

                # --- Construire la première liste : [0, ... autres sauf 1] ---
                # candidats possibles : tous sauf 0 et 1, car 0 sera ajouté manuellement et 1 exclu
                [subset, remaining] = make_subset(cfg["n"], h)

                kk += 1
                LEGENDS_Hybrid_full.append(f"({cfg["Hybrid_funcs_"][secMeth][0]}: {h}, {cfg["Hybrid_funcs_"][secMeth][1]}: {cfg["n"] - h})")
                cfg["Hybrid_sets"].append([subset, remaining])
    else:
        x_zoom_interval = st.slider(
            label="🔍 Select X-axis zoom interval",
            min_value=0,
            max_value=cfg["T"],
            value=(0, cfg["T"]),  # default: full x range
            step=1
        )
        cfg["x_zoom_interval"] = x_zoom_interval
    LEGENDS = LEGENDS_Hybrid + LEGENDS
    cfg["LEGENDS"]=LEGENDS



    cfg["Players2See"] = list(range(0, 1))

    if cfg["metric"] in ["Inst.Payoff", "Avg_Payoff", "Bid", "Avg_Bid"]:
        cfg["Players2See"] =  st.text_area(
            "List of players to display metrics",
            value=", ".join(str(x) for x in cfg.get("Players2See", cfg["Players2See"])),
            help="Comma-separated list of γ (a_i heterogeneity) values."
        )
        # Convert input string to list of floats
        try:
            cfg["Players2See"] = [int(x.strip()) for x in cfg["Players2See"].split(",") if x.strip()]
        except:
            st.error("Invalid format for Players to See, please enter numbers separated by commas.")

    cfg["ylog_scale"] = st.sidebar.checkbox("Y log scale", value=cfg["ylog_scale"])
    cfg["plot_step"] = st.number_input("Plot step", 1, 1000, int(cfg["plot_step"]), step=1)

    cfg["pltText"] = st.sidebar.checkbox("Display values", value=cfg["pltText"])


    st.sidebar.download_button("⬇️ Download config JSON", data=json.dumps(cfg, indent=2),
                               file_name="config.json", mime="application/json")


# -----------------------
# DESCRIPTION & FORMULA
# -----------------------
selected_algo = st.selectbox("Choose algorithm to describe", list(ALGO_DESCRIPTIONS.keys()))
if selected_algo != "None":
    with st.expander(f"Description of {selected_algo}", expanded=False):
        st.code(ALGO_DESCRIPTIONS[selected_algo], language='python')

if st.checkbox("Show Formulations"):
    st.latex(r"""
    \varphi_i^{\alpha}(x_i) =
    \begin{cases}
    
    a_i\frac{x_i^{1-\alpha}}{1-\alpha} - \lambda z_i, & \alpha \neq 1 \\
    a_i\log(x_i) - \lambda z_i, & \alpha = 1
    \end{cases}
    \quad , \quad
    a_i = a - i\gamma
    \quad, \quad x_i =\frac{z_i}{\sum_{j=1}^n z_j + \delta}
    """)

    st.markdown(r"""
    **Where:**  
    - \($x_i$\)  allocated resource share for player $i$  
    - \($\alpha \ge 0$\) is the fairness parameter  
    - \($a$\) is the base utility scale  
    - \($\gamma \ge 0$\) controls heterogeneity across players
    - \($\lambda$\) the price
    """)


# Checkbox to show metrics info
if st.checkbox("ℹ️ Show information about metrics"):
    st.markdown("""
    ### 📊 Metrics Used in the Simulator  

    - **[Social Welfare (SW)](https://en.wikipedia.org/wiki/Social_welfare_function)**:  
      The aggregate efficiency of the allocation, defined as the sum of agents’ utilities at each iteration.  

    - **Distance to Optimum Social Welfare ($\\text{Dist2SW}^*$):**  
      Since the SW maximization is a concave optimization problem, we solve the KKT conditions via a bisection algorithm to obtain the optimal $\\text{SW}^*$.  
      The distance is:  
      $$
      \\text{Dist2SW}^*(\\mathbf{z}) = \\big| \\text{SW}^* - \\text{SW}(\\mathbf{z}) \\big|
      $$  

    - **Speed:**  
      A performance indicator based on the **Convergence Residual**, defined as the $\\ell_2$-distance  
      $$
      \\|\\text{BR}(\\mathbf{z}(t)) - \\mathbf{z}(t)\\|_2,
      $$  
      which measures how close the system is to a Nash equilibrium.  
      This value decreases as the algorithm converges, and thresholds below $10^{-5}$ are typically treated as equilibrium.
    
    - **[Utility](https://en.wikipedia.org/wiki/Utility)**: A measure of individual satisfaction or payoff.  
    - **[Bid](https://en.wikipedia.org/wiki/Auction)**: The amount an agent submits as demand for resources.  


    - **Average Bids:**  
      The long-run time-averaged bid per agent, as an indicator demand and budget usage.  
      $$
      \\frac{1}{T} \\sum_{t=1}^T \\mathbf{z}_i(t)),
      $$  

    - **Average Utility:**  
      The long-run time-averaged utility per agent:  
      $$
      \\frac{1}{T} \\sum_{t=1}^T \\varphi_i(\\mathbf{z}(t)),
      $$  
      which highlights fairness and satisfaction across players.  
    """)
# -----------------------
# RUN SIMULATION
# -----------------------
st.header("1. Learning-dynamics simulation")
st.caption(
    "Compare selected learning rules over time for the metric chosen in the sidebar."
)
with st.expander("Why run this?"):
    st.write("Running the simulation computes the equilibrium for each learning method...")

if st.button("▶️ Run Simulation"):
    # Conteneur pour le chrono
    st.write("""
        This section runs the simulation:
        1. Initialize bids for all selected learning methods  
        2. Evaluate the chosen performance metric  
        3. Repeat steps (1–2) for the specified number of simulations  
        4. Compute the average metric values across all runs  
        """)

    chrono_placeholder = st.empty()

    start_time = time.time()

    with st.spinner("Simulating..."):
        # --- Chrono en temps réel ---
        while True:
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
       #     chrono_placeholder.info(f"⏱️ Elapsed time: {minutes:02d}:{seconds:02d}")

            # Ici tu lances la simulation
            runner = SimulationRunner(cfg)
            results = runner.run_simulation()
            break  # on sort de la boucle une fois la simulation finie

    # --- Stop chrono ---
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)

    # Stocker les résultats dans la session
    st.session_state.results = results
    st.session_state.config = cfg

    # Affichage final
    chrono_placeholder.success(f"✅ Simulation finished in {minutes:02d}:{seconds:02d}")

    # -----------------------
    # PLOTLY VISUALISATION
    # -----------------------

def convert_results_to_csv(results):
    # Fonction pour convertir les résultats en CSV
    # Implémentation simplifiée
    return "Simulation,Results,Would,Be,Here\n1,2,3,4,5"


def parse_positive_float_list(raw_text, min_value=None, inclusive_min=False):
    values = []
    for item in raw_text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if min_value is not None:
            invalid = value < min_value if inclusive_min else value <= min_value
            if invalid:
                comparator = "greater than or equal to" if inclusive_min else "greater than"
                raise ValueError(f"All values must be {comparator} {min_value}.")
        values.append(value)
    if not values:
        raise ValueError("At least one value is required.")
    return values


def compute_lpoa_for_mu_epsilon(config, mu_values, epsilon_values, algorithm="BR", progress_bar=None):
    lpoa_results = {
        "algorithm": algorithm,
        "T": int(config["T"]),
        "mu_values": list(mu_values),
        "epsilon_values": list(epsilon_values),
        "series": {
            "Nash_LPoA": {},
            "Jain_Index_NE": {},
            "SW_Loss_pct": {},
        },
        "time_series": {
            "Nash_LPoA": {},
            "Jain_Index_NE": {},
            "SW_Loss_pct": {},
        },
        "rows": [],
    }
    total_runs = max(1, len(mu_values) * len(epsilon_values))
    run_idx = 0

    for epsilon in epsilon_values:
        epsilon_key = float(epsilon)
        for metric_series in lpoa_results["series"].values():
            metric_series[epsilon_key] = []

        for mu in mu_values:
            run_idx += 1
            local_cfg = dict(config)
            local_cfg["mu"] = float(mu)
            local_cfg["epsilon"] = float(epsilon)
            local_cfg["lrMethods"] = [algorithm]
            local_cfg["Learning_rates"] = [config.get("eta", 0.1)]
            local_cfg["num_hybrids"] = 0
            local_cfg["num_hybrid_set"] = 0
            local_cfg["Hybrid_funcs"] = []
            local_cfg["Hybrid_sets"] = []

            n = int(local_cfg["n"])
            c = float(local_cfg["c"])
            delta = float(local_cfg["delta"])
            price = float(local_cfg["price"])
            alpha = float(local_cfg["alpha"])
            tol = float(local_cfg["tol"])
            T = int(local_cfg["T"])
            eta = float(local_cfg.get("eta", 0.1))
            lr_vary = bool(local_cfg.get("lr_vary", False))

            if epsilon >= c:
                raise ValueError("Each epsilon value must be smaller than c.")

            eps = torch.tensor(float(epsilon), dtype=torch.float64)
            a_vector = torch.tensor(local_cfg["a_vector"], dtype=torch.float64)
            c_vector = torch.tensor([max(c - i * float(mu), float(epsilon)) for i in range(n)], dtype=torch.float64)
            bid0 = (c - float(epsilon)) * torch.rand(n, dtype=torch.float64) + float(epsilon)

            dmin = a_vector * torch.log((eps + torch.sum(c_vector) - c_vector + delta) / eps)
            d_vector = 0.7 * dmin * 0

            x_opt = x_log_opt(c_vector, a_vector, d_vector, eps, delta, price, bid0, alpha)
            lsw_opt = LSW_func(x_opt, c_vector, a_vector, d_vector, alpha)
            sw_opt = SW_func(x_opt, c_vector, a_vector, d_vector, alpha)

            z_br = bid0.clone()
            residual = torch.tensor(float("inf"), dtype=torch.float64)
            br_iterations = T
            curve_key = f"epsilon={float(epsilon):g}, mu={float(mu):g}"
            lpoa_results["time_series"]["Nash_LPoA"][curve_key] = {"tau": [0], "values": []}
            lpoa_results["time_series"]["Jain_Index_NE"][curve_key] = {"tau": [0], "values": []}
            lpoa_results["time_series"]["SW_Loss_pct"][curve_key] = {"tau": [0], "values": []}
            game_set = GameKelly(n, price, eps, delta, alpha, tol)
            if not hasattr(game_set, algorithm):
                raise ValueError(f"Algorithm {algorithm} is not available in GameKelly.")
            update_method = getattr(game_set, algorithm)
            acc_grad = torch.zeros(n, dtype=torch.float64)

            x_initial = z_br / (torch.sum(z_br) + delta)
            lsw_initial = LSW_func(x_initial, c_vector, a_vector, d_vector, alpha)
            sw_initial = SW_func(x_initial, c_vector, a_vector, d_vector, alpha)
            jain_initial = torch.sum(z_br) ** 2 / (n * torch.sum(z_br ** 2))
            lpoa_initial = torch.nan
            if torch.abs(lsw_initial) > torch.tensor(1e-12, dtype=torch.float64):
                lpoa_initial = lsw_opt / lsw_initial
            sw_loss_initial = torch.nan
            if torch.abs(sw_opt) > torch.tensor(1e-12, dtype=torch.float64):
                sw_loss_initial = torch.abs((sw_opt - sw_initial) / sw_opt) #* 100
            lpoa_results["time_series"]["Nash_LPoA"][curve_key]["values"].append(float(lpoa_initial.detach().cpu().numpy()))
            lpoa_results["time_series"]["Jain_Index_NE"][curve_key]["values"].append(float(jain_initial.detach().cpu().numpy()))
            lpoa_results["time_series"]["SW_Loss_pct"][curve_key]["values"].append(float(sw_loss_initial.detach().cpu().numpy()))

            for iteration in range(1, T + 1):
                p = torch.sum(z_br) - z_br + delta
                if algorithm == "BR" and alpha not in [0, 1, 2]:
                    z_next = solve_nonlinear_eq(a_vector, p, alpha, eps, c_vector, price, max_iter=1000, tol=tol)
                    z_next = Q1(z_next, eps, c_vector, price)
                elif algorithm == "BR":
                    z_next = BR_alpha_fair(eps, c_vector, z_br, p, a_vector, delta, alpha, price, b=0)
                else:
                    z_next, acc_grad = update_method(
                        iteration,
                        a_vector,
                        c_vector,
                        d_vector,
                        eta,
                        z_br,
                        acc_grad,
                        p=p,
                        vary=lr_vary,
                    )
                z_next = z_next.to(dtype=torch.float64)
                z_br = z_next
                residual = game_set.check_NE(z_br, a_vector, c_vector, d_vector).to(dtype=torch.float64)

                x_iter = z_br / (torch.sum(z_br) + delta)
                lsw_iter = LSW_func(x_iter, c_vector, a_vector, d_vector, alpha)
                sw_iter = SW_func(x_iter, c_vector, a_vector, d_vector, alpha)
                jain_iter = torch.sum(z_br) ** 2 / (n * torch.sum(z_br ** 2))
                lpoa_iter = torch.nan
                if torch.abs(lsw_iter) > torch.tensor(1e-12, dtype=torch.float64):
                    lpoa_iter = lsw_opt / lsw_iter
                sw_loss_iter = torch.nan
                if torch.abs(sw_opt) > torch.tensor(1e-12, dtype=torch.float64):
                    sw_loss_iter = torch.abs((sw_opt - sw_iter) / sw_opt) * 100
                for metric_key, value in [
                    ("Nash_LPoA", lpoa_iter),
                    ("Jain_Index_NE", jain_iter),
                    ("SW_Loss_pct", sw_loss_iter),
                ]:
                    lpoa_results["time_series"][metric_key][curve_key]["tau"].append(iteration)
                    lpoa_results["time_series"][metric_key][curve_key]["values"].append(float(value.detach().cpu().numpy()))

                if residual <= tol:
                    br_iterations = iteration
                    break

            x_br = z_br / (torch.sum(z_br) + delta)
            lsw_br = LSW_func(x_br, c_vector, a_vector, d_vector, alpha)
            sw_br = SW_func(x_br, c_vector, a_vector, d_vector, alpha)
            jain_index_ne = torch.sum(z_br) ** 2 / (n * torch.sum(z_br ** 2))
            lpoa = torch.nan
            if torch.abs(lsw_br) > torch.tensor(1e-12, dtype=torch.float64):
                lpoa = lsw_opt / lsw_br
            sw_loss = torch.nan
            if torch.abs(sw_opt) > torch.tensor(1e-12, dtype=torch.float64):
                sw_loss = torch.abs((sw_opt - sw_br) / sw_opt) * 100
            is_nash = bool(residual <= tol)
            lpoa_value = float(lpoa.detach().cpu().numpy())
            jain_value = float(jain_index_ne.detach().cpu().numpy())
            sw_loss_value = float(sw_loss.detach().cpu().numpy())

            row = {
                "epsilon": float(epsilon),
                "mu": float(mu),
                "Nash_LPoA": lpoa_value if is_nash else np.nan,
                "Last_iterate_LPoA": lpoa_value,
                "Jain_Index_NE": jain_value if is_nash else np.nan,
                "Last_iterate_Jain_Index": jain_value,
                "SW_Loss_pct": sw_loss_value if is_nash else np.nan,
                "Last_iterate_SW_Loss_pct": sw_loss_value,
                "LSW_opt": float(lsw_opt.detach().cpu().numpy()),
                "LSW_BR": float(lsw_br.detach().cpu().numpy()),
                "SW_opt": float(sw_opt.detach().cpu().numpy()),
                "SW_BR": float(sw_br.detach().cpu().numpy()),
                "Nash_residual": float(residual.detach().cpu().numpy()),
                "Nash_tol": tol,
                "Algorithm": algorithm,
                "Iterations": br_iterations,
                "Is_Nash": is_nash,
            }
            lpoa_results["rows"].append(row)
            lpoa_results["series"]["Nash_LPoA"][epsilon_key].append(row["Nash_LPoA"])
            lpoa_results["series"]["Jain_Index_NE"][epsilon_key].append(row["Jain_Index_NE"])
            lpoa_results["series"]["SW_Loss_pct"][epsilon_key].append(row["SW_Loss_pct"])

            if progress_bar is not None:
                progress_bar.progress(run_idx / total_runs)

    return lpoa_results


def compute_convergence_iterations_vs_mu(config, mu_values, progress_bar=None):
    rows = []
    series = {}
    total_runs = max(1, len(mu_values))

    for run_idx, mu in enumerate(mu_values, start=1):
        local_cfg = dict(config)
        local_cfg["mu"] = float(mu)

        runner = SimulationRunner(local_cfg)
        results = runner.run_simulation()

        for algorithm, metrics in results.get("methods", {}).items():
            convergence_iter = float(metrics.get("convergence_iter", local_cfg["T"]))
            final_residual = np.nan
            speed_values = np.asarray(metrics.get("Speed", []))
            if speed_values.size:
                final_residual = float(speed_values[-1])

            converged = bool(convergence_iter < int(local_cfg["T"]) or final_residual <= float(local_cfg["tol"]))
            row = {
                "mu": float(mu),
                "Algorithm": algorithm,
                "Iterations to converge": convergence_iter if converged else np.nan,
                "Final residual": final_residual,
                "Converged": converged,
            }
            rows.append(row)
            series.setdefault(algorithm, []).append(row["Iterations to converge"])

        if progress_bar is not None:
            progress_bar.progress(run_idx / total_runs)

    return {
        "mu_values": [float(mu) for mu in mu_values],
        "algorithms": list(series.keys()),
        "series": series,
        "rows": rows,
        "T": int(config["T"]),
        "tol": float(config["tol"]),
    }


def build_convergence_matrix(convergence_results):
    rows = convergence_results["rows"]
    algorithms = convergence_results["algorithms"]
    mu_values = convergence_results["mu_values"]
    values = []

    for algorithm in algorithms:
        algorithm_values = []
        for mu in mu_values:
            matching_row = next(
                row for row in rows
                if row["Algorithm"] == algorithm and np.isclose(row["mu"], mu)
            )
            algorithm_values.append(matching_row["Iterations to converge"])
        values.append(algorithm_values)

    return algorithms, mu_values, values


def save_convergence_mu_pdf(config, convergence_results):
    import matplotlib.pyplot as plt

    algorithms, mu_values, matrix = build_convergence_matrix(convergence_results)
    values = np.asarray(matrix, dtype=float)
    plot_values = np.ma.masked_invalid(values)
    figpath = f"{figure_prefix(config, metric='Convergence_Iterations_Mu')}_plot.pdf"

    fig, ax = plt.subplots(figsize=(14, max(5, 0.55 * len(algorithms))))
    image = ax.imshow(plot_values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(mu_values)))
    ax.set_xticklabels([f"{mu:g}" for mu in mu_values], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(algorithms)))
    ax.set_yticklabels(algorithms)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("Algorithm")
    ax.set_title(r"Iterations to converge ($\mathrm{residual} \leq \mathrm{tol}$)")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            label = "NC" if np.isnan(value) else f"{value:.0f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="white", fontweight="bold")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Iterations")
    fig.tight_layout()
    Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return figpath


def save_convergence_mu_log_pdf(
    config,
    convergence_results,
    fontsize=40,
    markersize=40,
    linewidth=12,
    linestyle="-",
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    algorithms, mu_values, matrix = build_convergence_matrix(convergence_results)
    values = np.asarray(matrix, dtype=float)
    figpath = f"{figure_prefix(config, metric='Convergence_Iterations_Mu_Log')}_plot.pdf"
    legend_path = f"{figure_prefix(config, metric='Convergence_Iterations_Mu_Log')}_legend.pdf"

    fig, ax = plt.subplots(figsize=JOURNAL_FIGSIZE)
    plt.rcParams.update({"font.size": fontsize})
    algorithms_plot = [legend_map.get(algorithm, algorithm) for algorithm in algorithms]
    annotation_indices = sorted({0, len(mu_values) // 2, len(mu_values) - 1})
    legend_handles = []
    legend_labels = []

    for algorithm_index, algorithm in enumerate(algorithms_plot):
        color = COLORS_METHODS[algorithm] if algorithm in METHODS else colors[algorithm_index % len(colors)]
        marker = MARKERS_METHODS[algorithm] if algorithm in METHODS else markers[algorithm_index % len(markers)]
        line, = ax.plot(
            mu_values,
            values[algorithm_index],
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
            markeredgecolor="black",
            color=color,
            label=algorithm,
        )
        legend_handles.append(Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            markersize=markersize,
            markeredgecolor="black",
            linestyle=linestyle,
            linewidth=linewidth,
        ))
        legend_labels.append(algorithm)

        for annotation_index in annotation_indices:
            value = values[algorithm_index, annotation_index]
            if np.isnan(value):
                continue
            label = f"{value:.0f}"
            ax.text(
                mu_values[annotation_index],
                value,
                label,
                fontsize=1.2*fontsize,
                fontweight="bold",
                color="black",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                ha="center",
                va="bottom",
            )

    ax.set_yscale("log")
    ax.set_xlabel(r"$\mu$", fontsize=2 * fontsize, )
    #ax.set_ylabel("Iterations", fontsize=2 * fontsize,)
    ax.tick_params(axis="both", which="major", labelsize=1.5 * fontsize)
    ax.tick_params(axis="both", which="minor", labelsize=1.2 * fontsize)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    Path(figpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.close(fig)

    legend_fig = plt.figure(figsize=(10, 2))
    legend_fig.legend(
        legend_handles,
        legend_labels,
        loc="center",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        prop={"weight": "bold", "size": 0.9 * fontsize},
        ncol=max(1, len(algorithms_plot)),
    )
    legend_fig.savefig(legend_path, format="pdf", bbox_inches="tight")
    plt.close(legend_fig)
    return figpath, legend_path


def get_lpoa_metric_specs():
    return {
        "Nash_LPoA": {
            "title": "Verified Nash Liquid Price of Anarchy vs budget heterogeneity",
            "yaxis": "Nash LPoA",
            "ylabel": r"$\mathrm{Nash\ LPoA}$",
            "filename": "LPoA",
        },
        "Jain_Index_NE": {
            "title": "Jain index of the Nash bid vs budget heterogeneity",
            "yaxis": "Jain index",
            "ylabel": r"$\mathrm{Jain}(z^{NE})$",
            "filename": "Jain_Index_NE",
        },
        "SW_Loss_pct": {
            "title": "Loss at Nash vs budget heterogeneity",
            "yaxis": "Loss",
            "ylabel": "Loss",
            "filename": "Loss",
            "scientific_percent": True,
        },
    }


def apply_scientific_yaxis(ax, fontsize=48):
    import matplotlib.ticker as mticker

    formatter = mticker.ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_fontsize(fontsize)
    offset_text.set_fontweight("bold")


def save_lpoa_metric_plot_pdf(config, lpoa_results, metric_key):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    metric_specs = get_lpoa_metric_specs()
    metric_spec = metric_specs[metric_key]
    algorithm = lpoa_results.get("algorithm", "BR")
    filename_metric = f"{metric_spec['filename']}_{algorithm}"
    figpath_plot = f"{figure_prefix(config, metric=filename_metric)}_plot.pdf"
    figpath_legend = f"{figure_prefix(config, metric=filename_metric)}_legend.pdf"
    plt.figure(figsize=(18, 12))
    plt.rcParams.update({"font.size": 40})

    markers_lpoa = ["o", "s", "D", "^", "v", "P", "X", "*"]
    colors_lpoa = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    legend_handles = []
    legend_labels = []
    for idx, (epsilon, values) in enumerate(lpoa_results["series"][metric_key].items()):
        color = colors_lpoa[idx % len(colors_lpoa)]
        marker = markers_lpoa[idx % len(markers_lpoa)]
        plt.plot(
            lpoa_results["mu_values"],
            values,
            linestyle="-",
            linewidth=12,
            marker=marker,
            markersize=40,
            markeredgecolor="black",
            color=color,
        )
        legend_handles.append(Line2D(
            [0], [0],
            color=color,
            marker=marker,
            markeredgecolor="black",
            linestyle="-",
            linewidth=12,
            markersize=40,
        ))
        legend_labels.append(rf"$\epsilon={epsilon:g}$")

    if config.get("ylog_scale", False):
        plt.yscale("log")
    if metric_spec.get("scientific_percent", False):
        ax = plt.gca()
        apply_scientific_yaxis(ax, fontsize=48)
        plt.text(
            0.0,
            1.02,
            "(%)",
            transform=ax.transAxes,
            fontsize=48,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    plt.xlabel(r"$\mu$", fontsize=55)
    plt.ylabel(metric_spec["ylabel"], fontsize=55)
    plt.xticks(fontsize=48, fontweight="bold")
    plt.yticks(fontsize=48, fontweight="bold")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figpath_plot, format="pdf", bbox_inches="tight")
    plt.close()

    fig_legend = plt.figure(figsize=(12, 2))
    ax_leg = fig_legend.add_subplot(111)
    ax_leg.axis("off")
    ax_leg.legend(
        legend_handles,
        legend_labels,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        prop={"weight": "bold", "size": 35},
        ncol=max(1, len(legend_labels)),
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
    )
    fig_legend.savefig(figpath_legend, format="pdf", bbox_inches="tight")
    plt.close(fig_legend)

    return figpath_plot, figpath_legend


def save_lpoa_time_plot_pdf(config, lpoa_results, metric_key, max_tau=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    metric_specs = get_lpoa_metric_specs()
    metric_spec = metric_specs[metric_key]
    algorithm = lpoa_results.get("algorithm", "BR")
    filename_metric = f"{metric_spec['filename']}_{algorithm}"
    figpath_plot = f"{figure_prefix(config, metric=filename_metric, suffix='time')}_time_plot.pdf"
    figpath_legend = f"{figure_prefix(config, metric=filename_metric, suffix='time')}_time_legend.pdf"
    plt.figure(figsize=(18, 12))
    plt.rcParams.update({"font.size": 40})

    markers_time = ["o", "s", "D", "^", "v", "P", "X", "*"]
    colors_time = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    legend_handles = []
    legend_labels = []
    for idx, (curve_label, curve_data) in enumerate(lpoa_results["time_series"][metric_key].items()):
        tau_values = np.asarray(curve_data["tau"])
        metric_values = np.asarray(curve_data["values"])
        if max_tau is not None:
            keep_mask = tau_values <= max_tau
            tau_values = tau_values[keep_mask]
            metric_values = metric_values[keep_mask]
        color = colors_time[idx % len(colors_time)]
        marker = markers_time[idx % len(markers_time)]
        plt.plot(
            tau_values,
            metric_values,
            linestyle="-",
            linewidth=8,
            marker=marker,
            markersize=24,
            markeredgecolor="black",
            color=color,
        )
        legend_handles.append(Line2D(
            [0], [0],
            color=color,
            marker=marker,
            markeredgecolor="black",
            linestyle="-",
            linewidth=8,
            markersize=24,
        ))
        legend_labels.append(curve_label)

    if config.get("ylog_scale", False):
        plt.yscale("log")
    if metric_spec.get("scientific_percent", False):
        ax = plt.gca()
        apply_scientific_yaxis(ax, fontsize=48)
        plt.text(
            0.0,
            1.02,
            "(%)",
            transform=ax.transAxes,
            fontsize=48,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    plt.xlabel(r"Iteration ($t$)", fontsize=55)
    plt.ylabel(metric_spec["ylabel"], fontsize=55)
    plt.xticks(fontsize=48, fontweight="bold")
    plt.yticks(fontsize=48, fontweight="bold")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figpath_plot, format="pdf", bbox_inches="tight")
    plt.close()

    fig_legend = plt.figure(figsize=(18, 3))
    ax_leg = fig_legend.add_subplot(111)
    ax_leg.axis("off")
    ax_leg.legend(
        legend_handles,
        legend_labels,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        prop={"weight": "bold", "size": 26},
        ncol=min(3, max(1, len(legend_labels))),
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
    )
    fig_legend.savefig(figpath_legend, format="pdf", bbox_inches="tight")
    plt.close(fig_legend)

    return figpath_plot, figpath_legend


def method_plot_color(method, fallback_index):
    if method in METHODS:
        return COLORS_METHODS[method]
    return colors[fallback_index % len(colors)]


def selected_player_indices(config, data):
    n_players = np.asarray(data).shape[1]
    raw_players = config.get("Players2See", list(range(n_players)))
    return [player for player in raw_players if 0 <= player < n_players]


def add_bid_traces(fig, x_data, y_data, legends, config):
    player_markers = [
        "circle", "square", "diamond", "triangle-up", "triangle-down",
        "pentagon", "star", "x", "cross"
    ]
    player_dashes = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

    for method_index, (data, legend) in enumerate(zip(y_data, legends)):
        bid_data = np.asarray(data)
        if bid_data.ndim != 2:
            continue

        x_plot = x_data[::config["plot_step"]]
        players_to_plot = selected_player_indices(config, bid_data)
        method_color = method_plot_color(legend, method_index)

        for player in players_to_plot:
            y_plot = bid_data[::config["plot_step"], player]
            initial_bid = bid_data[0, player]
            final_bid = bid_data[-1, player]
            player_label = f"Player {player + 1}"

            fig.add_trace(go.Scatter(
                x=x_plot,
                y=y_plot,
                mode="lines",
                name=f"{legend} - {player_label}",
                legendgroup=f"{legend}-{player}",
                line=dict(
                    color=method_color,
                    width=3,
                    dash=player_dashes[player % len(player_dashes)],
                ),
                opacity=0.75,
                customdata=np.column_stack([
                    np.full(len(x_plot), legend),
                    np.full(len(x_plot), player + 1),
                ]),
                hovertemplate=(
                    "Algorithm=%{customdata[0]}<br>"
                    "Player=%{customdata[1]}<br>"
                    "t=%{x}<br>"
                    "Bid=%{y:.6g}<extra></extra>"
                ),
            ))

            fig.add_trace(go.Scatter(
                x=[x_data[0]],
                y=[initial_bid],
                mode="markers+text",
                name=f"{legend} - {player_label} initial",
                legendgroup=f"{legend}-{player}",
                showlegend=False,
                marker=dict(
                    color=method_color,
                    symbol=player_markers[player % len(player_markers)],
                    size=13,
                    line=dict(color="black", width=1.5),
                ),
                text=["Initial"],
                textposition="top center",
                hovertemplate=(
                    f"Algorithm={legend}<br>{player_label}<br>"
                    "Initial bid=%{y:.6g}<extra></extra>"
                ),
            ))

            fig.add_trace(go.Scatter(
                x=[x_data[-1]],
                y=[final_bid],
                mode="markers+text",
                name=f"{legend} - {player_label} final",
                legendgroup=f"{legend}-{player}",
                showlegend=False,
                marker=dict(
                    color=method_color,
                    symbol="star",
                    size=16,
                    line=dict(color="black", width=1.5),
                ),
                text=["Final"],
                textposition="top center",
                hovertemplate=(
                    f"Algorithm={legend}<br>{player_label}<br>"
                    "Final bid=%{y:.6g}<extra></extra>"
                ),
            ))


def build_bid_summary_table(y_data, legends, all_players=False):
    rows = []
    for data, legend in zip(y_data, legends):
        bid_data = np.asarray(data)
        if bid_data.ndim != 2:
            continue

        if all_players:
            players = range(bid_data.shape[1])
        else:
            players = selected_player_indices(config, bid_data)

        for player in players:
            initial_bid = float(bid_data[0, player])
            final_bid = float(bid_data[-1, player])
            bid_change = final_bid - initial_bid
            relative_change = np.nan
            if abs(initial_bid) > 1e-12:
                relative_change = 100 * bid_change / initial_bid

            rows.append({
                "Algorithm": legend,
                "Player": player + 1,
                "Initial bid": initial_bid,
                "Final bid": final_bid,
                "Change": bid_change,
                "Change (%)": relative_change,
            })
    return rows


def build_bid_comparison_table(y_data, legends):
    rows_by_player = {}
    for data, legend in zip(y_data, legends):
        bid_data = np.asarray(data)
        if bid_data.ndim != 2:
            continue

        for player in range(bid_data.shape[1]):
            initial_bid = float(bid_data[0, player])
            final_bid = float(bid_data[-1, player])
            bid_change = final_bid - initial_bid
            row = rows_by_player.setdefault(player, {"Player": player + 1})
            row[f"{legend} initial"] = initial_bid
            row[f"{legend} final"] = final_bid
            row[f"{legend} change"] = bid_change

    return [rows_by_player[player] for player in sorted(rows_by_player)]


# =========================================================
# 📊 DISPLAY RESULTS (if available)
# =========================================================
#try:
if 'results' in st.session_state:
    # --- Retrieve data from session ---
    results = st.session_state.results
    config = st.session_state.config

    st.header("📈 Simulation Results")


    # --- Summary metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Players (n)", config['n'])
    with col2:
        st.metric("Iterations (T)", config['T'])
    with col3:
        st.metric("Fairness (α)", config['alpha'])

    # =====================================================
    # 📘 Step 1: Prepare data
    # =====================================================
    x_data = np.arange(config["T"])
    y_data, legends = [], []
    y_data_avg = []


    for method in LEGENDS:
        if method!= "Hybrid" and cfg["num_hybrid_set"] <1:
            y_data.append(results['methods'][method][cfg["metric"]])
            if cfg["metric"] in ["Inst.Payoff"]:
                y_data_avg.append(results['methods'][method]["Avg_Payoff"])
            elif cfg["metric"] in ["Bid"]:
                y_data_avg.append(results['methods'][method]["Avg_Bid"])
        elif cfg["num_hybrids"] ==1  and cfg["num_hybrid_set"] >=1:
            y_data.append(results['methods']["Hybrid"][cfg["metric"]])
            #print(y_data)
            if cfg["metric"] in ["Inst.Payoff"]:
                y_data_avg.append(results['methods']["Hybrid"]["Avg_Payoff"])
            elif cfg["metric"] in ["Bid"]:
                y_data_avg.append(results['methods']["Hybrid"]["Inst.Payoff"])
        legends.append(method)
    if cfg["num_hybrids"] >1  and cfg["num_hybrid_set"] >=1:
        y_data_alpha = []
        for hybrid_meth in LEGENDS_Hybrid_full:
            y_data.append(results['methods'][hybrid_meth][cfg["metric"]])
            #print(hybrid_meth, results['methods'][hybrid_meth][cfg["metric"]])
            if cfg["metric"] in ["Inst.Payoff"]:
                y_data_avg.append(results['methods'][hybrid_meth]["Avg_Payoff"])
            elif cfg["metric"] in ["Bid"]:
                y_data_avg.append(results['methods'][hybrid_meth]["Avg_Bid"])
            legends.append(hybrid_meth)
   # print(len(y_data))
    # --- Add optimal baseline if needed ---
    if cfg["metric"] in ["LSW", "SW"]:
        if cfg["metric"] == "SW":
            y_data.append(np.full_like(y_data[0], results['optimal']['SW']))

        else:
            y_data.append(np.full_like(y_data[0], results['optimal']['LSW']))
        #LEGENDS.append("Optimal")

    # 📘 Step 2: Build Plotly figure
    # =====================================================
    fig = go.Figure()
    markers2 = [
        "pentagon", "star", "x", "cross", "square", "triangle-up",
        "triangle-down", "diamond", "circle"
    ]

    h_idx = 1
    if cfg["Track"] :
        if cfg["metric"] == "Bid":
            add_bid_traces(fig, x_data, y_data, legends, cfg)
        else:
            for i, (data, legend) in enumerate(zip(y_data, legends)):

                if cfg["metric"] in ["Avg_Bid", "Inst.Payoff", "Avg_Payoff", "Res_Payoff"] and cfg["num_hybrids"] ==1  and cfg["num_hybrid_set"] ==1:
                    # Pour les graphiques multidimensionnels
                    try:
                        for j in range(np.array(data).shape[1]):
                            try:
                                ydata = data[0]
                                xdata = x_data[::cfg["plot_step"]]
                            except:
                                ydata = data[:,0][::cfg["plot_step"]]
                                xdata = np.array(np.arange(1, np.array(data).shape[0]+1)[:num_hybrids])/ cfg["n"]*100

                            fig.add_trace(go.Scatter(
                                x=xdata,
                                y=ydata,
                                mode="lines+markers",  # ✅ ligne + marqueur
                                name=f"{legend} -- Player {j + 1}",
                                line=dict(color=("red" if legend == "Optimal" else COLORS_METHODS[legends[i]] if legends[i] in METHODS else colors[i]), width=3),  # couleur de ligne
                                marker=dict(
                                    symbol=markers2[j % len(markers2)],  # type de marqueur
                                    size=10,  # ✅ taille fixe (indépendante de plot_step)
                                    line=dict(width=1, color="black")  # contour noir (optionnel pour visibilité)
                                ),
                                opacity=0.8
                            ))
                    except:
                        continue


                else:
                    #print(data)
                    try:
                        fig.add_trace(go.Scatter(
                            x=x_data[::cfg["plot_step"]],
                            y=data[::cfg["plot_step"]],
                            mode='lines+markers',
                            name=legend,
                            line=dict(color=("red" if legend == "Optimal" else COLORS_METHODS[legends[i]] if legends[i] in METHODS else colors[i]), width=3),
                        ))

                    except:
                        fig.add_trace(go.Scatter(
                            x=x_data[::cfg["plot_step"]],
                            y=y_data[::cfg["plot_step"]],
                            mode='lines+markers',
                            name=legend,
                            line=dict(color=("red" if legend == "Optimal" else COLORS_METHODS[legends[i]] if legends[i] in METHODS else colors[i]), width=3),
                        ))
           # print(legend, i, LEGENDS, data)


    # =====================================================
    # 📘 Step 3: Format layout
    # =====================================================
    y_label_map = {
        "Speed": str(rf"$||BR(z(t)) -z(t)||_{{2}}$"),
        "LSW": "LSW",
        "SW": "Social Welfare ",
        "Bid": "Bid",
        "Avg_Bid": "Average Bid",
        "epsilon_error": rf"$\epsilon(z(t))$",
        "epsilon_error_Hybrid": rf"$\epsilon^H(z(t))$",
        "Jain_Index": "Jain Index",
        "Inst.Payoff": "Inst.Payoff",
        "Avg_Payoff": "Average Payoff",
        "Res_Payoff": "Payoff Residual",
        "Dist_To_Optimum_SW": "Distance to Optimal SW",
        "Relative_Efficienty_Loss": r"$\rho(z(t))$",
        "Pareto": "Pareto Check",
        "Potential": "Potential"
    }
    config["y_label"] = y_label_map[cfg["metric"]]
    cfg["y_label"] = y_label_map[cfg["metric"]]
    fig.update_layout(
        title=f"Evolution of {y_label_map[cfg["metric"]]}",
        hovermode="x unified",
        height=600,
        template="plotly_white",
        font=dict(size=18),
        xaxis=dict(
            title=dict(text="Time step (t)", font=dict(size=24)),
            tickfont=dict(size=20),
            exponentformat="power",
            showexponent="all",
        ),
        yaxis=dict(
            title=dict(text=y_label_map[cfg["metric"]], font=dict(size=24)),
            tickfont=dict(size=20),
            exponentformat="power",
            showexponent="all",
        ),
    )

    #z_sol_equ = solve_quadratic(cfg["n"], cfg["a"], cfg["delta"])
    #x_ne = z_sol_equ / (cfg["n"] * z_sol_equ + cfg["delta"])
    x_opt = results['optimal']["x_opt"]
    x_ne = results['optimal']["x_ne"]
    z_ne = results['optimal']["z_ne"]

    payoff_ne = results['optimal']["payoff_ne"]
    Valuation_ne = Valuation(x_ne, cfg["a"], cfg["d_vector"], cfg["alpha"])
    SW_ne = results['optimal']["SW_NE"]
    Jain_idx_ne = results["optimal"]["Jain_index_NE"]
    SW_opt = results['optimal']["SW"]
    Residual_ne = results['optimal']["Residual_ne"]
    RLoss = torch.abs((SW_ne - SW_opt) / SW_opt)*100

    figpath_plot = None
    figpath_legend = None
    figpath_zoom = None

    #y_data = {"speed": y_data_speed, "sw": y_data_sw, "lsw": y_data_lsw}
    if cfg["metric"] in ["Bid", "Avg_Bid", "Inst.Payoff", "Avg_Payoff", "Res_Payoff","Pareto"]  or (cfg["num_hybrids"] >=1  or cfg["num_hybrid_set"] >=1):
        save_to = figure_prefix(cfg)
        if cfg["num_hybrids"] ==1  and cfg["num_hybrid_set"] ==1:
            y_data_2 = y_data
            LEGENDS2 = LEGENDS
            #print(func_group, len(y_data_2))
            if cfg["metric"] in ["Inst.Payoff", "Avg_Payoff", "Res_Payoff"]:
                if cfg["metric"] == "Inst.Payoff":
                    cfg["y_label"] = "Inst.Payoff"
                baseline = payoff_ne  # * np.ones_like(y_data_2[0])
                # y_data_2.append(np.array(baseline))
                func_group.insert(0, cfg["Hybrid_funcs"][0][0])

                func_group.append("NE")

            if cfg["metric"] in ["Bid", "Avg_Bid"]:
                baseline = z_ne.detach().numpy()  # * np.ones_like(y_data_2[0])
                # y_data_2.append(np.array(baseline))
                func_group.insert(0, cfg["Hybrid_funcs"][0][0])
                func_group.append("NE")
            if cfg["metric"] in ["epsilon_error_Hybrid"]:
                baseline =cfg["tol"] *np.ones(2)
                # y_data_2.append(np.array(baseline))
                func_group.insert(0, cfg["Hybrid_funcs"][0][0])
                func_group.append("NE")

            if cfg["metric"] in ["Bid", "Avg_Bid", "Inst.Payoff", "Avg_Payoff", "Res_Payoff","Pareto"]:
                LEGENDS3 = LEGENDS2
                LEGENDS3.append(cfg["Hybrid_funcs"][0][0])
                #print(len(y_data[1]), y_data_2[0])



                if cfg["metric"]=="":
                    outdir = "figures/tmp"
                    os.makedirs(outdir, exist_ok=True)

                    figpath_plot = os.path.join(outdir, "bid_payoff.pdf")
                    figpath_legend = os.path.join(outdir, "legend_algorithms.pdf")

                else:
                    figpath_plot, figpath_legend, figpath_zoom = plotGame_dim_N(cfg, x_data, y_data, y_data_avg,
                                                                                baseline[0],
                                                                                cfg["x_label"],
                                                                                cfg["y_label"], func_group,
                                                                                saveFileName=save_to,
                                                                                fontsize=40, markersize=45,
                                                                                linewidth=16, linestyle="--",
                                                                                Players2See=cfg["Players2See"],
                                                                                ylog_scale=cfg["ylog_scale"],
                                                                                pltText=cfg["pltText"],
                                                                                show_y_axis = cfg["show_y_axis"],
                                                                                step=cfg["plot_step"])

            else:
                figpath_plot, figpath_legend, figpath_zoom = plotGame(cfg, x_data, y_data, cfg["x_label"],
                                                                      cfg["y_label"], LEGENDS,
                                                                      saveFileName=save_to, fontsize=40, markersize=45,
                                                                      linewidth=12, linestyle="--",
                                                                      ylog_scale=cfg["ylog_scale"],
                                                                      pltText=cfg["pltText"], show_y_axis=cfg["show_y_axis"], step=cfg["plot_step"])
        elif cfg["num_hybrids"] ==1  and cfg["num_hybrid_set"] >1:
            y_data_2 = y_data
            LEGENDS2 = LEGENDS
            if cfg["metric"] == "Inst.Payoff":
                cfg["y_label"] = "Inst.Payoff"
            #print(func_group, len(y_data_2))
            if cfg["metric"] in ["Inst.Payoff", "Avg_Payoff", "Res_Payoff"]:
                baseline = payoff_ne  # * np.ones_like(y_data_2[0])
                # y_data_2.append(np.array(baseline))
                func_group.insert(0, cfg["Hybrid_funcs"][0][0])

                func_group.append("NE")

            if cfg["metric"] in ["Bid", "Avg_Bid"]:
                baseline = z_ne.detach().numpy()  # * np.ones_like(y_data_2[0])
                func_group.insert(0, cfg["Hybrid_funcs"][0][0])
                func_group.append("NE")

            if cfg["metric"] in ["Relative_Efficienty_Loss"]:
                #cfg["y_label"] = r"$\rho(z(T))$"
                baseline = RLoss.detach().numpy()
                #.append(np.array(baseline))
                func_group.append("NE")
            elif cfg["metric"] == "epsilon_error":
                #cfg["y_label"] = r"$\epsilon(z(T))$"
                baseline = cfg["tol"]#cfg["tol"]
               # y_data_2.append(np.array(baseline))
                func_group.append("NE")
            if cfg["metric"] in ["Inst.Payoff", "Avg_Payoff", "Res_Payoff", "Bid", "Avg_Bid","epsilon_error_Hybrid"]:
                LEGENDS3 = LEGENDS2
                LEGENDS3.append(cfg["Hybrid_funcs"][0][0])
                if cfg["metric"] == "epsilon_error_Hybrid":
                    baseline = cfg["tol"]
                figpath_plot, figpath_legend, figpath_zoom =plotGame_dim_N(cfg,x_data, y_data_2,y_data_avg,baseline, cfg["x_label"], cfg["y_label"], LEGENDS2, saveFileName=save_to,
                                                                 fontsize=40, markersize=45, linewidth=12,linestyle="--", show_y_axis = cfg["show_y_axis"],
                                                                 Players2See=cfg["Players2See"],
                                             ylog_scale=cfg["ylog_scale"], pltText=cfg["pltText"], step=cfg["plot_step"])
            if cfg["metric"] in ["epsilon_error", "Relative_Efficienty_Loss", "Speed"]:

                figpath_plot, figpath_legend, figpath_zoom = plotGame(cfg, x_data, y_data, cfg["x_label"],
                                                                      cfg["y_label"], LEGENDS,
                                                                      saveFileName=save_to, fontsize=40, markersize=45,
                                                                      linewidth=12, linestyle="--",
                                                                      ylog_scale=cfg["ylog_scale"],
                                                                      pltText=cfg["pltText"],show_y_axis=cfg["show_y_axis"], step=cfg["plot_step"])
        elif cfg["num_hybrids"] > 1 and cfg["num_hybrid_set"] >= 1:
            y_label_map["Inst.Payoff"] = "Inst.Payoff"
            x_data_2 = np.array(cfg["Nb_A1"]) / cfg["n"] * 100
            if num_hybrid_set > 1:
                x_data_2 = np.array(cfg["Nb_A1"][:num_hybrid_set]) / cfg["n"] * 100
            y_data_2 = y_data.copy()

            # y_data_2 = [el.detach().cpu().numpy() if hasattr(el, "detach") else np.array(el)
            #            for el in y_data_2]
            funcs_ = cfg["Hybrid_funcs"][0]

            if num_hybrids >= 1 and num_hybrid_set >= 1:

                x_data_2 = np.array(cfg["Nb_A1"][:num_hybrids]) / cfg["n"] * 100
                save_to2 = figure_prefix(cfg, suffix="player")
                xlab = rf"$\alpha_{{{funcs_[0]}}}$"

                if cfg["metric"] in ["Inst.Payoff", "Avg_Payoff", "Res_Payoff"]:
                    func_group.insert(0, cfg["Hybrid_funcs"][0][0])
                    baseline = payoff_ne[0]  # * np.ones_like(y_data_2[0])

                    # y_data_2.append(np.array(baseline))

                    func_group.append("NE")
                elif cfg["metric"] in ["Bid", "Avg_Bid"]:
                    func_group.insert(0, cfg["Hybrid_funcs"][0][0])
                    baseline = z_ne.detach().numpy()  # * np.ones_like(y_data_2[0])
                    # y_data_2.append(np.array(baseline))
                    baseline = baseline[0]
                elif cfg["metric"] in ["Speed"]:
                    cfg["y_label"] = str(rf"$||BR(z(T)) -z(T)||_2$")
                    #func_group.insert(0, cfg["Hybrid_funcs"][0][0])
                    baseline = Residual_ne  # np.ones_like(y_data_2[0])
                    # y_data_2.append(np.array(baseline))

                    func_group.append("NE")
                elif cfg["metric"] in ["Jain_Index"]:
                    baseline = Jain_idx_ne  # * np.ones_like(y_data_2[0])

                    # y_data_2.append(np.array(baseline))

                    func_group.append("NE")
                elif cfg["metric"] == "Relative_Efficienty_Loss":
                    cfg["y_label"] = r"$\rho(z(T))$"
                    baseline = RLoss.detach().numpy()  # * np.ones_like(y_data_2[0])
                    # y_data_2.append(np.array(baseline))
                    func_group.append("NE")
                elif cfg["metric"] == "epsilon_error":
                    cfg["y_label"] = r"$\epsilon(z(T))$"
                    baseline = Residual_ne  # * np.ones_like(y_data_2[0])
                    # y_data_2.append(np.array(baseline))
                    func_group.append("NE")
                figpath_plot, figpath_zoom, figpath_legend = plotGame_Hybrid_last(cfg, x_data_2, y_data_2,y_data_avg, baseline,
                                                                                  xlab, cfg["y_label"],
                                                                                  cfg["lrMethods"],
                                                                                  saveFileName=save_to2,
                                                                                  funcs_=func_group,
                                                                                  fontsize=40, markersize=45,
                                                                                  linewidth=12,
                                                                                  linestyle="--",
                                                                                  Players2See=cfg["Players2See"],
                                                                                  ylog_scale=cfg["ylog_scale"],
                                                                                  pltText=cfg["pltText"], step=1)

        else:
            #try:

                y_data_2 = y_data
                LEGENDS2 = LEGENDS

                # baseline: une ligne plate de payoff_opt avec la bonne longueur

                if cfg["metric"] in ["Inst.Payoff", "Avg_Payoff", "Res_Payoff"]:
                    baseline = payoff_ne #* np.ones_like(y_data_2[0])
                    y_data_2.append(baseline)


                    LEGENDS2.append("NE")

                elif cfg["metric"] in ["Bid", "Avg_Bid"] :
                    baseline = z_ne.detach().numpy() #* np.ones_like(y_data_2[0])
                   # y_data_2.append(np.array(baseline))
                    LEGENDS2.append("NE")

                figpath_plot, figpath_legend, figpath_zoom =plotGame_dim_N(cfg,x_data, y_data_2,y_data_avg,baseline[0], cfg["x_label"], cfg["y_label"], LEGENDS2, saveFileName=save_to,
                                                                 fontsize=40, markersize=45, linewidth=12,linestyle="--",show_y_axis = cfg["show_y_axis"],
                                                                 Players2See=cfg["Players2See"],
                                             ylog_scale=cfg["ylog_scale"], pltText=cfg["pltText"], step=cfg["plot_step"])
            #except Exception as e:
                save_to = figure_prefix(cfg)
    else:
        save_to = figure_prefix(cfg)
        try:
            #xlab = rf"$\alpha_{{{cfg["Hybrid_funcs"][0][0]}}}$"

            figpath_plot, figpath_legend, figpath_zoom = plotGame(cfg,x_data, y_data, cfg["x_label"], cfg["y_label"], LEGENDS,
                                                    saveFileName=save_to,fontsize=40, markersize=45, linewidth=12,linestyle="--",
                                                        ylog_scale=cfg["ylog_scale"], pltText=cfg["pltText"],show_y_axis=cfg["show_y_axis"], step=cfg["plot_step"])
            #col1, col2 = st.columns([2, 1])
            #with col2:
            #    st.subheader("Infos")
            #    st.write("- MP4 via ffmpeg")
            #    st.write("- Affichage + téléchargement")
            #    st.write("- Génération locale (serveur Streamlit)")
            #with st.spinner("Génération de la vidéo..."):


                #tmpdir = tempfile.mkdtemp()
                #out_base = os.path.join(tmpdir, "convergence_all_methods")
                #video_path = animateGame(cfg,x_data=x_data, y_data=y_data, x_label=cfg["x_label"], y_label=cfg["y_label"], legends=legends,
                #    saveFileName=save_to, ylog_scale=cfg["ylog_scale"], fontsize=40, markersize=45, linewidth=12,
                #    linestyle="-", fps=25, step=cfg["plot_step"], fmt="mp4", dpi=200, show_text=False,
                #)

                # --- read video ---
                #with open(video_path, "rb") as f:
                #    video_bytes = f.read()

            #video_buffer = io.BytesIO(video_bytes)
            #video_buffer.seek(0)

            #with col1:
            #    st.subheader("Vidéo")
            #    st.video(video_buffer)

            #st.download_button(
            #    label="⬇️ Télécharger la vidéo (MP4)",
            #    data=video_buffer,
            #    file_name="convergence.mp4",
            #    mime="video/mp4"
            #)
        except Exception as exc:
            st.error(f"Could not generate the Matplotlib figure: {exc}")

    fig.update_layout(
        title=f"Evolution of {y_label_map[cfg['metric']]}",
        template="plotly_white",
        hovermode="x unified",
        font=dict(size=18),
        xaxis=dict(
            title=dict(text="Time step (t)", font=dict(size=24)),
            tickfont=dict(size=20),
            exponentformat="power",
            showexponent="all",
        ),
        yaxis=dict(
            title=dict(text=y_label_map[cfg['metric']], font=dict(size=24)),
            tickfont=dict(size=20),
            exponentformat="power",
            showexponent="all",
        ),
    )
    if cfg["ylog_scale"]:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)

    if cfg["metric"] == "Bid":
        bid_comparison_rows = build_bid_comparison_table(y_data, legends)
        if bid_comparison_rows:
            comparison_column_config = {
                column: st.column_config.NumberColumn(format="%.6g")
                for column in bid_comparison_rows[0]
                if column != "Player"
            }
            st.subheader("Bid comparison by player")
            st.dataframe(
                bid_comparison_rows,
                use_container_width=True,
                hide_index=True,
                column_config=comparison_column_config,
            )

        bid_summary_rows = build_bid_summary_table(y_data, legends, all_players=True)
        if bid_summary_rows:
            with st.expander("Detailed bid summary", expanded=False):
                st.dataframe(
                    bid_summary_rows,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Initial bid": st.column_config.NumberColumn(format="%.6g"),
                        "Final bid": st.column_config.NumberColumn(format="%.6g"),
                        "Change": st.column_config.NumberColumn(format="%.6g"),
                        "Change (%)": st.column_config.NumberColumn(format="%.2f%%"),
                    },
                )

    # Affichage des valeurs finales
    st.subheader("Final values")
    cols = st.columns(len(cfg["lrMethods"]) + 1)
    for i, method in enumerate(LEGENDS):#(cfg["lrMethods"]):
        if method in results['methods']:
            with cols[i]:
                st.metric(
                    label=method,
                    value=f"{results['methods'][method]['convergence_iter']} iterations",
                    help=f"Last error: {results['methods'][method]['Speed'][-1]:.6f}"
                )

    with cols[-1]:
        st.metric(
            label="Optimal",
            value=f"LSW: {results['optimal']['LSW']:.2f}",
            help=f"SW: {results['optimal']['SW']:.2f}"
        )


    pdf_outputs = [
        ("⬇️ Plot PDF", figpath_plot),
        ("⬇️ Legend PDF", figpath_legend),
        ("⬇️ Zoom PDF", figpath_zoom),
    ]
    available_pdfs = [(label, path) for label, path in pdf_outputs if path and os.path.exists(path)]
    if available_pdfs:
        st.subheader("📂 Download Outputs")
        btn_cols = st.columns(len(available_pdfs))
        for btn_col, (label, path) in zip(btn_cols, available_pdfs):
            with open(path, "rb") as pdf_file:
                btn_col.download_button(label, pdf_file, file_name=os.path.basename(path))
    else:
        st.info("PDF files not available yet.")
else:
     st.info("ℹ️ No results yet. Click ▶️ Run Simulation to start.")

#except Exception:
       #st.info("ℹ️ No results available yet. Please press **▶️ Run Simulation** to start.")

# The former Liquid Price of Anarchy UI was removed to keep the simulator focused.

# -----------------------
# CONVERGENCE ITERATIONS VS BUDGET HETEROGENEITY
# -----------------------

st.header("2. Convergence under budget heterogeneity")
st.caption("Compare how many iterations each selected method needs to satisfy the Nash residual tolerance.")
st.write("This section runs the selected algorithms for increasing budget heterogeneity μ and records the first iteration where the Nash residual is <= tol.")

with st.expander("Convergence controls", expanded=False):
    default_mu_convergence_grid = ", ".join(f"{x:.3g}" for x in [0, 8, 16, 24, 32, 40,48,56,64,72,80])#np.linspace(0, float(cfg["c"]) / cfg["n"], 6))
    convergence_mu_text = st.text_input("μ values for convergence", value=default_mu_convergence_grid)
    st.caption(
        f"Uses selected algorithms: {', '.join(cfg['lrMethods'])}. "
        f"Each point uses T={cfg['T']}, tol={cfg['tol']}, and Nb_random_sim={cfg['Nb_random_sim']}."
    )

    if st.button("⏱️ Run convergence vs μ"):
        try:
            convergence_mu_grid = parse_positive_float_list(convergence_mu_text, min_value=0.0, inclusive_min=True)
            convergence_progress = st.progress(0)
            with st.spinner("Computing convergence iterations over μ..."):
                st.session_state.Convergence_Mu_Results = compute_convergence_iterations_vs_mu(
                    cfg,
                    convergence_mu_grid,
                    progress_bar=convergence_progress,
                )
            st.success("Convergence experiment finished.")
        except Exception as exc:
            st.error(f"Could not compute convergence results: {exc}")

if "Convergence_Mu_Results" in st.session_state:
    convergence_results = st.session_state.Convergence_Mu_Results
    algorithms, mu_values, convergence_matrix = build_convergence_matrix(convergence_results)
    convergence_array = np.asarray(convergence_matrix, dtype=float)
    convergence_tabs = st.tabs(["Log comparison", "Heatmap", "Ranked bars"])

    with convergence_tabs[0]:
        fig_convergence = go.Figure()
        for algorithm_index, algorithm in enumerate(algorithms):
            algorithm_color = method_plot_color(legend_map.get(algorithm, algorithm), algorithm_index)
            fig_convergence.add_trace(go.Scatter(
                x=mu_values,
                y=convergence_results["series"][algorithm],
                mode="lines+markers",
                name=algorithm,
                line=dict(color=algorithm_color, width=3),
                marker=dict(color=algorithm_color, size=10, line=dict(color="black", width=1)),
                hovertemplate=(
                    "μ=%{x}<br>"
                    "Iterations=%{y}<extra>%{fullData.name}</extra>"
                ),
            ))

        fig_convergence.update_layout(
            title="Iterations to converge vs budget heterogeneity μ",
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Budget heterogeneity μ",
            yaxis_title="Iterations (log scale)",
            font=dict(size=16),
        )
        fig_convergence.update_yaxes(type="log")
        st.plotly_chart(fig_convergence, use_container_width=True)

    with convergence_tabs[1]:
        heatmap_text = [
            ["NC" if np.isnan(value) else f"{value:.0f}" for value in row]
            for row in convergence_array
        ]
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=convergence_array,
            x=[f"{mu:g}" for mu in mu_values],
            y=algorithms,
            colorscale="Viridis_r",
            colorbar=dict(title="Iterations"),
            text=heatmap_text,
            texttemplate="%{text}",
            hovertemplate=(
                "Algorithm=%{y}<br>"
                "μ=%{x}<br>"
                "Iterations=%{z}<extra></extra>"
            ),
        ))
        fig_heatmap.update_layout(
            title="Convergence iterations heatmap",
            template="plotly_white",
            xaxis_title="Budget heterogeneity μ",
            yaxis_title="Algorithm",
            height=max(420, 70 * len(algorithms)),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with convergence_tabs[2]:
        selected_mu = st.selectbox(
            "μ to rank",
            options=mu_values,
            index=len(mu_values) - 1,
            format_func=lambda value: f"{value:g}",
        )
        selected_index = mu_values.index(selected_mu)
        ranked_rows = []
        for algorithm_index, algorithm in enumerate(algorithms):
            iterations = convergence_array[algorithm_index, selected_index]
            ranked_rows.append({
                "Algorithm": algorithm,
                "Iterations": iterations,
            })
        ranked_rows = sorted(
            ranked_rows,
            key=lambda row: np.inf if np.isnan(row["Iterations"]) else row["Iterations"],
        )
        fig_ranked = go.Figure(go.Bar(
            x=[row["Iterations"] for row in ranked_rows],
            y=[row["Algorithm"] for row in ranked_rows],
            orientation="h",
            marker=dict(
                color=[
                    method_plot_color(legend_map.get(row["Algorithm"], row["Algorithm"]), idx)
                    for idx, row in enumerate(ranked_rows)
                ],
                line=dict(color="black", width=1),
            ),
            text=["NC" if np.isnan(row["Iterations"]) else f"{row['Iterations']:.0f}" for row in ranked_rows],
            textposition="outside",
            hovertemplate="Algorithm=%{y}<br>Iterations=%{x}<extra></extra>",
        ))
        fig_ranked.update_layout(
            title=f"Algorithm ranking at μ={selected_mu:g}",
            template="plotly_white",
            xaxis_title="Iterations to converge",
            yaxis_title="Algorithm",
            height=max(420, 60 * len(ranked_rows)),
        )
        st.plotly_chart(fig_ranked, use_container_width=True)

    try:
        convergence_log_pdf, convergence_log_legend_pdf = save_convergence_mu_log_pdf(cfg, convergence_results)
        convergence_heatmap_pdf = save_convergence_mu_pdf(cfg, convergence_results)
        pdf_cols = st.columns(3)
        with open(convergence_log_pdf, "rb") as pdf_file:
            pdf_cols[0].download_button(
                "⬇️ Log comparison PDF",
                pdf_file,
                file_name=os.path.basename(convergence_log_pdf),
                mime="application/pdf",
            )
        with open(convergence_log_legend_pdf, "rb") as pdf_file:
            pdf_cols[1].download_button(
                "⬇️ Log legend PDF",
                pdf_file,
                file_name=os.path.basename(convergence_log_legend_pdf),
                mime="application/pdf",
            )
        with open(convergence_heatmap_pdf, "rb") as pdf_file:
            pdf_cols[2].download_button(
                "⬇️ Convergence heatmap PDF",
                pdf_file,
                file_name=os.path.basename(convergence_heatmap_pdf),
                mime="application/pdf",
            )
    except Exception as exc:
        st.info(f"Convergence PDF not available yet: {exc}")

    st.dataframe(
        convergence_results["rows"],
        use_container_width=True,
        hide_index=True,
        column_config={
            "mu": st.column_config.NumberColumn(format="%.6g"),
            "Iterations to converge": st.column_config.NumberColumn(format="%.0f"),
            "Final residual": st.column_config.NumberColumn(format="%.3e"),
        },
    )


# -----------------------
# SIMULATION TABLE
# -----------------------
st.header("3. Parameter-sweep tables")
st.caption("Summarize repeated simulations across the configured player and heterogeneity grids.")

if st.button("📊 Run Simulation Table"):
    with st.spinner("Simulating..."):
        results_table, results_table_eps = run_simulation_table_avg(cfg, GameKelly)
        st.success("Done.")

        st.session_state.results_table = results_table
        st.session_state.results_table_eps = results_table_eps
        st.session_state.config = cfg
        #display_results_streamlit_dict(results_table, cfg, save_path="results/table_results.csv")

if "results_table" in st.session_state:
    try:
        results_table = st.session_state.results_table
        display_results_streamlit_dict(
            results_table,
            cfg,
            save_path="results/table_results.csv",
            convergence_measure="bid",
        )
    except Exception:
        st.info("ℹ️ No results available yet. Please press **📊 Run Simulation Table** to start.")
if "results_table_eps" in st.session_state:
    try:
        results_table_eps = st.session_state.results_table_eps
        display_results_streamlit_dict(
            results_table_eps,
            cfg,
            save_path="results/table_results.csv",
            convergence_measure="payoff",
        )
    except Exception:
        st.info("ℹ️ No results available yet. Please press **📊 Run Simulation Table** to start.")

st.header("4. Fairness versus heterogeneity")
st.caption("Measure how Jain's allocation index changes with γ and the number of players.")

if st.button("📈 Run Jain versus γ"):

    with st.spinner("Running Jain index simulations over γ and n..."):
        jain_results, gamma_grid = run_jain_vs_gamma(cfg, GameKelly)
    st.success("Done.")

    st.session_state.jain_results = jain_results
    st.session_state.config = cfg
    st.session_state.gamma_grid = gamma_grid

if 'jain_results' in st.session_state:
    # --- Retrieve data from session ---
    jain_results = st.session_state.jain_results
    config = st.session_state.config
    gamma_grid = st.session_state.gamma_grid

    fig = plot_jain_vs_gamma(jain_results, gamma_grid, config)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("📂 Download Outputs")
    figpath_plot = plotGame_Jain(cfg,jain_results, gamma_grid,
    ylog_scale=False, fontsize=40, markersize=40, linewidth=12,
    linestyle="-", pltText=False, step=1,tol=1e-6
)
    btn_cols = st.columns(1)
    try:
        with open(figpath_plot, "rb") as f1:
            btn_cols[0].download_button("⬇️ Plot PDF", f1, file_name=figpath_plot)
    except:
        st.info("PDF files not available yet.")
    else:
         st.info("ℹ️ No results yet. Click 📈 Run Jain(gamma) to start.")


import os

import numpy as np
import matplotlib.pyplot as plt

def save_bar_gamma_curvature(
    df,
    filename=str(JOURNAL_FIGURE_DIR / "bar_gamma_curvature"),
    eps_floor=1e-12,
    log_scale=True,
    dpi=300,
):
    """
    Grouped bar plot (Linear vs Log) with:
      - y = 100*rho_mean (percentage)
      - optional log-scale y-axis
      - numeric % labels on bars
      - saves both PDF and JPEG
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import os

    gammas = sorted(df["gamma"].unique())
    utilities = ["Log", "Linear"]

    width = 0.35
    x = np.arange(len(gammas))

    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "Linear": "red",
        "Log": "blue",
    }

    for i, util in enumerate(utilities):
        vals = (
            df[df["Utility"] == util]
            .sort_values("gamma")["rho_mean"]
            .to_numpy()
        ) * 100.0

        vals_plot = np.maximum(vals, 100.0 * eps_floor)

        bars = ax.bar(
            x + (i - 0.5) * width,
            vals_plot,
            width,
            color=colors[util],
            label=util,
        )

        # ---- ADD PERCENTAGE LABELS ----
        for bar, v in zip(bars, vals):
            label = f"{v:.2e}%" if v < 0.001 else f"{v:.3g}%"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.05,
                label,
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([fr"$\gamma={g}$" for g in gammas])

    if log_scale:
        ax.set_yscale("log")

    ax.set_ylabel(r"$100\times \rho_{\mathrm{mean}}$ (\%)")
    ax.legend()
    ax.grid(True, axis="y", which="both", alpha=0.3)

    # ---------- SAVE ----------
    output_base = Path(filename)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = str(output_base.with_suffix(".pdf"))
    jpeg_path = str(output_base.with_suffix(".jpg"))

    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(
        jpeg_path,
        format="jpg",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    return pdf_path, jpeg_path


def save_bar_gamma_curvature_pdf(
    df,
    filename=str(JOURNAL_FIGURE_DIR / "bar_gamma_curvature.pdf"),
    eps_floor=1e-12,log_scale=True,
):
    """
    Grouped bar plot (Linear vs Log) with:
      - y = 100*rho_mean (percentage)
      - log-scale y-axis
      - numeric % labels on bars
    """

    gammas = sorted(df["gamma"].unique())
    utilities = [ "Log", "Linear"]

    width = 0.35
    x = np.arange(len(gammas))
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "Linear": "red",
        "Log": "blue",
    }

    for i, util in enumerate(utilities):
        vals = (
            df[df["Utility"] == util]
            .sort_values("gamma")["rho_mean"]
            .to_numpy()
        ) * 100.0

        # avoid log(0)
        vals_plot = np.maximum(vals, 100.0 * eps_floor)


        bars = ax.bar(
            x + (i - 0.5) * width,
            vals_plot,
            width,
            color = colors[util],
            label=util
        )

        # ---- ADD PERCENTAGE LABELS ----
        for bar, v in zip(bars, vals):
            if v < 0.001:
                label = f"{v:.2e}%"
            else:
                label = f"{v:.3g}%"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.05,   # offset in log-scale
                label,
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                rotation=0
            )

    # ---- Axes & style ----
    if log_scale:
        ax.set_yscale("log")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.set_xticks(x)
    ax.set_xticklabels(gammas)
    ax.set_ylim(bottom=1e-3, top=25.0)

    ax.set_xlabel(str(r"Heterogeneity ($\gamma$)"), fontsize=25)

    ax.set_ylabel(str(r"$\rho(z^{NE})$"), fontsize=25)
    #ax.set_title("Relative efficiency loss vs heterogeneity γ")

    ax.legend(frameon=False, prop={'weight': 'bold'})
    ax.grid(True, which="both", axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return str(output_path)


st.header("5. Linear versus logarithmic utility")
st.caption("Compare efficiency loss for α = 0 and α = 1 across heterogeneity levels.")

if st.button("📈 Run linear versus logarithmic utility"):

    with st.spinner("Running Linear VS Log)..."):
        compare_results = run_main_gamma_curvature(cfg, GameKelly, lrMethod_fixed="DAQ_F", n_fixed=cfg["n"])
    st.success("Done.")

    st.session_state.compare_results = compare_results
    st.session_state.config = cfg
   # st.session_state.gamma_grid = gamma_grid

if 'compare_results' in st.session_state:
    # --- Retrieve data from session ---
    compare_results = st.session_state.compare_results
    config = st.session_state.config
    df, fig_compare = plot_main_bar_gamma_curvature(compare_results, lrMethod_fixed="DAQ_F")
    #print(df)

    st.subheader("📂 Download Outputs")


    pdf_path = save_bar_gamma_curvature_pdf(df,log_scale=cfg["ylog_scale"])

    with open(pdf_path, "rb") as f:
        st.download_button(
            "⬇️ Download plot (PDF)",
            f,
            file_name=pdf_path,
            mime="application/pdf"
        )


def save_multiresource_figures_pdf(result, selected_alpha):
    """Save every multi-resource plot using the gamma-figure publication format."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogFormatter

    records = result["records"]
    alpha_values = np.asarray([row["alpha"] for row in records], dtype=float)
    algorithm = result["algorithm"]
    compared_algorithms = result.get("algorithms", [algorithm])
    output_prefix = JOURNAL_FIGURE_DIR / (
        "multiresource_" + "_vs_".join(compared_algorithms)
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {}

    def style_axis(ax, xlabel, ylabel):
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_ylabel(ylabel, fontsize=25)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.legend(frameon=False, prop={"weight": "bold"})
        ax.grid(True, which="both", alpha=0.3)
        fig = ax.figure
        fig.tight_layout()

    # Keep export labels parser-free. Unicode labels are robust even when another
    # part of the application has changed Matplotlib's global math-text settings.
    with plt.rc_context({"font.size": 16, "text.usetex": False}):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            alpha_values, [row["efficiency"] for row in records],
            color="blue", marker="o", label="Weighted efficiency",
        )
        ax.plot(
            alpha_values, [row["jain"] for row in records],
            color="red", marker="s", label="Jain fairness",
        )
        ax.set_ylim(0.0, 1.05)
        style_axis(ax, "Fairness parameter (α)", "Index")
        paths["tradeoff"] = str(output_prefix.with_name(
            f"{output_prefix.name}_tradeoff.pdf"
        ))
        fig.savefig(paths["tradeoff"], format="pdf", bbox_inches="tight")
        plt.close(fig)

        # Match the main Run Simulation figure: 18×12, thick curves, large
        # labels, and a separate legend-only PDF. Each curve is one alpha.
        fig, ax = plt.subplots(figsize=(18, 12))
        legend_handles, legend_labels = [], []
        algorithm_outputs = result.get("algorithm_results", {
            algorithm: {
                "records": records,
                "convergence_histories": result["convergence_histories"],
            }
        })
        residual_mode = result["residual_mode"]
        available_iterations = max(
            len(values) - (1 if residual_mode == "best_response" else 0)
            for output in algorithm_outputs.values()
            for values in output["convergence_histories"].values()
        )
        plot_iterations = min(
            int(result.get("plot_iterations", available_iterations)),
            available_iterations,
        )
        # Algorithms retain exactly the color and marker assigned to them in
        # Run Simulation. Alpha is encoded only through the line style.
        line_styles = ["-", "--", ":", "-."]
        all_alpha_values = sorted({
            float(row["alpha"])
            for output in algorithm_outputs.values() for row in output["records"]
        })
        alpha_line_style = {
            value: line_styles[index % len(line_styles)]
            for index, value in enumerate(all_alpha_values)
        }
        for algorithm_index, (method, output) in enumerate(algorithm_outputs.items()):
            styled_method = legend_map.get(method, method)
            color = COLORS_METHODS.get(
                styled_method, colors[algorithm_index % len(colors)]
            )
            marker = MARKERS_METHODS.get(
                styled_method, markers[algorithm_index % len(markers)]
            )
            for row in output["records"]:
                alpha_value = float(row["alpha"])
                residual_history = np.maximum(np.asarray(
                    output["convergence_histories"][alpha_value], dtype=float
                ), 1e-16)[:plot_iterations + 1]
                iterations = np.arange(len(residual_history)) if residual_mode == "best_response" else np.arange(1, len(residual_history) + 1)
                linestyle = alpha_line_style[alpha_value]
                mark_every = max(1, len(iterations) // 20)
                ax.plot(iterations, residual_history, color=color, linestyle=linestyle,
                        linewidth=12, marker=marker, markersize=28,
                        markeredgecolor="black", markevery=mark_every)
                legend_handles.append(Line2D(
                    [0], [0], color=color, linestyle=linestyle, linewidth=12,
                    marker=marker, markersize=28, markeredgecolor="black",
                ))
                if result.get("alpha_mode") == "One α per device":
                    alpha_description = ", ".join(f"{value:g}" for value in row["alpha_values"])
                    alpha_label = f"αᵢ = ({alpha_description})"
                else:
                    alpha_label = f"α = {alpha_value:g}"
                legend_labels.append(f"{method} — {alpha_label}")

        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(LogFormatter(base=10))
        ax.set_xlabel("Iteration (t)", fontsize=55)
        residual_ylabel = (
            rf"$‖z^t − z^{{t+1}}‖_2$"
            if residual_mode == "iterate_difference"
            else rf"$‖BR(z(t)) − z(t)‖_2$"
        )
        ax.set_ylabel(residual_ylabel, fontsize=2*40)
        ax.tick_params(axis="both", which="major", labelsize=48)
        ax.tick_params(axis="both", which="minor", labelsize=40)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        paths["convergence"] = str(
            output_prefix.with_name(f"{output_prefix.name}_convergence_time.pdf")
        )
        fig.savefig(paths["convergence"], format="pdf", bbox_inches="tight")
        plt.close(fig)

        legend_fig = plt.figure(figsize=(12, 2))
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis("off")
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="center",
            ncol=max(1, min(len(legend_labels), 4)),
            frameon=True,
            facecolor="white",
            edgecolor="black",
            prop={"weight": "bold", "size": 24},
        )
        paths["convergence_legend"] = str(output_prefix.with_name(
            f"{output_prefix.name}_convergence_time_legend.pdf"
        ))
        legend_fig.savefig(
            paths["convergence_legend"], format="pdf", bbox_inches="tight"
        )
        plt.close(legend_fig)

        selected_x = result["allocations"][float(selected_alpha)]
        players = np.arange(selected_x.shape[0])
        width = 0.8 / selected_x.shape[1]
        fig, ax = plt.subplots(figsize=(8, 5))
        for resource in range(selected_x.shape[1]):
            offset = (resource - (selected_x.shape[1] - 1) / 2.0) * width
            ax.bar(
                players + offset, selected_x[:, resource], width,
                label=f"Resource {resource + 1}",
            )
        ax.set_xticks(players)
        ax.set_xticklabels([str(player + 1) for player in players])
        style_axis(ax, "Player", "Allocation (xᵢʳ)")
        alpha_label = f"{float(selected_alpha):g}".replace(".", "p")
        paths["allocation"] = str(output_prefix.with_name(
            f"{output_prefix.name}_allocation_alpha{alpha_label}.pdf"
        ))
        fig.savefig(paths["allocation"], format="pdf", bbox_inches="tight")
        plt.close(fig)

        # Resource-centric view: each bar is one resource and its stacked
        # segments show the share received by every player.
        resource_totals = selected_x.sum(axis=0)
        shares = np.divide(
            selected_x,
            resource_totals[None, :],
            out=np.zeros_like(selected_x),
            where=resource_totals[None, :] > 0,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        resource_positions = np.arange(selected_x.shape[1])
        bottom = np.zeros(selected_x.shape[1])
        for player in range(selected_x.shape[0]):
            ax.bar(
                resource_positions, shares[player], bottom=bottom,
                label=f"Player {player + 1}",
            )
            bottom += shares[player]
        ax.set_xticks(resource_positions)
        ax.set_xticklabels([
            f"Resource {resource + 1}" for resource in resource_positions
        ])
        ax.set_ylim(0.0, 1.0)
        style_axis(ax, "Resource", "Player share of allocated resource")
        paths["resource_share"] = str(output_prefix.with_name(
            f"{output_prefix.name}_resource_share_alpha{alpha_label}.pdf"
        ))
        fig.savefig(paths["resource_share"], format="pdf", bbox_inches="tight")
        plt.close(fig)

    return paths


# =========================================================
# MULTI-RESOURCE EFFICIENCY–FAIRNESS EXPERIMENT
# =========================================================
st.header("6. Multi-resource allocation")
st.caption(
    "Compare learning rules when players share several capacity-constrained resources."
)
st.markdown(r"""
Each player $i$ bids $b_i^r$ for resource $r$, giving the final allocation
$$
x_i^r=C_r\frac{b_i^r}{\sum_j b_j^r+\delta}.
$$
The experiment varies the α-fair utility parameter and can compare exact
**BR** (for $\alpha\in\{0,1,2\}$), **OGD_F**, with
$\eta_i=\kappa D_i/(G_i\sqrt{T})$, and **OGD_V**, with
$\eta_{i,t}=\kappa D_i/(G_i\sqrt{t})$. It also supports **DAQ_F**, which
projects the fixed-step cumulative gradient, and **RRM_V**, which projects
the cumulative sequence of varying-step gradients. Here, $\kappa$ is the selectable
step-size scale, $D_i$ is the Euclidean diameter of
player $i$'s budget-simplex action set and $G_i$ bounds the norm of that
player's payoff gradient. The practical playerwise option is
$$
G_i=\max_k\left(\frac{a_i^k}{\epsilon}+1\right).
$$
Alternatively, the domain $\ell_\infty$ option computes
$$
G_i=\max_k\sup_{\mathbf z\in\mathcal Z}
\left|\frac{\partial u_i}{\partial z_i^k}\right|.
$$
Each player chooses a nonempty subset $S_i$ of required resources. Bids on
resources outside $S_i$ are fixed to zero, and the minimum-bid and budget
constraints apply only on $S_i$. Jain's index and the convergence residual
are tracked separately for every resource (over the players requiring it).
Their resource-wise mean is used for the aggregate fairness curve. Efficiency
is valuation-weighted throughput relative to the
unconstrained maximum.

The utility used for every player–resource allocation is
$$
U_\alpha(x)=
\begin{cases}
x, & \alpha=0,\\
\log(x), & \alpha=1,\\
\dfrac{x^{1-\alpha}}{1-\alpha}, & \text{otherwise}.
\end{cases}
$$
Device $i$ uses the same $\alpha_i$ for all resources. Its total utility and
payoff are
$$
U_i(\mathbf{x}_i)=\sum_{r\in S_i} a_i^r V^{\alpha_i}(x_i^r),
\qquad
u_i=U_i(\mathbf{x}_i)-\sum_r b_i^r,
\qquad \sum_r b_i^r\le B_i,
$$
where $a_i^r=a_i\theta_r$. Here $a_i$ is device $i$'s interest factor and
$\theta_r$ is the user-selected importance of resource $r$. The bid price is
currently one.

Choose one of two residual definitions:
$$
R_{\mathrm{step}}^t=\lVert\mathbf z^t-\mathbf z^{t+1}\rVert_2,
\qquad
R_{\mathrm{BR}}^t=\lVert\mathbf z^{\mathrm{BR}}(\mathbf z^t)-\mathbf z^t\rVert_2.
$$
For the best-response residual, all players' best responses are collected in
$\mathbf z^{\mathrm{BR}}$. With $\mathbf z_{-i}^t$ fixed, player $i$ solves
$$
\mathbf z_i^{\mathrm{BR}}\in\arg\max_{\mathbf z_i}
\left\{\sum_{r\in S_i}\left[a_i^rV^{\alpha_i}(x_i^r)-z_i^r\right]\right\}
\quad\text{s.t.}\quad
\sum_r z_i^r\le B_i,\;\;z_i^r\ge\epsilon.
$$
""")

mr_col1, mr_col2, mr_col3 = st.columns(3)
with mr_col1:
    mr_resources = st.number_input("Resources", 1, 10, 2, key="mr_resources")
    mr_algorithms = st.multiselect(
        "Learning algorithms to compare",
        ["BR", "OGD_F", "OGD_V", "RRM_V", "DAQ_F"],
        default=["OGD_F", "OGD_V"], key="mr_algorithms",
        help="BR uses the exact closed-form player response for α = 0, 1, or 2.",
    )
    mr_iterations = st.number_input(
        "Iterations", 10, 100000, min(int(cfg["T"]), 5000), step=10, key="mr_iterations"
    )
    mr_step_scale = st.number_input(
        "OGD step-size scale κ",
        min_value=0.01,
        max_value=100.0,
        value=10.0,
        step=0.5,
        key="mr_step_scale",
        help="κ=1 is the conservative theoretical step; κ=10 is faster in practice.",
    )
    mr_gradient_bound_label = st.selectbox(
        "Gradient bound Gᵢ",
        [
            "Practical bound",
            "Domain ℓ∞ bound",
            "Legacy Run Simulation bound (one resource)",
        ],
        key="mr_gradient_bound_mode",
        help=(
            "The domain ℓ∞ option bounds the absolute full payoff gradient "
            "over all feasible own bids and opponent competition."
        ),
    )
with mr_col2:
    mr_alpha_mode = st.radio(
        "α configuration",
        ["Same α for every device", "One α per device"],
        key="mr_alpha_mode",
        help="Each device always uses its selected α for all resources.",
    )
    mr_alpha_text = st.text_input(
        "α values to plot",
        "0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4",
        key="mr_alpha_values",
        disabled=mr_alpha_mode == "One α per device",
        help=(
            "Enter the exact nonnegative α values to simulate, separated by "
            "commas. One residual curve is drawn for each value."
        ),
    )
    default_device_alphas = ", ".join(
        [str(float(cfg["alpha"]))] * int(cfg["n"])
    )
    mr_device_alpha_text = st.text_input(
        "α values by device (α₁, …, αₙ)",
        default_device_alphas,
        key="mr_device_alphas",
        disabled=mr_alpha_mode == "Same α for every device",
    )
    mr_convergence_tolerance = st.number_input(
        "Convergence tolerance", min_value=1e-10, value=1e-4,
        format="%.1e", key="mr_convergence_tolerance"
    )
    mr_residual_label = st.radio(
        "Residual definition",
        [
            "Iterate difference ‖zᵗ − zᵗ⁺¹‖₂",
            "Best response ‖zᴮᴿ(zᵗ) − zᵗ‖₂",
        ],
        key="mr_residual_mode",
        help=(
            "The best-response option jointly optimizes every player's "
            "resource bids under that player's total budget."
        ),
    )
    mr_residual_interval = st.number_input(
        "Exact BR residual sampling interval",
        min_value=1,
        max_value=1000,
        value=10,
        step=1,
        key="mr_residual_interval",
        disabled=mr_residual_label.startswith("Iterate difference"),
        help=(
            "Evaluate the expensive exact BR diagnostic every k iterations; "
            "the last value is carried between samples. Use 1 for every iteration."
        ),
    )
with mr_col3:
    default_capacities = ", ".join(["1"] * int(mr_resources))
    mr_capacity_text = st.text_input(
        "Resource capacities Cᵣ", default_capacities, key="mr_capacities"
    )
    default_resource_importance = ", ".join(
        str(resource) for resource in range(1, int(mr_resources) + 1)
    )
    mr_theta_text = st.text_input(
        "Resource importance θᵣ",
        default_resource_importance,
        key="mr_resource_importance",
        help="Enter one positive θ value per resource. Default: 1, 2, …, m.",
    )
    default_requirements = "; ".join(
        [",".join(str(r + 1) for r in range(int(mr_resources)))] * int(cfg["n"])
    )
    mr_requirements_text = st.text_area(
        "Required resources by player",
        default_requirements,
        key="mr_required_resources",
        help=(
            "One semicolon-separated subset per player; resources are numbered "
            "from 1. Example: 1,2; 2; 1,3."
        ),
    )
    mr_repetitions = st.number_input(
        "Random repetitions", 1, 50, 3,
        key="mr_repetitions",
    )
    mr_minimum_bid = st.number_input(
        "Minimum bid", 1e-6, 1.0, float(np.clip(cfg["epsilon"], 1e-4, 1.0)),
        format="%.6f", key="mr_minimum_bid"
    )
    mr_delta = st.number_input(
        "Allocation slack δ", min_value=0.0, value=float(cfg["delta"]),
        step=0.1, key="mr_delta"
    )

if st.button("⚖️ Run multi-resource experiment"):
    try:
        if not mr_algorithms:
            raise ValueError("Select at least one learning algorithm.")
        mr_capacities = np.asarray(
            [float(value.strip()) for value in mr_capacity_text.split(",") if value.strip()],
            dtype=float,
        )
        if len(mr_capacities) != int(mr_resources) or np.any(mr_capacities <= 0):
            raise ValueError(f"Enter exactly {int(mr_resources)} positive capacities.")
        mr_theta = np.asarray(
            [float(value.strip()) for value in mr_theta_text.split(",") if value.strip()],
            dtype=float,
        )
        if len(mr_theta) != int(mr_resources) or np.any(mr_theta <= 0):
            raise ValueError(
                f"Enter exactly {int(mr_resources)} positive resource-importance values."
            )

        mr_n = int(cfg["n"])
        requirement_rows = [row.strip() for row in mr_requirements_text.split(";")]
        if len(requirement_rows) != mr_n or any(not row for row in requirement_rows):
            raise ValueError(f"Enter exactly {mr_n} nonempty resource subsets.")
        mr_requirements = np.zeros((mr_n, int(mr_resources)), dtype=bool)
        for player, row in enumerate(requirement_rows):
            selected = {int(value.strip()) for value in row.split(",") if value.strip()}
            if not selected or min(selected) < 1 or max(selected) > int(mr_resources):
                raise ValueError(
                    f"Player {player + 1}'s subset must use resource numbers 1–{int(mr_resources)}."
                )
            mr_requirements[player, np.asarray(sorted(selected)) - 1] = True
        effective_step_scale = float(mr_step_scale)
        effective_tolerance = float(mr_convergence_tolerance)
        effective_repetitions = int(mr_repetitions)
        gradient_bound_mode = {
            "Practical bound": "practical",
            "Domain ℓ∞ bound": "infinity",
            "Legacy Run Simulation bound (one resource)": "legacy",
        }[mr_gradient_bound_label]
        residual_mode = {
            "Iterate difference ‖zᵗ − zᵗ⁺¹‖₂": "iterate_difference",
            "Best response ‖zᴮᴿ(zᵗ) − zᵗ‖₂": "best_response",
        }[mr_residual_label]
        if gradient_bound_mode == "legacy":
            if int(mr_resources) != 1:
                raise ValueError(
                    "Select exactly one resource to use the legacy Run Simulation bound."
                )
            effective_step_scale = 1.0
        player_scale = np.asarray(cfg["a_vector"][:mr_n], dtype=float)
        if len(player_scale) != mr_n:
            player_scale = np.asarray(
                [max(cfg["a"] - cfg["gamma"] * i, cfg["a_min"]) for i in range(mr_n)]
            )
        # a_i^k = a_i * theta_k
        mr_valuations = player_scale[:, None] * mr_theta[None, :]
        mr_budgets = np.asarray(
            [max(cfg["c"] - i * cfg["mu"], mr_minimum_bid * mr_requirements[i].sum())
             for i in range(mr_n)]
        )
        mr_player_alphas = None
        if mr_alpha_mode == "One α per device":
            mr_player_alphas = np.asarray(
                [
                    float(value.strip())
                    for value in mr_device_alpha_text.split(",")
                    if value.strip()
                ],
                dtype=float,
            )
            if len(mr_player_alphas) != mr_n or np.any(mr_player_alphas < 0):
                raise ValueError(
                    f"Enter exactly {mr_n} nonnegative α values, one per device."
                )
            mr_alphas = np.asarray([float(np.mean(mr_player_alphas))])
        else:
            mr_alphas = np.asarray(
                [
                    float(value.strip())
                    for value in mr_alpha_text.replace(";", ",").split(",")
                    if value.strip()
                ],
                dtype=float,
            )
            if (
                mr_alphas.size == 0
                or np.any(~np.isfinite(mr_alphas))
                or np.any(mr_alphas < 0.0)
            ):
                raise ValueError(
                    "Enter at least one finite, nonnegative α value."
                )
            if len(np.unique(mr_alphas)) != len(mr_alphas):
                raise ValueError("Enter each α value only once.")
        if "BR" in mr_algorithms:
            selected_alpha_values = (
                mr_player_alphas if mr_player_alphas is not None else mr_alphas
            )
            if mr_player_alphas is not None and any(
                not any(np.isclose(value, supported) for supported in (0.0, 1.0, 2.0))
                for value in selected_alpha_values
            ):
                raise ValueError(
                    "With one α per device, exact BR requires every device α to be 0, 1, or 2."
                )

        with st.spinner("Running multi-resource α experiment..."):
            mr_algorithm_results = {}
            for algorithm in mr_algorithms:
                algorithm_alphas = mr_alphas
                if algorithm == "BR" and mr_player_alphas is None:
                    algorithm_alphas = np.asarray([
                        value for value in mr_alphas
                        if any(np.isclose(value, supported) for supported in (0.0, 1.0, 2.0))
                    ])
                    if algorithm_alphas.size == 0:
                        raise ValueError("Exact BR needs at least one α value among 0, 1, and 2.")
                run_output = run_multiresource_alpha_experiment(
                    mr_n, mr_capacities, mr_valuations, mr_budgets, algorithm_alphas,
                    algorithm, int(mr_iterations), float(mr_minimum_bid),
                    effective_repetitions, delta=float(mr_delta),
                    convergence_tolerance=effective_tolerance,
                    player_alphas=mr_player_alphas,
                    step_scale=effective_step_scale,
                    gradient_bound_mode=gradient_bound_mode,
                    residual_mode=residual_mode,
                    requirements=mr_requirements,
                    residual_evaluation_interval=int(mr_residual_interval),
                )
                mr_algorithm_results[algorithm] = {
                    "records": run_output[0], "allocations": run_output[1],
                    "convergence_histories": run_output[2],
                    "resource_convergence_histories": run_output[3],
                }
            primary_algorithm = mr_algorithms[0]
            primary = mr_algorithm_results[primary_algorithm]
            mr_records = primary["records"]
            mr_allocations = primary["allocations"]
            mr_convergence_histories = primary["convergence_histories"]
            mr_resource_convergence_histories = primary["resource_convergence_histories"]
        st.session_state.mr_results = {
            "records": mr_records,
            "allocations": mr_allocations,
            "capacities": mr_capacities,
            "algorithm": primary_algorithm,
            "algorithms": list(mr_algorithms),
            "algorithm_results": mr_algorithm_results,
            "delta": float(mr_delta),
            "alpha_mode": mr_alpha_mode,
            "theta": mr_theta,
            "interest_factors": mr_valuations,
            "requirements": mr_requirements,
            "step_scale": effective_step_scale,
            "convergence_tolerance": effective_tolerance,
            "repetitions": effective_repetitions,
            "gradient_bound_mode": gradient_bound_mode,
            "residual_mode": residual_mode,
            "residual_evaluation_interval": int(mr_residual_interval),
            "convergence_histories": mr_convergence_histories,
            "resource_convergence_histories": mr_resource_convergence_histories,
            "iterations": int(mr_iterations),
            "plot_iterations": min(
                int(cfg["T_plot"]), int(mr_iterations)
            ),
        }
    except Exception as exc:
        st.error(f"Multi-resource experiment failed: {exc}")

if (
    "mr_results" in st.session_state
    and (
        "convergence_histories" not in st.session_state.mr_results
        or "resource_convergence_histories" not in st.session_state.mr_results
        or "residual_mode" not in st.session_state.mr_results
        or "algorithm_results" not in st.session_state.mr_results
    )
):
    del st.session_state.mr_results
    st.info(
        "The convergence export now records the selected residual definition. "
        "Please rerun the multi-resource experiment once."
    )

if "mr_results" in st.session_state:
    mr_result = st.session_state.mr_results
    # T_plot is a live presentation control. Updating it redraws stored
    # trajectories and PDFs without recomputing the simulation.
    mr_result["plot_iterations"] = min(
        int(cfg["T_plot"]),
        int(mr_result.get("iterations", cfg["T_plot"])),
    )
    mr_records = mr_result["records"]
    alpha_values = [row["alpha"] for row in mr_records]
    per_device_alpha = mr_result.get("alpha_mode") == "One α per device"
    alpha_axis_title = "Mean αᵢ" if per_device_alpha else "α"

    tradeoff_fig = go.Figure()
    tradeoff_fig.add_trace(go.Scatter(
        x=alpha_values,
        y=[row["efficiency"] for row in mr_records],
        name="Weighted efficiency",
        mode="lines+markers",
    ))
    tradeoff_fig.add_trace(go.Scatter(
        x=alpha_values,
        y=[row["jain"] for row in mr_records],
        name="Jain fairness",
        mode="lines+markers",
    ))
    tradeoff_fig.update_layout(
        title=f"Efficiency and fairness ({mr_result['algorithm']})",
        xaxis_title=alpha_axis_title,
        yaxis_title="Index (0–1)",
        yaxis_range=[0, 1.05],
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(tradeoff_fig, use_container_width=True)

    per_resource_fig = go.Figure()
    for resource in range(len(mr_result["capacities"])):
        per_resource_fig.add_trace(go.Scatter(
            x=alpha_values,
            y=[row["jain_by_resource"][resource] for row in mr_records],
            mode="lines+markers",
            name=f"Jain — resource {resource + 1}",
        ))
        per_resource_fig.add_trace(go.Scatter(
            x=alpha_values,
            y=[row["residual_by_resource"][resource] for row in mr_records],
            mode="lines+markers",
            name=f"Residual — resource {resource + 1}",
            yaxis="y2",
        ))
    per_resource_fig.update_layout(
        title="Per-resource fairness and final residual",
        xaxis_title=alpha_axis_title,
        yaxis=dict(title="Jain index", range=[0, 1.05]),
        yaxis2=dict(title="Residual", type="log", overlaying="y", side="right"),
        template="plotly_white",
    )
    st.plotly_chart(per_resource_fig, use_container_width=True)

    utility_fig = go.Figure(go.Scatter(
        x=alpha_values,
        y=[row["total_utility"] for row in mr_records],
        mode="lines+markers",
        name="Total α-fair utility",
    ))
    utility_fig.update_layout(
        title=r"Total utility $\sum_i\sum_r a_i^r V^{\alpha_i}(x_i^r)$",
        xaxis_title=alpha_axis_title,
        yaxis_title="Total utility",
        template="plotly_white",
    )
    st.plotly_chart(utility_fig, use_container_width=True)

    convergence_fig = go.Figure(go.Scatter(
        x=alpha_values,
        y=[
            row["convergence_residual"]
            for row in mr_records
        ],
        mode="lines+markers",
        name=(
            "Iterate-difference residual"
            if mr_result["residual_mode"] == "iterate_difference"
            else "Best-response residual"
        ),
        marker=dict(
            color=["green" if row["converged"] else "red" for row in mr_records]
        ),
    ))
    convergence_fig.update_layout(
        title="Convergence diagnostic (green = all repetitions converged)",
        xaxis_title=alpha_axis_title,
        yaxis_title=(
            "‖zᵗ − zᵗ⁺¹‖₂ (log scale)"
            if mr_result["residual_mode"] == "iterate_difference"
            else "‖zᴮᴿ(zᵗ) − zᵗ‖₂ (log scale)"
        ),
        yaxis_type="log",
        template="plotly_white",
    )
    st.plotly_chart(convergence_fig, use_container_width=True)

    convergence_time_fig = go.Figure()
    algorithm_results = mr_result.get("algorithm_results", {
        mr_result["algorithm"]: {
            "records": mr_records,
            "convergence_histories": mr_result["convergence_histories"],
        }
    })
    mr_available_iterations = max(
        len(values) - (1 if mr_result["residual_mode"] == "best_response" else 0)
        for output in algorithm_results.values()
        for values in output["convergence_histories"].values()
    )
    mr_plot_iterations = min(
        int(mr_result.get("plot_iterations", mr_available_iterations)),
        mr_available_iterations,
    )
    dash_styles = ["solid", "dash", "dot", "dashdot"]
    plotly_marker_symbols = {
        "o": "circle", "s": "square", "D": "diamond", "d": "diamond",
        "^": "triangle-up", "v": "triangle-down", "<": "triangle-left",
        ">": "triangle-right", "*": "star", "p": "pentagon",
        "P": "cross", "X": "x", "H": "hexagon", "x": "x",
    }
    all_residual_alphas = sorted({
        float(row["alpha"])
        for output in algorithm_results.values() for row in output["records"]
    })
    alpha_dash = {
        value: dash_styles[index % len(dash_styles)]
        for index, value in enumerate(all_residual_alphas)
    }
    for algorithm_index, (algorithm, output) in enumerate(algorithm_results.items()):
        styled_algorithm = legend_map.get(algorithm, algorithm)
        algorithm_color = COLORS_METHODS.get(
            styled_algorithm, colors[algorithm_index % len(colors)]
        )
        matplotlib_marker = MARKERS_METHODS.get(
            styled_algorithm, markers[algorithm_index % len(markers)]
        )
        for row in output["records"]:
            alpha_value = float(row["alpha"])
            residual_history = np.asarray(
                output["convergence_histories"][alpha_value]
            )[:mr_plot_iterations + 1]
            if per_device_alpha:
                alpha_name = "αᵢ = (" + ", ".join(
                    f"{value:g}" for value in row["alpha_values"]
                ) + ")"
            else:
                alpha_name = f"α = {alpha_value:g}"
            convergence_time_fig.add_trace(go.Scatter(
                x=(
                    np.arange(len(residual_history))
                    if mr_result["residual_mode"] == "best_response"
                    else np.arange(1, len(residual_history) + 1)
                ),
                y=residual_history,
                mode="lines+markers",
                name=f"{algorithm} — {alpha_name}",
                line=dict(color=algorithm_color, dash=alpha_dash[alpha_value]),
                marker=dict(
                    color=algorithm_color,
                    symbol=plotly_marker_symbols.get(matplotlib_marker, "circle"),
                    size=6,
                ),
            ))
    convergence_time_fig.update_layout(
        title=(
            f"Residual over time ({', '.join(algorithm_results)}, "
            f"iterations 0–{mr_plot_iterations})"
        ),
        xaxis_title="Iteration (t)",
        yaxis_title=(
            "‖zᵗ − zᵗ⁺¹‖₂"
            if mr_result["residual_mode"] == "iterate_difference"
            else "‖zᴮᴿ(zᵗ) − zᵗ‖₂"
        ),
        yaxis_type="log",
        template="plotly_white",
    )
    st.plotly_chart(convergence_time_fig, use_container_width=True)
    converged_count = sum(row["converged"] for row in mr_records)
    if converged_count == len(mr_records):
        st.success("The algorithm converged for every tested α and every repetition.")
    else:
        st.warning(
            f"Converged for {converged_count}/{len(mr_records)} tested α values "
            "in every repetition. Increase iterations or tune the learning rate "
            "for the red points."
        )

    st.caption(
        "Resource importance θ = "
        + ", ".join(f"{value:g}" for value in mr_result["theta"])
        + f"; OGD step-size scale κ = {mr_result.get('step_scale', 1.0):g}"
        + "; gradient bound = "
        + (
            "domain ℓ∞"
            if mr_result.get("gradient_bound_mode") == "infinity"
            else (
                "legacy Run Simulation"
                if mr_result.get("gradient_bound_mode") == "legacy"
                else "practical"
            )
        )
        + (
            f"; exact BR residual sampled every "
            f"{mr_result.get('residual_evaluation_interval', 1)} iteration(s)"
            if mr_result["residual_mode"] == "best_response"
            else ""
        )
    )
    with st.expander("Required-resource subsets"):
        st.dataframe(
            {
                f"Resource {resource + 1}": mr_result["requirements"][:, resource]
                for resource in range(mr_result["requirements"].shape[1])
            },
            use_container_width=True,
        )
    with st.expander("Interest factors aᵢᵏ = aᵢ θₖ"):
        st.dataframe(
            {
                f"Resource {resource + 1}": mr_result["interest_factors"][:, resource]
                for resource in range(mr_result["interest_factors"].shape[1])
            },
            use_container_width=True,
        )
    if per_device_alpha:
        st.info(
            "Per-device α values: "
            + ", ".join(
                f"α{i + 1}={value:g}"
                for i, value in enumerate(mr_records[0]["alpha_values"])
            )
            + ". Each value is shared across that device's resources."
        )

    capacity_total = float(np.sum(mr_result["capacities"]))
    if mr_result.get("delta", 0.0) == 0.0:
        st.info(f"With δ=0, total allocated capacity is {capacity_total:.6g}.")
    else:
        st.info(
            f"With δ={mr_result['delta']:g}, some capacity remains unallocated "
            "according to the proportional-allocation denominator."
        )

    allocation_algorithm = st.selectbox(
        "Inspect allocations for algorithm",
        options=list(algorithm_results),
        key="mr_allocation_algorithm",
    )
    allocation_output = algorithm_results[allocation_algorithm]
    allocation_alpha_values = [
        float(row["alpha"]) for row in allocation_output["records"]
    ]
    selected_alpha = st.select_slider(
        "Inspect final allocations at α",
        options=allocation_alpha_values,
        value=allocation_alpha_values[-1],
        key=f"mr_selected_alpha_{allocation_algorithm}",
    )
    selected_x = allocation_output["allocations"][float(selected_alpha)]
    resource_residual_fig = go.Figure()
    selected_resource_history = np.asarray(
        allocation_output["resource_convergence_histories"][float(selected_alpha)]
    )[:mr_plot_iterations + 1]
    resource_iterations = (
        np.arange(len(selected_resource_history))
        if mr_result["residual_mode"] == "best_response"
        else np.arange(1, len(selected_resource_history) + 1)
    )
    for resource in range(selected_resource_history.shape[1]):
        resource_residual_fig.add_trace(go.Scatter(
            x=resource_iterations,
            y=selected_resource_history[:, resource],
            mode="lines",
            name=f"Resource {resource + 1}",
        ))
    resource_residual_fig.update_layout(
        title=f"Residual by resource ({allocation_algorithm}, α={selected_alpha:g})",
        xaxis_title="Iteration (t)",
        yaxis_title="Per-resource residual",
        yaxis_type="log",
        template="plotly_white",
    )
    st.plotly_chart(resource_residual_fig, use_container_width=True)

    allocation_fig = go.Figure()
    for resource in range(selected_x.shape[1]):
        allocation_fig.add_trace(go.Bar(
            x=[f"Player {i + 1}" for i in range(selected_x.shape[0])],
            y=selected_x[:, resource],
            name=f"Resource {resource + 1}",
        ))
    allocation_fig.update_layout(
        barmode="group",
        title=f"Final allocations ({allocation_algorithm}, α={selected_alpha:g})",
        xaxis_title="Player",
        yaxis_title="Allocation xᵢʳ",
        template="plotly_white",
    )
    st.plotly_chart(allocation_fig, use_container_width=True)

    resource_totals = selected_x.sum(axis=0)
    selected_shares = np.divide(
        selected_x,
        resource_totals[None, :],
        out=np.zeros_like(selected_x),
        where=resource_totals[None, :] > 0,
    )
    resource_share_fig = go.Figure()
    for player in range(selected_x.shape[0]):
        resource_share_fig.add_trace(go.Bar(
            x=[f"Resource {resource + 1}" for resource in range(selected_x.shape[1])],
            y=selected_shares[player],
            name=f"Player {player + 1}",
        ))
    resource_share_fig.update_layout(
        barmode="stack",
        title=f"Player share by resource ({allocation_algorithm}, α={selected_alpha:g})",
        xaxis_title="Resource",
        yaxis_title="Share of allocated resource",
        yaxis_range=[0, 1],
        template="plotly_white",
    )
    st.plotly_chart(resource_share_fig, use_container_width=True)

    export_signature = (
        tuple(algorithm_results), allocation_algorithm, float(selected_alpha),
        int(mr_result.get("plot_iterations", 0)),
    )
    if st.button("📄 Prepare multi-resource PDFs", key="mr_prepare_pdfs"):
        try:
            export_result = dict(mr_result)
            export_result.update({
                "algorithm": allocation_algorithm,
                "records": allocation_output["records"],
                "allocations": allocation_output["allocations"],
                "convergence_histories": allocation_output["convergence_histories"],
                "resource_convergence_histories": allocation_output["resource_convergence_histories"],
            })
            mr_pdf_paths = save_multiresource_figures_pdf(export_result, selected_alpha)
            st.session_state.mr_pdf_exports = {
                key: {
                    "name": os.path.basename(path),
                    "data": Path(path).read_bytes(),
                }
                for key, path in mr_pdf_paths.items()
            }
            st.session_state.mr_pdf_signature = export_signature
        except Exception as exc:
            st.error(
                "Could not prepare multi-resource PDFs "
                f"({type(exc).__name__}): {exc!r}"
            )

    if (
        st.session_state.get("mr_pdf_signature") == export_signature
        and "mr_pdf_exports" in st.session_state
    ):
        st.subheader("📂 Download multi-resource figures (PDF)")
        labels = {
            "tradeoff": "⬇️ Trade-off PDF",
            "convergence": "⬇️ Residual-over-time PDF",
            "convergence_legend": "⬇️ Residual legend PDF",
            "allocation": "⬇️ Allocation by player PDF",
            "resource_share": "⬇️ Player share by resource PDF",
        }
        pdf_columns = st.columns(3)
        for index, (figure_key, label) in enumerate(labels.items()):
            exported = st.session_state.mr_pdf_exports[figure_key]
            pdf_columns[index % len(pdf_columns)].download_button(
                label,
                exported["data"],
                file_name=exported["name"],
                mime="application/pdf",
                key=f"mr_download_{figure_key}_{allocation_algorithm}_{selected_alpha}",
            )

    st.dataframe(mr_records, use_container_width=True)
