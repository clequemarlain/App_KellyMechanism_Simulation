# EdgeKelly Lab

EdgeKelly Lab is an interactive research simulator for α-fair resource allocation in Kelly mechanism games. It compares decentralized learning algorithms, measures convergence and welfare, studies heterogeneity, and supports single-resource and multi-resource experiments.

[Open the deployed application](https://app-kellymechanism-simulation.streamlit.app/)

## Model

Player `i` submits a nonnegative bid `z_i`. With capacity `C` and allocation slack `δ`, its allocation is `x_i = C z_i / (Σ_j z_j + δ)`. Its valuation uses the α-fair family:

```text
V⁰(x) = x
V¹(x) = log(x)
Vᵅ(x) = x^(1-α)/(1-α), otherwise.
```

The payoff combines weighted utility and bid cost. Valuation heterogeneity is controlled by `γ`; budget heterogeneity is controlled by `μ`. The multi-resource model applies proportional allocation to every resource, supports player-specific required-resource subsets, and gives each player one shared bid budget.

## Application pages

- **Simulator** contains all runnable experiments and global configuration.
- **Experiment guide** helps new users choose a workflow and interpret its output.
- **Methodology** summarizes the model, algorithms, residuals, and reproducibility requirements.

## Experiments

1. **Learning dynamics** compares bid, payoff, welfare, fairness, or residual trajectories for selected algorithms.
2. **Convergence under heterogeneity** records the first iteration at which each method satisfies the Nash-residual tolerance.
3. **Parameter-sweep tables** summarize repeated simulations over configured player-count and heterogeneity grids. Separate tables report bid-space Nash convergence and payoff-gain (ε-Nash) convergence; each method has distinct iteration and minimum-residual columns.
4. **Fairness versus heterogeneity** measures Jain's index as `γ` and the number of players change.
5. **Linear versus logarithmic utility** compares α = 0 with α = 1 under the same heterogeneity values.
6. **Multi-resource allocation** compares algorithms across several capacities, resource requirements, and importance factors.

The multi-resource workflow displays aggregate and per-resource fairness, residual trajectories, final allocations `x_i^r`, and each player's share of every resource. Exact multi-resource BR is available for α = 0, 1, and 2 using closed-form resource responses and scalar KKT bisection for a binding shared budget.

## Learning algorithms

| Algorithm | Principle | Schedule |
|---|---|---|
| BR | Synchronous best response | No learning rate |
| NumBR | Numerically optimized best response | No learning rate |
| OGD_F | Projected current-gradient update | Fixed horizon, proportional to `1/√T` |
| OGD_V | Projected current-gradient update | Varying, proportional to `1/√t` |
| DAQ_F | Quadratic dual averaging of cumulative gradients | Fixed horizon |
| DAQ_V | Quadratic dual averaging of cumulative gradients | Varying |
| RRM_V | Reinforcement-style accumulation of scaled gradients | Varying |
| DAE / XL | Score-based learning variants | Configured in the simulator |
| Hybrid | Different player groups use different rules | Group-specific |

Stable plot styles are used throughout: BR is purple/star, OGD_F green/down-triangle, OGD_V teal/up-triangle, DAQ_F dark orange/square, and RRM_V magenta/pentagon.

## Important parameters

| Parameter | Meaning |
|---|---|
| `n` | Number of players |
| `T`, `T_plot` | Simulation and displayed horizons |
| `Nb_random_sim` | Independent random repetitions |
| `α` | Utility curvature / fairness parameter |
| `a_i`, `γ` | Player valuation and valuation heterogeneity |
| `c_i`, `μ` | Player budget and budget heterogeneity |
| `ε`, `δ`, `λ` | Minimum bid, allocation slack, and bid price |
| `tol` | Convergence tolerance |

Advanced values and sweep grids are in the sidebar. Use **Download config JSON** to retain the configuration with results.

## Diagnostics

The best-response residual `R_BR(t) = ||BR(z(t)) - z(t)||₂` directly tests the Nash fixed-point condition. The cheaper iterate residual `R_step(t) = ||z(t+1) - z(t)||₂` measures motion, but a small step is not always proof of Nash convergence. Always report the selected definition.

Jain's index lies between 0 and 1, with values closer to 1 representing a more even allocation. Normalized efficiency is better when larger; efficiency loss and Price of Anarchy are better when smaller.

## Runtime guidance

Runtime grows approximately with `players × resources × iterations × algorithms × repetitions`. Start with 5–10 players, 500 iterations, one repetition, and two algorithms. Exact BR residuals are expensive; use a multi-resource sampling interval above 1 for exploration and interval 1 only when every iteration is required. Converged multi-resource runs stop internally while preserving fixed-length curves.

## Exports

The experiments generate publication-oriented PDFs. For multi-resource results:

1. select the allocation algorithm and α;
2. press **Prepare multi-resource PDFs**;
3. download residual-over-time, legend, allocation-by-player, player-share-by-resource, or trade-off PDFs.

Prepared files persist across Streamlit reruns. Tables are downloadable from Streamlit, and the configuration is available as JSON.

## Local installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python launch.py
```

Useful alternatives are `python launch.py --port 8502`, `python launch.py --no-browser`, and `streamlit run app.py`. In PyCharm, select `.venv/bin/python` and run the bundled **Kelly Simulator** configuration.

## Deployment

For Streamlit Community Cloud, push the repository to GitHub, create an app, select `app.py`, and use Python 3.10+. On another host, run `streamlit run app.py --server.address 0.0.0.0`. The `.streamlit/config.toml` file supplies the shared visual theme.

## Repository structure

```text
app.py                         Main simulator and experiment UI
pages/                         Experiment guide and methodology pages
launch.py                      Local/PyCharm launcher
main.py                        Main simulation runner
main_table_simulation.py       Parameter-sweep table runner
linear_VS_log.py               Utility-curvature comparison
src/game/config.py             Default configuration
src/game/description.py        Algorithm descriptions
src/game/utils.py              Game logic, learning rules, metrics, plots
Journal2025/                   Publication plotting style and figures
.streamlit/config.toml         Application theme
requirements.txt               Python dependencies
```

## Reproducibility checklist

Retain the configuration JSON, random-repetition count, initialization policy, algorithm names, learning rates, horizons, tolerance, α, γ, μ, ε, δ, price, budgets, residual definition and sampling interval, and any selected player/resource or hybrid-group definitions.
