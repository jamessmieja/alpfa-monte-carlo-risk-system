import numpy as np
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Core simulation engine
# ─────────────────────────────────────────────

def cholesky_correlated(n: int, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate two correlated standard normal samples via Cholesky decomposition."""
    L = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho**2)]])
    z = np.random.standard_normal((2, n))
    corr = L @ z
    return corr[0], corr[1]


def antithetic_poisson(lam: float, n: int) -> np.ndarray:
    """Poisson samples with antithetic variate variance reduction."""
    half = n // 2
    u = np.random.uniform(size=half)
    u_anti = 1 - u
    s1 = stats.poisson.ppf(u, mu=lam).astype(int)
    s2 = stats.poisson.ppf(u_anti, mu=lam).astype(int)
    return np.concatenate([s1, s2])


def lognormal_sample(mu_log: float, sigma_log: float, n: int) -> np.ndarray:
    return np.random.lognormal(mean=mu_log, sigma=sigma_log, size=n)


def pareto_sample(xm: float, alpha: float, n: int) -> np.ndarray:
    """Fat-tailed Pareto — captures rare large cost surprises."""
    u = np.random.uniform(size=n)
    return xm / np.power(u, 1.0 / alpha)


def pert_sample(low: float, mode: float, high: float, n: int) -> np.ndarray:
    """PERT distribution via Beta — min/most-likely/max parameterization."""
    mean = (low + 4 * mode + high) / 6
    std  = (high - low) / 6
    alpha = ((mean - low) / (high - low)) * ((mean - low) * (high - mean) / std**2 - 1)
    beta  = alpha * (high - mean) / (mean - low)
    alpha = max(alpha, 0.5)
    beta  = max(beta,  0.5)
    return low + (high - low) * np.random.beta(alpha, beta, size=n)


# ─────────────────────────────────────────────
# Risk metrics
# ─────────────────────────────────────────────

def compute_var(pnl: np.ndarray, confidence: float = 0.95) -> float:
    """Value at Risk — maximum loss at given confidence level."""
    return float(np.percentile(pnl, (1 - confidence) * 100))


def compute_cvar(pnl: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) — average loss beyond VaR."""
    var = compute_var(pnl, confidence)
    tail = pnl[pnl <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def sensitivity_analysis(params: dict, n: int = 5000) -> dict[str, float]:
    """
    One-at-a-time sensitivity (Sobol-style).
    Shocks each parameter +/- and measures P&L impact delta.
    """
    base_pnl = run_simulation(n=n, **params)["pnl"]
    base_mean = base_pnl.mean()

    shocks = {
        "ticket_price":        ("ticket_price",       params["ticket_price"] * 1.20),
        "expected_attendees":  ("expected_attendees",  params["expected_attendees"] * 1.20),
        "sponsor_max":         ("sponsor_max",         params["sponsor_max"] * 0.50),
        "sponsor_prob":        ("sponsor_prob",        params["sponsor_prob"] * 0.50),
        "venue_cost":          ("venue_cost",          params["venue_cost"] * 1.30),
        "catering_per_head":   ("catering_per_head",   params["catering_per_head"] * 1.30),
        "fixed_costs":         ("fixed_costs",         params["fixed_costs"] * 1.30),
        "surprise_cap":        ("surprise_cap",        params["surprise_cap"] * 1.50),
    }

    impacts = {}
    for label, (key, shocked_val) in shocks.items():
        shocked_params = {**params, key: shocked_val}
        shocked_pnl = run_simulation(n=n, **shocked_params)["pnl"]
        impacts[label] = float(shocked_pnl.mean() - base_mean)

    return dict(sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True))


# ─────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────

def run_simulation(
    n: int                  = 10_000,
    ticket_price: float     = 25.0,
    expected_attendees: int = 90,
    sponsor_max: float      = 1500.0,
    sponsor_prob: float     = 0.65,
    venue_cost: float       = 600.0,
    catering_per_head: float= 18.0,
    fixed_costs: float      = 400.0,
    surprise_cap: float     = 200.0,
    rho_attend_rev: float   = 0.82,
) -> dict:
    """
    Monte Carlo simulation of event P&L.

    Distributions:
      - Attendance:   Poisson (count data) with antithetic variate sampling
      - Sponsorship:  Bernoulli × LogNormal (binary commit × variable payout)
      - Surprise costs: Pareto tail (fat-tailed, captures rare blowouts)
      - Catering:     PERT (bounded min/mode/max)
      - Correlated factors via Cholesky decomposition (Gaussian copula)

    Returns a dict with arrays and summary statistics.
    """
    rng_seed = None  # set an int here for reproducibility

    # Correlated noise: attendance ↔ revenue (ρ = rho_attend_rev)
    z_attend, z_rev = cholesky_correlated(n, rho_attend_rev)

    # ── Revenue ──────────────────────────────
    base_attendance = antithetic_poisson(expected_attendees, n)
    attendance = np.maximum(
        0,
        (base_attendance + np.sqrt(expected_attendees) * z_attend).round().astype(int)
    )

    ticket_revenue = attendance * ticket_price * (1 + 0.05 * z_rev)

    # Sponsorship: Bernoulli gate × LogNormal amount
    sponsor_commits = np.random.uniform(size=n) < sponsor_prob
    sponsor_mu      = np.log(sponsor_max * 0.60)
    sponsor_amount  = sponsor_commits * lognormal_sample(sponsor_mu, 0.40, n)

    # Extra marketing spend when sponsor drops out (negative cross-effect)
    extra_marketing = (~sponsor_commits) * fixed_costs * 0.30

    merch = lognormal_sample(np.log(80), 0.50, n)

    total_revenue = ticket_revenue + sponsor_amount + merch

    # ── Costs ────────────────────────────────
    venue_actual    = venue_cost + venue_cost * 0.10 * np.random.standard_normal(n)
    catering_actual = pert_sample(catering_per_head * 0.85, catering_per_head,
                                  catering_per_head * 1.30, n)
    catering_total  = attendance * catering_actual

    # Fat-tailed surprise costs (Pareto α=2.5)
    surprise_cost = np.minimum(surprise_cap * 2, pareto_sample(surprise_cap * 0.10, 2.5, n))

    total_cost = venue_actual + catering_total + fixed_costs + extra_marketing + surprise_cost

    pnl = total_revenue - total_cost

    return {
        "pnl":           pnl,
        "attendance":    attendance,
        "total_revenue": total_revenue,
        "total_cost":    total_cost,
        "sponsor_rate":  float(sponsor_commits.mean()),
    }


# ─────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────

def print_report(results: dict, params: dict, confidence: float = 0.95):
    pnl  = results["pnl"]
    var  = compute_var(pnl, confidence)
    cvar = compute_cvar(pnl, confidence)

    c = int(confidence * 100)
    print("=" * 52)
    print("  ALPFA Event P&L Risk Report")
    print("=" * 52)
    print(f"  Simulations run:       {len(pnl):,}")
    print(f"  Confidence level:      {c}%")
    print("-" * 52)
    print(f"  Expected P&L:          ${pnl.mean():>+8.0f}")
    print(f"  Std deviation:         ${pnl.std():>8.0f}")
    print(f"  Median P&L:            ${np.median(pnl):>+8.0f}")
    print("-" * 52)
    print(f"  Break-even prob:       {(pnl >= 0).mean() * 100:>7.1f}%")
    print(f"  P(profit > $500):      {(pnl > 500).mean() * 100:>7.1f}%")
    print(f"  P(loss > $1,000):      {(pnl < -1000).mean() * 100:>7.1f}%")
    print("-" * 52)
    print(f"  VaR  ({c}%):          ${var:>+8.0f}   ← max loss, {c}% conf.")
    print(f"  CVaR ({c}%):          ${cvar:>+8.0f}   ← avg loss in tail")
    print(f"  Recommended reserve:   ${abs(cvar):>8.0f}")
    print("-" * 52)
    print(f"  5th  pct P&L:          ${np.percentile(pnl, 5):>+8.0f}")
    print(f"  25th pct P&L:          ${np.percentile(pnl, 25):>+8.0f}")
    print(f"  75th pct P&L:          ${np.percentile(pnl, 75):>+8.0f}")
    print(f"  95th pct P&L:          ${np.percentile(pnl, 95):>+8.0f}")
    print("=" * 52)

    print("\n  Sensitivity Analysis (P&L impact of 1-factor shock):")
    print("  " + "-" * 48)
    sens = sensitivity_analysis(params, n=3000)
    for var_name, impact in sens.items():
        bar = "█" * int(abs(impact) / max(abs(v) for v in sens.values()) * 20)
        direction = "▲ revenue" if impact > 0 else "▼ cost"
        print(f"  {var_name:<22} {bar:<20} ${impact:>+7.0f}  ({direction})")
    print("=" * 52)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    params = dict(
        ticket_price      = 25.0,
        expected_attendees= 90,
        sponsor_max       = 1500.0,
        sponsor_prob      = 0.65,
        venue_cost        = 600.0,
        catering_per_head = 18.0,
        fixed_costs       = 400.0,
        surprise_cap      = 200.0,
    )

    print("Running simulation...")
    results = run_simulation(n=10_000, **params)
    print_report(results, params)
