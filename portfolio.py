"""
portfolio.py — Semester Event Portfolio Risk
─────────────────────────────────────────────
Models a full semester of ALPFA events simultaneously with correlated
outcomes. Computes portfolio-level VaR, CVaR, and diversification benefit.

Finance analogy: each event is an asset; the semester is a portfolio.
Cross-event correlation is modeled via a full Cholesky factor matrix.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from simulation import run_simulation, compute_var, compute_cvar


# ─────────────────────────────────────────────
# Event definition
# ─────────────────────────────────────────────

@dataclass
class EventConfig:
    name: str
    event_type: str                    # networking | workshop | fundraiser | social | speaker
    ticket_price: float       = 25.0
    expected_attendees: int   = 80
    sponsor_max: float        = 1000.0
    sponsor_prob: float       = 0.60
    venue_cost: float         = 500.0
    catering_per_head: float  = 15.0
    fixed_costs: float        = 300.0
    surprise_cap: float       = 150.0
    month: int                = 1      # 1–4 within semester

    # Cross-event attendance spillover weight (0 = independent, 1 = fully linked)
    momentum_weight: float    = 0.25


# ─────────────────────────────────────────────
# Default semester templates
# ─────────────────────────────────────────────

SEMESTER_TEMPLATES = {
    "Fall": [
        EventConfig("Kickoff Mixer",        "social",       ticket_price=15,  expected_attendees=120, sponsor_max=800,  sponsor_prob=0.55, venue_cost=400,  catering_per_head=12, fixed_costs=250, surprise_cap=100, month=1),
        EventConfig("Resume Workshop",      "workshop",     ticket_price=10,  expected_attendees=80,  sponsor_max=500,  sponsor_prob=0.70, venue_cost=200,  catering_per_head=8,  fixed_costs=150, surprise_cap=80,  month=2),
        EventConfig("Corporate Panel",      "speaker",      ticket_price=20,  expected_attendees=100, sponsor_max=2000, sponsor_prob=0.65, venue_cost=600,  catering_per_head=15, fixed_costs=400, surprise_cap=200, month=2),
        EventConfig("Networking Gala",      "networking",   ticket_price=35,  expected_attendees=150, sponsor_max=3000, sponsor_prob=0.60, venue_cost=1200, catering_per_head=25, fixed_costs=500, surprise_cap=300, month=3),
        EventConfig("Fundraiser Dinner",    "fundraiser",   ticket_price=50,  expected_attendees=90,  sponsor_max=2500, sponsor_prob=0.70, venue_cost=900,  catering_per_head=30, fixed_costs=350, surprise_cap=200, month=4),
    ],
    "Spring": [
        EventConfig("Spring Kickoff",       "social",       ticket_price=15,  expected_attendees=100, sponsor_max=700,  sponsor_prob=0.50, venue_cost=350,  catering_per_head=12, fixed_costs=200, surprise_cap=100, month=1),
        EventConfig("Finance Case Workshop","workshop",     ticket_price=15,  expected_attendees=70,  sponsor_max=600,  sponsor_prob=0.65, venue_cost=250,  catering_per_head=10, fixed_costs=200, surprise_cap=80,  month=2),
        EventConfig("Industry Night",       "networking",   ticket_price=25,  expected_attendees=110, sponsor_max=2000, sponsor_prob=0.60, venue_cost=700,  catering_per_head=20, fixed_costs=400, surprise_cap=200, month=3),
        EventConfig("ALPFA Gala",           "fundraiser",   ticket_price=60,  expected_attendees=130, sponsor_max=4000, sponsor_prob=0.65, venue_cost=1500, catering_per_head=35, fixed_costs=600, surprise_cap=400, month=4),
    ],
}


# ─────────────────────────────────────────────
# Cross-event correlation matrix by type
# ─────────────────────────────────────────────

# Events of similar type are more correlated (same audience pool, same sponsors)
TYPE_CORR = {
    ("networking", "networking"): 0.70,
    ("fundraiser", "fundraiser"): 0.65,
    ("workshop",   "workshop"):   0.55,
    ("social",     "social"):     0.60,
    ("speaker",    "networking"): 0.50,
    ("speaker",    "fundraiser"): 0.40,
    ("networking", "fundraiser"): 0.45,
    ("workshop",   "social"):     0.30,
}

def get_corr(type_a: str, type_b: str) -> float:
    key = tuple(sorted([type_a, type_b]))
    return TYPE_CORR.get(key, 0.25)   # default low correlation


def build_corr_matrix(events: list[EventConfig]) -> np.ndarray:
    """Build full correlation matrix across all events."""
    n = len(events)
    C = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            # Reduce correlation for events far apart in time
            time_decay = max(0.1, 1 - 0.15 * abs(events[i].month - events[j].month))
            base_corr  = get_corr(events[i].event_type, events[j].event_type)
            rho = base_corr * time_decay
            C[i, j] = rho
            C[j, i] = rho
    return C


def cholesky_factor(C: np.ndarray) -> np.ndarray:
    """Compute Cholesky factor with jitter for numerical stability."""
    jitter = 1e-6
    try:
        return np.linalg.cholesky(C + jitter * np.eye(len(C)))
    except np.linalg.LinAlgError:
        # Fallback: nearest PSD matrix via eigenvalue clipping
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, jitter)
        C_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return np.linalg.cholesky(C_psd)


# ─────────────────────────────────────────────
# Portfolio simulation
# ─────────────────────────────────────────────

def run_portfolio_simulation(
    events: list[EventConfig],
    n: int = 10_000,
    confidence: float = 0.95,
    momentum: bool = True,
) -> dict:
    """
    Simulate a semester portfolio of events with correlated P&L.

    Correlation structure:
    - Cross-event type correlation (networking events move together, etc.)
    - Time-decay: events further apart in the semester are less correlated
    - Momentum: early event attendance predicts later event attendance

    Returns per-event P&L arrays + portfolio-level risk metrics.
    """
    k = len(events)
    C = build_corr_matrix(events)
    L = cholesky_factor(C)

    # Correlated standard normals: shape (k, n)
    Z = np.random.standard_normal((k, n))
    corr_Z = L @ Z    # shape (k, n)

    event_pnls    = []
    event_results = []
    attendance_history = np.zeros(n)   # running avg for momentum

    for i, ev in enumerate(events):
        # Apply momentum: early attendance predicts later
        momentum_boost = 0.0
        if momentum and i > 0:
            # Positive early-semester attendance → higher later attendance
            momentum_boost = ev.momentum_weight * (attendance_history / max(1, ev.expected_attendees) - 1)

        # Inject correlated noise into attendance
        attend_noise = corr_Z[i]
        attend_sd    = np.sqrt(ev.expected_attendees)
        attendance   = np.maximum(
            0,
            np.round(ev.expected_attendees * (1 + momentum_boost) + attend_sd * attend_noise).astype(int)
        )

        # Revenue
        ticket_rev      = attendance * ev.ticket_price * (1 + 0.04 * corr_Z[i])
        sponsor_commits = np.random.uniform(size=n) < ev.sponsor_prob
        sponsor_mu      = np.log(ev.sponsor_max * 0.60)
        sponsor_amt     = sponsor_commits * np.random.lognormal(sponsor_mu, 0.40, n)
        extra_mktg      = (~sponsor_commits) * ev.fixed_costs * 0.30
        merch           = np.random.lognormal(np.log(60), 0.5, n)
        total_rev       = ticket_rev + sponsor_amt + merch

        # Costs
        venue_act    = ev.venue_cost * (1 + 0.10 * np.random.standard_normal(n))
        catering_act = np.random.triangular(
            ev.catering_per_head * 0.85,
            ev.catering_per_head,
            ev.catering_per_head * 1.30,
            n
        )
        catering_tot = attendance * catering_act
        surprise     = np.minimum(
            ev.surprise_cap * 2,
            ev.surprise_cap * 0.10 / np.random.power(2.5, n).clip(1e-9)
        )
        total_cost   = venue_act + catering_tot + ev.fixed_costs + extra_mktg + surprise

        pnl = total_rev - total_cost
        event_pnls.append(pnl)

        # Update attendance history for momentum
        attendance_history = (attendance_history * i + attendance) / (i + 1)

        event_results.append({
            "name":         ev.name,
            "type":         ev.event_type,
            "month":        ev.month,
            "pnl":          pnl,
            "attendance":   attendance,
            "revenue":      total_rev,
            "cost":         total_cost,
            "expected_pnl": float(pnl.mean()),
            "std":          float(pnl.std()),
            "var":          compute_var(pnl, confidence),
            "cvar":         compute_cvar(pnl, confidence),
            "breakeven_prob": float((pnl >= 0).mean()),
        })

    # Portfolio-level P&L (sum across events, per scenario)
    portfolio_pnl = np.sum(event_pnls, axis=0)

    # Diversification benefit: sum of individual VaRs vs portfolio VaR
    sum_individual_var = sum(r["var"] for r in event_results)
    portfolio_var      = compute_var(portfolio_pnl, confidence)
    diversification_benefit = sum_individual_var - portfolio_var   # should be positive

    # Correlation matrix for display
    corr_actual = np.corrcoef(event_pnls) if k > 1 else np.array([[1.0]])

    return {
        "events":                  event_results,
        "portfolio_pnl":           portfolio_pnl,
        "portfolio_var":           portfolio_var,
        "portfolio_cvar":          compute_cvar(portfolio_pnl, confidence),
        "portfolio_expected":      float(portfolio_pnl.mean()),
        "portfolio_std":           float(portfolio_pnl.std()),
        "portfolio_breakeven":     float((portfolio_pnl >= 0).mean()),
        "sum_individual_var":      sum_individual_var,
        "diversification_benefit": diversification_benefit,
        "corr_matrix":             corr_actual,
        "corr_matrix_assumed":     C,
        "event_pnls":              np.array(event_pnls),
        "n_events":                k,
        "n_sims":                  n,
    }


# ─────────────────────────────────────────────
# CLI report
# ─────────────────────────────────────────────

def print_portfolio_report(results: dict, confidence: float = 0.95):
    c = int(confidence * 100)
    print("=" * 60)
    print("  ALPFA Semester Portfolio Risk Report")
    print("=" * 60)

    for ev in results["events"]:
        sign = "✅" if ev["expected_pnl"] >= 0 else "⚠️ "
        print(f"\n  {sign} {ev['name']} ({ev['type']}, Month {ev['month']})")
        print(f"     Expected P&L:   ${ev['expected_pnl']:>+8.0f}  |  "
              f"Break-even: {ev['breakeven_prob']*100:.1f}%")
        print(f"     VaR ({c}%):    ${ev['var']:>+8.0f}  |  "
              f"CVaR: ${ev['cvar']:>+8.0f}")

    print("\n" + "=" * 60)
    print("  📊 PORTFOLIO SUMMARY")
    print("=" * 60)
    print(f"  Expected semester P&L:  ${results['portfolio_expected']:>+10.0f}")
    print(f"  Std deviation:          ${results['portfolio_std']:>10.0f}")
    print(f"  Break-even probability:  {results['portfolio_breakeven']*100:>9.1f}%")
    print(f"  Portfolio VaR ({c}%):  ${results['portfolio_var']:>+10.0f}")
    print(f"  Portfolio CVaR ({c}%): ${results['portfolio_cvar']:>+10.0f}")
    print(f"\n  Sum of individual VaRs: ${results['sum_individual_var']:>+10.0f}")
    print(f"  Diversification benefit:${results['diversification_benefit']:>+10.0f}  ← correlation reduces total risk")
    print(f"  Recommended reserve:    ${abs(results['portfolio_cvar']):>10.0f}")
    print("=" * 60)


if __name__ == "__main__":
    events = SEMESTER_TEMPLATES["Fall"]
    print("Running semester portfolio simulation...")
    results = run_portfolio_simulation(events, n=10_000)
    print_portfolio_report(results)
