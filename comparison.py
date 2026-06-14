"""
comparison.py — Event Format Comparison Engine
───────────────────────────────────────────────
Side-by-side Monte Carlo comparison of two event formats.
Outputs risk-adjusted metrics, probability that Format A beats Format B,
and a recommendation with reasoning.

Finance analogy: comparing two investment strategies on a risk-adjusted basis
using Sharpe ratio, drawdown, and probability of outperformance.
"""

import numpy as np
from dataclasses import dataclass
from simulation import run_simulation, compute_var, compute_cvar


# ─────────────────────────────────────────────
# Format presets
# ─────────────────────────────────────────────

@dataclass
class FormatConfig:
    name: str
    label: str
    ticket_price: float
    expected_attendees: int
    sponsor_max: float
    sponsor_prob: float
    venue_cost: float
    catering_per_head: float
    fixed_costs: float
    surprise_cap: float
    description: str = ""


FORMAT_PRESETS = {
    "In-Person": FormatConfig(
        name="In-Person",
        label="in_person",
        ticket_price=35,
        expected_attendees=100,
        sponsor_max=2000,
        sponsor_prob=0.70,
        venue_cost=900,
        catering_per_head=22,
        fixed_costs=450,
        surprise_cap=250,
        description="Full venue, catering, A/V — highest revenue ceiling, highest cost floor"
    ),
    "Hybrid": FormatConfig(
        name="Hybrid",
        label="hybrid",
        ticket_price=25,
        expected_attendees=130,
        sponsor_max=1800,
        sponsor_prob=0.65,
        venue_cost=600,
        catering_per_head=15,
        fixed_costs=550,
        surprise_cap=200,
        description="Smaller venue + streaming — broader reach, higher tech costs"
    ),
    "Virtual": FormatConfig(
        name="Virtual",
        label="virtual",
        ticket_price=10,
        expected_attendees=200,
        sponsor_max=800,
        sponsor_prob=0.50,
        venue_cost=0,
        catering_per_head=0,
        fixed_costs=300,
        surprise_cap=80,
        description="No venue cost, high attendance ceiling, lower ticket price and sponsor appeal"
    ),
    "Small In-Person": FormatConfig(
        name="Small In-Person",
        label="small_inperson",
        ticket_price=20,
        expected_attendees=50,
        sponsor_max=800,
        sponsor_prob=0.60,
        venue_cost=350,
        catering_per_head=18,
        fixed_costs=200,
        surprise_cap=100,
        description="Intimate venue — lower risk, lower upside, good for workshops"
    ),
}


# ─────────────────────────────────────────────
# Risk-adjusted metrics
# ─────────────────────────────────────────────

def sharpe_ratio(pnl: np.ndarray, risk_free: float = 0.0) -> float:
    """
    Event Sharpe ratio: expected excess return per unit of volatility.
    Analogous to portfolio Sharpe — higher is better.
    """
    excess = pnl.mean() - risk_free
    return float(excess / pnl.std()) if pnl.std() > 0 else 0.0


def sortino_ratio(pnl: np.ndarray, risk_free: float = 0.0) -> float:
    """
    Sortino ratio: like Sharpe but only penalizes downside volatility.
    Better metric when upside variance is welcome.
    """
    downside = pnl[pnl < risk_free]
    downside_std = downside.std() if len(downside) > 1 else 1e-9
    return float((pnl.mean() - risk_free) / downside_std)


def max_drawdown_equiv(pnl: np.ndarray) -> float:
    """Worst-case single scenario loss (analog to max drawdown)."""
    return float(pnl.min())


def prob_outperform(pnl_a: np.ndarray, pnl_b: np.ndarray) -> float:
    """P(Format A P&L > Format B P&L) — pairwise scenario comparison."""
    return float((pnl_a > pnl_b).mean())


def calmar_ratio(pnl: np.ndarray) -> float:
    """Expected return / max loss — analog to Calmar ratio in hedge funds."""
    worst = abs(pnl.min())
    return float(pnl.mean() / worst) if worst > 0 else float("inf")


# ─────────────────────────────────────────────
# Comparison engine
# ─────────────────────────────────────────────

def run_comparison(
    config_a: FormatConfig,
    config_b: FormatConfig,
    n: int = 10_000,
    confidence: float = 0.95,
) -> dict:
    """
    Run side-by-side Monte Carlo comparison of two event formats.
    Uses the same random seed basis so results are directly comparable.
    """
    def cfg_to_params(cfg: FormatConfig) -> dict:
        return dict(
            ticket_price       = cfg.ticket_price,
            expected_attendees = cfg.expected_attendees,
            sponsor_max        = cfg.sponsor_max,
            sponsor_prob       = cfg.sponsor_prob,
            venue_cost         = cfg.venue_cost,
            catering_per_head  = cfg.catering_per_head,
            fixed_costs        = cfg.fixed_costs,
            surprise_cap       = cfg.surprise_cap,
        )

    res_a = run_simulation(n=n, **cfg_to_params(config_a))
    res_b = run_simulation(n=n, **cfg_to_params(config_b))

    pnl_a = res_a["pnl"]
    pnl_b = res_b["pnl"]

    def build_profile(cfg, res, pnl) -> dict:
        return {
            "name":           cfg.name,
            "description":    cfg.description,
            "pnl":            pnl,
            "expected_pnl":   float(pnl.mean()),
            "std":            float(pnl.std()),
            "median_pnl":     float(np.median(pnl)),
            "var":            compute_var(pnl, confidence),
            "cvar":           compute_cvar(pnl, confidence),
            "breakeven_prob": float((pnl >= 0).mean()),
            "upside_prob":    float((pnl > 500).mean()),
            "sharpe":         sharpe_ratio(pnl),
            "sortino":        sortino_ratio(pnl),
            "calmar":         calmar_ratio(pnl),
            "max_loss":       max_drawdown_equiv(pnl),
            "p95_upside":     float(np.percentile(pnl, 95)),
            "p5_downside":    float(np.percentile(pnl, 5)),
            "attendance":     res["attendance"],
            "revenue":        res["total_revenue"],
        }

    profile_a = build_profile(config_a, res_a, pnl_a)
    profile_b = build_profile(config_b, res_b, pnl_b)

    p_a_beats_b = prob_outperform(pnl_a, pnl_b)

    # Scoring: weighted across dimensions (org-relevant weights)
    def score(p: dict) -> float:
        return (
            0.30 * p["breakeven_prob"] +
            0.25 * min(1.0, max(0.0, (p["sharpe"] + 1) / 2)) +
            0.20 * min(1.0, max(0.0, p["upside_prob"])) +
            0.15 * min(1.0, max(0.0, (p["sortino"] + 1) / 2)) +
            0.10 * min(1.0, max(0.0, (p["calmar"] + 1) / 4))
        )

    score_a = score(profile_a)
    score_b = score(profile_b)

    # Recommendation logic
    if abs(score_a - score_b) < 0.03:
        recommendation = "toss-up"
        rec_reason = (
            f"Both formats are nearly equivalent on a risk-adjusted basis "
            f"(scores: {config_a.name} {score_a:.2f} vs {config_b.name} {score_b:.2f}). "
            f"Choose based on member experience goals rather than financials."
        )
    elif score_a > score_b:
        recommendation = config_a.name
        rec_reason = (
            f"{config_a.name} scores higher on risk-adjusted return ({score_a:.2f} vs {score_b:.2f}). "
            f"Break-even probability is {profile_a['breakeven_prob']*100:.1f}% vs "
            f"{profile_b['breakeven_prob']*100:.1f}%, and Sharpe ratio is "
            f"{profile_a['sharpe']:.2f} vs {profile_b['sharpe']:.2f}."
        )
    else:
        recommendation = config_b.name
        rec_reason = (
            f"{config_b.name} scores higher on risk-adjusted return ({score_b:.2f} vs {score_a:.2f}). "
            f"Break-even probability is {profile_b['breakeven_prob']*100:.1f}% vs "
            f"{profile_a['breakeven_prob']*100:.1f}%, and Sharpe ratio is "
            f"{profile_b['sharpe']:.2f} vs {profile_a['sharpe']:.2f}."
        )

    return {
        "a":                 profile_a,
        "b":                 profile_b,
        "p_a_beats_b":       p_a_beats_b,
        "p_b_beats_a":       1 - p_a_beats_b,
        "score_a":           score_a,
        "score_b":           score_b,
        "recommendation":    recommendation,
        "rec_reason":        rec_reason,
        "diff_pnl":          pnl_a - pnl_b,     # scenario-by-scenario difference
        "n_sims":            n,
        "confidence":        confidence,
    }


# ─────────────────────────────────────────────
# CLI report
# ─────────────────────────────────────────────

def print_comparison_report(results: dict):
    a, b = results["a"], results["b"]
    c = int(results["confidence"] * 100)

    def w(val_a, val_b, higher_is_better=True):
        """Return winner marker."""
        if higher_is_better:
            return ("◀", "  ") if val_a > val_b else ("  ", "▶")
        else:
            return ("◀", "  ") if val_a < val_b else ("  ", "▶")

    print("=" * 68)
    print(f"  Event Format Comparison: {a['name']} vs {b['name']}")
    print("=" * 68)
    print(f"  {'Metric':<28} {a['name']:>14}   {b['name']:>14}")
    print("-" * 68)

    metrics = [
        ("Expected P&L",        a["expected_pnl"],   b["expected_pnl"],   True,  "${:>+,.0f}"),
        ("Median P&L",          a["median_pnl"],     b["median_pnl"],     True,  "${:>+,.0f}"),
        ("Std deviation",       a["std"],            b["std"],            False, "${:>,.0f}"),
        (f"VaR ({c}%)",         a["var"],            b["var"],            True,  "${:>+,.0f}"),
        (f"CVaR ({c}%)",        a["cvar"],           b["cvar"],           True,  "${:>+,.0f}"),
        ("Break-even prob",     a["breakeven_prob"], b["breakeven_prob"], True,  "{:>.1%}"),
        ("P(profit > $500)",    a["upside_prob"],    b["upside_prob"],    True,  "{:>.1%}"),
        ("Sharpe ratio",        a["sharpe"],         b["sharpe"],         True,  "{:>.3f}"),
        ("Sortino ratio",       a["sortino"],        b["sortino"],        True,  "{:>.3f}"),
        ("Calmar ratio",        a["calmar"],         b["calmar"],         True,  "{:>.3f}"),
        ("95th pct upside",     a["p95_upside"],     b["p95_upside"],     True,  "${:>+,.0f}"),
        ("5th pct downside",    a["p5_downside"],    b["p5_downside"],    True,  "${:>+,.0f}"),
    ]

    for label, va, vb, hib, fmt in metrics:
        wa, wb = w(va, vb, hib)
        print(f"  {label:<28} {wa} {fmt.format(va):>13}   {wb} {fmt.format(vb):>13}")

    print("=" * 68)
    print(f"\n  P({a['name']} beats {b['name']}): {results['p_a_beats_b']*100:.1f}%")
    print(f"  P({b['name']} beats {a['name']}): {results['p_b_beats_a']*100:.1f}%")
    print(f"\n  Risk-adjusted score:   {a['name']} {results['score_a']:.3f}   {b['name']} {results['score_b']:.3f}")
    print(f"\n  ✅ Recommendation: {results['recommendation']}")
    print(f"  {results['rec_reason']}")
    print("=" * 68)


if __name__ == "__main__":
    cfg_a = FORMAT_PRESETS["In-Person"]
    cfg_b = FORMAT_PRESETS["Hybrid"]

    print(f"Comparing {cfg_a.name} vs {cfg_b.name}...")
    results = run_comparison(cfg_a, cfg_b, n=10_000)
    print_comparison_report(results)
