import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from simulation import run_simulation, compute_var, compute_cvar, sensitivity_analysis
from portfolio import run_portfolio_simulation, SEMESTER_TEMPLATES, EventConfig
from comparison import run_comparison, FORMAT_PRESETS, FormatConfig

st.set_page_config(page_title="ALPFA Risk Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
.block-container { padding-top: 1.5rem; }
.stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# NAV
# ═══════════════════════════════════════════════════════════

st.title("📊 ALPFA Event Risk Management System")
st.caption("Monte Carlo simulation · Portfolio risk · Event comparison · Stochastic P&L modeling")

tab1, tab2, tab3 = st.tabs([
    "🎯  Single Event Risk",
    "📅  Semester Portfolio",
    "⚖️  Event Comparison",
])

COLORS = {"profit": "#5DCAA5", "loss": "#F09595", "tail": "#A32D2D",
          "blue": "#185FA5", "amber": "#FAC775", "neutral": "#888"}

def fmt(v): return f"${v:+,.0f}"
def pct(v): return f"{v*100:.1f}%"


# ═══════════════════════════════════════════════════════════
# TAB 1 — Single Event
# ═══════════════════════════════════════════════════════════

with tab1:
    with st.sidebar:
        st.markdown("## 🎯 Single Event Parameters")

        st.markdown("#### Revenue")
        ticket_price       = st.slider("Ticket price ($)",           5,   100,  25)
        expected_attendees = st.slider("Expected attendees",          10,  400,  90)
        sponsor_max        = st.slider("Max sponsorship ($)",         0,   10000,1500,step=100)
        sponsor_prob       = st.slider("Sponsor commit prob.",        0.0, 1.0,  0.65,step=0.01,format="%.2f")

        st.markdown("#### Costs")
        venue_cost        = st.slider("Venue cost ($)",              100, 5000, 600, step=50)
        catering_per_head = st.slider("Catering per head ($)",       5,   60,   18)
        fixed_costs       = st.slider("AV & marketing ($)",          0,   3000, 400, step=50)
        surprise_cap      = st.slider("Surprise cost budget ($)",    0,   2000, 200, step=50)

        st.markdown("#### Simulation")
        n_sims     = st.select_slider("Simulations", [1000,5000,10000,25000], value=10000)
        confidence = st.slider("Confidence level", 0.90, 0.99, 0.95, step=0.01, format="%.2f")
        rho        = st.slider("Attend↔Revenue ρ", 0.0, 1.0, 0.82, step=0.01)
        run_btn    = st.button("▶ Run Simulation", use_container_width=True, type="primary")

    params = dict(ticket_price=ticket_price, expected_attendees=expected_attendees,
                  sponsor_max=sponsor_max, sponsor_prob=sponsor_prob,
                  venue_cost=venue_cost, catering_per_head=catering_per_head,
                  fixed_costs=fixed_costs, surprise_cap=surprise_cap, rho_attend_rev=rho)

    if "res1" not in st.session_state or run_btn:
        with st.spinner("Simulating..."):
            st.session_state.res1  = run_simulation(n=n_sims, **params)
            st.session_state.sens1 = sensitivity_analysis(params, n=3000)

    res   = st.session_state.res1
    sens  = st.session_state.sens1
    pnl   = res["pnl"]
    var_v = compute_var(pnl, confidence)
    cvar_v= compute_cvar(pnl, confidence)
    c_pct = int(confidence*100)

    # Metrics
    cols = st.columns(6)
    cols[0].metric("Expected P&L",      fmt(pnl.mean()))
    cols[1].metric("Break-even prob",   pct((pnl>=0).mean()))
    cols[2].metric(f"VaR {c_pct}%",    fmt(var_v),  delta="Max loss", delta_color="inverse")
    cols[3].metric(f"CVaR {c_pct}%",   fmt(cvar_v), delta="Tail avg", delta_color="inverse")
    cols[4].metric("P(profit>$500)",    pct((pnl>500).mean()))
    cols[5].metric("Std deviation",     f"${pnl.std():,.0f}")

    st.markdown("---")
    cl, cr = st.columns(2)

    # Histogram
    with cl:
        st.subheader("P&L Distribution")
        counts, edges = np.histogram(pnl, bins=50)
        centers = (edges[:-1]+edges[1:])/2
        colors  = [COLORS["tail"] if c<var_v else COLORS["loss"] if c<0 else COLORS["profit"] for c in centers]
        fig = go.Figure(go.Bar(x=centers, y=counts, marker_color=colors,
                               hovertemplate="P&L≈$%{x:,.0f}<br>%{y} scenarios<extra></extra>"))
        fig.add_vline(x=0,      line_dash="dash", line_color="#888",          annotation_text="Break-even")
        fig.add_vline(x=var_v,  line_dash="dot",  line_color=COLORS["tail"],  annotation_text=f"VaR {c_pct}%")
        fig.add_vline(x=cvar_v, line_dash="dot",  line_color="#630000",       annotation_text=f"CVaR {c_pct}%")
        fig.update_layout(showlegend=False, height=300, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=10,r=10,t=10,b=40),
                          xaxis=dict(title="P&L ($)", tickformat="$,.0f"),
                          yaxis=dict(title="Scenarios"))
        fig.update_xaxes(
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )

        fig.update_yaxes(
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )
        fig.update_layout(
            font=dict(
                color="black",
                size=14
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"🟢 Profitable · 🟡 Loss · 🔴 Tail risk · Recommended reserve: **{fmt(abs(cvar_v))}**")

    # Tornado
    with cr:
        st.subheader("Sensitivity — Variance Drivers")
        labels  = list(sens.keys())
        impacts = list(sens.values())
        bar_clr = [COLORS["profit"] if v>0 else COLORS["loss"] for v in impacts]
        fig = go.Figure(go.Bar(x=impacts, y=labels, orientation="h", marker_color=bar_clr,
                               text=[fmt(v) for v in impacts], textposition="outside",
                               hovertemplate="%{y}: $%{x:+,.0f}<extra></extra>"))
        fig.add_vline(x=0, line_color="#888", line_width=1)
        fig.update_layout(showlegend=False, height=300, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=10,r=60,t=10,b=40),
                          xaxis=dict(title="P&L impact ($)", tickformat="$,.0f"),
                          yaxis=dict(autorange="reversed"))
        fig.update_xaxes(
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )

        fig.update_yaxes(
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )
        fig.update_layout(
            font=dict(
                color="black",
                size=14
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Which parameter moves your P&L the most under a shock scenario")

    cl2, cr2 = st.columns(2)

    # Scatter
    with cl2:
        st.subheader("Correlation Structure")
        idx = np.random.choice(len(pnl), size=min(400,len(pnl)), replace=False)
        fig = px.scatter(x=res["attendance"][idx], y=res["total_revenue"][idx],
                         color=pnl[idx], color_continuous_scale=["#F09595","#FAC775","#5DCAA5"],
                         opacity=0.45, labels={"x":"Attendees","y":"Revenue ($)","color":"P&L ($)"})
        fig.update_traces(marker_size=4)
        fig.update_layout(height=260, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=10,r=10,t=10,b=40))
        fig.update_xaxes(
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )

        fig.update_yaxes(
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"ρ(attendance, revenue) = {rho:.2f} via Cholesky decomposition")

    # CDF
    with cr2:
        st.subheader("Cumulative P&L Distribution")
        sp = np.sort(pnl)
        cdf = np.arange(1,len(sp)+1)/len(sp)*100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sp, y=cdf, mode="lines",
                                 line=dict(color=COLORS["blue"],width=2),
                                 hovertemplate="P&L: $%{x:,.0f}<br>Percentile: %{y:.1f}%<extra></extra>"))
        fig.add_hline(y=(1-confidence)*100, line_dash="dot", line_color=COLORS["tail"],
                      annotation_text=f"VaR {c_pct}%")
        fig.add_hline(y=50, line_dash="dash", line_color="#888", annotation_text="Median")
        fig.update_layout(height=260, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=10,r=10,t=10,b=40),
                          xaxis=dict(title="P&L ($)", tickformat="$,.0f"),
                          yaxis=dict(title="Cumulative %", ticksuffix="%"))
        fig.update_xaxes(
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )

        fig.update_yaxes(
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Read any threshold to get its probability of occurring")


# ═══════════════════════════════════════════════════════════
# TAB 2 — Semester Portfolio
# ═══════════════════════════════════════════════════════════

with tab2:
    st.subheader("Semester Portfolio Risk")
    st.caption("Simulates all events simultaneously with correlated P&L — models how a bad event affects later ones.")

    pcol1, pcol2, pcol3 = st.columns([1.5,1,1])
    with pcol1:
        semester      = st.selectbox("Semester template", ["Fall","Spring"])
        port_n        = st.select_slider("Simulations ", [1000,5000,10000], value=5000, key="port_n")
        port_conf     = st.slider("Confidence ", 0.90, 0.99, 0.95, step=0.01, format="%.2f", key="port_conf")
        momentum_on   = st.toggle("Enable attendance momentum", value=True)
        port_btn      = st.button("▶ Run Portfolio Simulation", type="primary")

    if "port_res" not in st.session_state or port_btn:
        events = SEMESTER_TEMPLATES[semester]
        with st.spinner("Simulating semester portfolio..."):
            st.session_state.port_res = run_portfolio_simulation(
                events, n=port_n, confidence=port_conf, momentum=momentum_on)

    pr = st.session_state.port_res
    c_p = int(port_conf*100) if "port_conf" in dir() else 95

    st.markdown("---")

    # Portfolio headline metrics
    pm = st.columns(6)
    pm[0].metric("Semester Expected P&L",  fmt(pr["portfolio_expected"]))
    pm[1].metric("Break-even prob",         pct(pr["portfolio_breakeven"]))
    pm[2].metric(f"Portfolio VaR {c_p}%",  fmt(pr["portfolio_var"]), delta="Max loss", delta_color="inverse")
    pm[3].metric(f"Portfolio CVaR {c_p}%", fmt(pr["portfolio_cvar"]), delta="Tail avg", delta_color="inverse")
    pm[4].metric("Diversification benefit", fmt(pr["diversification_benefit"]),
                  delta="vs sum of individual VaRs", delta_color="normal")
    pm[5].metric("Recommended reserve",     f"${abs(pr['portfolio_cvar']):,.0f}")

    st.markdown("---")
    lc, rc = st.columns(2)

    # Per-event bar chart
    with lc:
        st.subheader("Per-Event Expected P&L")
        ev_names  = [e["name"] for e in pr["events"]]
        ev_means  = [e["expected_pnl"] for e in pr["events"]]
        ev_stds   = [e["std"] for e in pr["events"]]
        ev_colors = [COLORS["profit"] if v>=0 else COLORS["loss"] for v in ev_means]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ev_names, y=ev_means, marker_color=ev_colors,
            error_y=dict(type="data", array=ev_stds, visible=True, color="#aaa"),
            hovertemplate="%{x}<br>Expected: $%{y:+,.0f}<extra></extra>"
        ))
        fig.add_hline(y=0, line_color="#888", line_dash="dash")
        fig.update_layout(height=300, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=10,r=10,t=10,b=80),
                          yaxis=dict(title="Expected P&L ($)", tickformat="$,.0f"),
                          xaxis=dict(tickangle=-20))
        fig.update_xaxes(
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )

        fig.update_yaxes(
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )
        fig.update_layout(
            font=dict(
                color="black",
                size=14
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Error bars = ±1 standard deviation")

    # Portfolio P&L distribution
    with rc:
        st.subheader("Semester P&L Distribution")
        port_pnl = pr["portfolio_pnl"]
        port_var = pr["portfolio_var"]
        counts, edges = np.histogram(port_pnl, bins=50)
        centers = (edges[:-1]+edges[1:])/2
        colors  = [COLORS["tail"] if c<port_var else COLORS["loss"] if c<0 else COLORS["profit"] for c in centers]

        fig = go.Figure(go.Bar(x=centers, y=counts, marker_color=colors,
                               hovertemplate="Semester P&L≈$%{x:,.0f}<br>%{y} scenarios<extra></extra>"))
        fig.add_vline(x=0,        line_dash="dash", line_color="#888",         annotation_text="Break-even")
        fig.add_vline(x=port_var, line_dash="dot",  line_color=COLORS["tail"], annotation_text=f"VaR")
        fig.update_layout(height=300, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=10,r=10,t=10,b=40),
                          xaxis=dict(title="Semester P&L ($)", tickformat="$,.0f"),
                          yaxis=dict(title="Scenarios"))
        fig.update_xaxes(
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )

        fig.update_yaxes(
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )
        fig.update_layout(
            font=dict(
                color="black",
                size=14
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Portfolio VaR is lower than sum of individual VaRs — diversification benefit: **{fmt(pr['diversification_benefit'])}**")

    # Event risk table
    st.subheader("Event-by-Event Risk Breakdown")
    rows = []
    for e in pr["events"]:
        rows.append({
            "Event":             e["name"],
            "Type":              e["type"].title(),
            "Month":             e["month"],
            "Expected P&L":      fmt(e["expected_pnl"]),
            "Std Dev":           f"${e['std']:,.0f}",
            f"VaR ({c_p}%)":     fmt(e["var"]),
            f"CVaR ({c_p}%)":    fmt(e["cvar"]),
            "Break-even":        pct(e["breakeven_prob"]),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Correlation heatmap
    st.subheader("Cross-Event P&L Correlation Matrix")
    corr = pr["corr_matrix"]
    ev_labels = [e["name"] for e in pr["events"]]
    fig = go.Figure(go.Heatmap(
        z=corr, x=ev_labels, y=ev_labels,
        colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr,2), texttemplate="%{text}",
        hovertemplate="ρ(%{x}, %{y}) = %{z:.2f}<extra></extra>"
    ))
    fig.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Higher correlation = less diversification benefit. Events of the same type tend to move together.")


# ═══════════════════════════════════════════════════════════
# TAB 3 — Event Comparison
# ═══════════════════════════════════════════════════════════

with tab3:
    st.subheader("Event Format Comparison")
    st.caption("Risk-adjusted side-by-side comparison using Sharpe, Sortino, Calmar ratios and P&L probability.")

    comp_col1, comp_col2, comp_col3 = st.columns([1,1,1])
    preset_names = list(FORMAT_PRESETS.keys())

    with comp_col1:
        st.markdown("**Format A**")
        preset_a = st.selectbox("Format A preset", preset_names, index=0, key="pa")
        cfg_a    = FORMAT_PRESETS[preset_a]
        st.caption(cfg_a.description)
        tp_a  = st.slider("Ticket price A",       5,  100, int(cfg_a.ticket_price),  key="tpa")
        att_a = st.slider("Attendees A",          10, 400, cfg_a.expected_attendees, key="ata")
        sp_a  = st.slider("Sponsor max A ($)",    0,  8000,int(cfg_a.sponsor_max),   key="sma", step=100)
        spp_a = st.slider("Sponsor prob A",       0.0,1.0, cfg_a.sponsor_prob,       key="spa", step=0.01, format="%.2f")
        vc_a  = st.slider("Venue cost A ($)",     0,  4000,int(cfg_a.venue_cost),    key="vca", step=50)
        cat_a = st.slider("Catering/head A ($)",  0,  50,  int(cfg_a.catering_per_head), key="cata")
        fc_a  = st.slider("Fixed costs A ($)",    0,  2000,int(cfg_a.fixed_costs),   key="fca", step=50)

    with comp_col2:
        st.markdown("**Format B**")
        preset_b = st.selectbox("Format B preset", preset_names, index=1, key="pb")
        cfg_b    = FORMAT_PRESETS[preset_b]
        st.caption(cfg_b.description)
        tp_b  = st.slider("Ticket price B",       5,  100, int(cfg_b.ticket_price),  key="tpb")
        att_b = st.slider("Attendees B",          10, 400, cfg_b.expected_attendees, key="atb")
        sp_b  = st.slider("Sponsor max B ($)",    0,  8000,int(cfg_b.sponsor_max),   key="smb", step=100)
        spp_b = st.slider("Sponsor prob B",       0.0,1.0, cfg_b.sponsor_prob,       key="spb", step=0.01, format="%.2f")
        vc_b  = st.slider("Venue cost B ($)",     0,  4000,int(cfg_b.venue_cost),    key="vcb", step=50)
        cat_b = st.slider("Catering/head B ($)",  0,  50,  int(cfg_b.catering_per_head), key="catb")
        fc_b  = st.slider("Fixed costs B ($)",    0,  2000,int(cfg_b.fixed_costs),   key="fcb", step=50)

    with comp_col3:
        st.markdown("**Settings**")
        comp_n    = st.select_slider("Simulations  ", [1000,5000,10000], value=5000, key="comp_n")
        comp_conf = st.slider("Confidence  ", 0.90, 0.99, 0.95, step=0.01, format="%.2f", key="comp_conf")
        comp_btn  = st.button("▶ Run Comparison", type="primary", use_container_width=True)
        st.markdown("---")
        st.markdown("**Scoring weights**")
        st.caption("Break-even: 30% · Sharpe: 25% · Upside: 20% · Sortino: 15% · Calmar: 10%")

    custom_a = FormatConfig(name=preset_a, label="a", ticket_price=tp_a,
                            expected_attendees=att_a, sponsor_max=sp_a, sponsor_prob=spp_a,
                            venue_cost=vc_a, catering_per_head=cat_a, fixed_costs=fc_a,
                            surprise_cap=150, description=cfg_a.description)
    custom_b = FormatConfig(name=preset_b, label="b", ticket_price=tp_b,
                            expected_attendees=att_b, sponsor_max=sp_b, sponsor_prob=spp_b,
                            venue_cost=vc_b, catering_per_head=cat_b, fixed_costs=fc_b,
                            surprise_cap=150, description=cfg_b.description)

    if "comp_res" not in st.session_state or comp_btn:
        with st.spinner("Running comparison..."):
            st.session_state.comp_res = run_comparison(custom_a, custom_b, n=comp_n, confidence=comp_conf)

    cr = st.session_state.comp_res
    a, b = cr["a"], cr["b"]
    cc = int(comp_conf*100) if "comp_conf" in dir() else 95

    st.markdown("---")

    # Recommendation banner
    rec_color = "#E6F7F1" if cr["recommendation"] == a["name"] else "#EEF4FB" if cr["recommendation"] == b["name"] else "#FAFAF0"
    st.markdown(f"""
    <div style="background:{rec_color};border-radius:10px;padding:1rem 1.25rem;margin-bottom:1rem;">
        <strong>✅ Recommendation: {cr['recommendation']}</strong><br>
        <span style="font-size:14px;color:#444;">{cr['rec_reason']}</span>
    </div>
    """, unsafe_allow_html=True)

    # Head-to-head metrics
    mc = st.columns(4)
    mc[0].metric(f"P({a['name']} wins)", pct(cr["p_a_beats_b"]))
    mc[1].metric(f"P({b['name']} wins)", pct(cr["p_b_beats_a"]))
    mc[2].metric(f"Score: {a['name']}", f"{cr['score_a']:.3f}")
    mc[3].metric(f"Score: {b['name']}", f"{cr['score_b']:.3f}")

    st.markdown("---")
    lcc, rcc = st.columns(2)

    # Overlapping P&L distributions
    with lcc:
        st.subheader("P&L Distributions — Overlaid")
        counts_a, edges_a = np.histogram(a["pnl"], bins=50, density=True)
        counts_b, edges_b = np.histogram(b["pnl"], bins=50, density=True)
        centers_a = (edges_a[:-1]+edges_a[1:])/2
        centers_b = (edges_b[:-1]+edges_b[1:])/2

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=centers_a, y=counts_a, mode="lines", fill="tozeroy",
                                 name=a["name"], line=dict(color=COLORS["blue"]),
                                 fillcolor="rgba(24,95,165,0.25)",
                                 hovertemplate=f"{a['name']}: $%{{x:,.0f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=centers_b, y=counts_b, mode="lines", fill="tozeroy",
                                 name=b["name"], line=dict(color=COLORS["profit"]),
                                 fillcolor="rgba(93,202,165,0.25)",
                                 hovertemplate=f"{b['name']}: $%{{x:,.0f}}<extra></extra>"))
        fig.add_vline(x=0, line_dash="dash", line_color="#888")
        fig.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=10,r=10,t=10,b=40),
                          legend=dict(orientation="h", y=1.0, x=0),
                          xaxis=dict(title="P&L ($)", tickformat="$,.0f"),
                          yaxis=dict(title="Density"))
        fig.update_xaxes(
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )

        fig.update_yaxes(
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        )
        fig.update_layout(
            font=dict(
                color="black",
                size=14
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # Risk-adjusted scorecard
    with rcc:
        st.subheader("Risk-Adjusted Scorecard")
        metrics_comp = {
            "Expected P&L":        (a["expected_pnl"],   b["expected_pnl"],   True,  fmt),
            "Break-even prob":     (a["breakeven_prob"], b["breakeven_prob"], True,  pct),
            f"VaR ({cc}%)":        (a["var"],            b["var"],            True,  fmt),
            f"CVaR ({cc}%)":       (a["cvar"],           b["cvar"],           True,  fmt),
            "Sharpe ratio":        (a["sharpe"],         b["sharpe"],         True,  lambda v: f"{v:.3f}"),
            "Sortino ratio":       (a["sortino"],        b["sortino"],        True,  lambda v: f"{v:.3f}"),
            "Calmar ratio":        (a["calmar"],         b["calmar"],         True,  lambda v: f"{v:.3f}"),
            "95th pct upside":     (a["p95_upside"],     b["p95_upside"],     True,  fmt),
        }

        rows = []
        for label, (va, vb, hib, fmtfn) in metrics_comp.items():
            winner_a = (va > vb) == hib
            rows.append({
                "Metric":    label,
                a["name"]:   ("🏆 " if winner_a  else "   ") + fmtfn(va),
                b["name"]:   ("🏆 " if not winner_a else "   ") + fmtfn(vb),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True, height=300)

    # P&L difference distribution
    st.subheader(f"P&L Difference Distribution: {a['name']} minus {b['name']}")
    diff = cr["diff_pnl"]
    counts_d, edges_d = np.histogram(diff, bins=50)
    centers_d = (edges_d[:-1]+edges_d[1:])/2
    clr_d = [COLORS["profit"] if c>=0 else COLORS["loss"] for c in centers_d]

    fig = go.Figure(go.Bar(x=centers_d, y=counts_d, marker_color=clr_d,
                           hovertemplate="Diff≈$%{x:,.0f}<br>%{y} scenarios<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color="#888",
                  annotation_text=f"← {b['name']} better  |  {a['name']} better →")
    fig.update_layout(height=220, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10,r=10,t=10,b=40),
                      xaxis=dict(title="P&L difference ($)", tickformat="$,.0f"),
                      yaxis=dict(title="Scenarios"))
    fig.update_xaxes(
        tickfont=dict(color="black"),
        title_font=dict(color="black")
    )

    fig.update_yaxes(
        tickfont=dict(color="black"),
        title_font=dict(color="black")
    )
    fig.update_layout(
        font=dict(
            color="black",
            size=14
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Green = {a['name']} outperforms · Red = {b['name']} outperforms · "
               f"Probability {a['name']} wins: **{pct(cr['p_a_beats_b'])}**")

    st.markdown("---")
    st.caption("Methodology: Monte Carlo · Poisson attendance · Bernoulli×LogNormal sponsorship · "
               "Pareto surprise costs · Sharpe/Sortino/Calmar risk-adjusted scoring")
