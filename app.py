"""
AI Economy Simulator - Interactive Dashboard

Models the economic impact of AI-driven automation across sectors,
inspired by Citrini Research's "2028 Global Intelligence Crisis" report.

Run with: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from simulation.config import (
    DEFAULT_SECTORS,
    SimulationParams,
    SCENARIO_PRESETS,
)
from simulation.engine import EconomySimulator

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Economy Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = px.colors.qualitative.Set2
SECTOR_COLORS = {s.name: COLORS[i % len(COLORS)] for i, s in enumerate(DEFAULT_SECTORS)}

# Common layout for all charts
CHART_THEME = dict(
    template="simple_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#262730"),
)


# ── Helper: build line chart ─────────────────────────────────────────
def line_chart(
    x, y, title, yaxis, color="#1f77b4", fmt=None, milestones=None, labels=None
):
    fig = go.Figure()
    hover = "%{y:.1f}" if fmt is None else fmt
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", line=dict(color=color, width=2.5),
            hovertemplate=hover + "<extra></extra>",
        )
    )
    if milestones and labels:
        for qi, label in milestones:
            if 0 <= qi < len(y):
                fig.add_annotation(
                    x=labels[qi], y=y[qi], text=label,
                    showarrow=True, arrowhead=2, arrowsize=1,
                    ax=0, ay=-40, font=dict(size=9),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#666", borderwidth=1,
                )
    fig.update_layout(
        **CHART_THEME,
        title=dict(text=title, font=dict(size=14)),
        yaxis_title=yaxis, height=320,
        margin=dict(l=50, r=20, t=35, b=30),
        hovermode="x unified",
    )
    return fig


def area_chart_sectors(x, employment, sector_names):
    fig = go.Figure()
    for i, name in enumerate(sector_names):
        fig.add_trace(
            go.Scatter(
                x=x, y=employment[:, i], name=name, stackgroup="one",
                line=dict(width=0.5),
                fillcolor=SECTOR_COLORS.get(name, COLORS[i % len(COLORS)]),
            )
        )
    fig.update_layout(
        **CHART_THEME,
        title=dict(text="Employment by Sector (Millions)", font=dict(size=14)),
        yaxis_title="Millions", height=380,
        margin=dict(l=50, r=20, t=35, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, font=dict(size=10)),
    )
    return fig


def multi_line(x, series_dict, title, yaxis, height=340):
    fig = go.Figure()
    for i, (name, vals) in enumerate(series_dict.items()):
        fig.add_trace(
            go.Scatter(
                x=x, y=vals, name=name, mode="lines",
                line=dict(color=SECTOR_COLORS.get(name, COLORS[i % len(COLORS)]), width=2),
            )
        )
    fig.update_layout(
        **CHART_THEME,
        title=dict(text=title, font=dict(size=14)),
        yaxis_title=yaxis, height=height,
        margin=dict(l=50, r=20, t=35, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, font=dict(size=10)),
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("Simulation Controls")

# Scenario presets
preset_name = st.sidebar.selectbox(
    "Scenario Preset",
    ["Custom"] + list(SCENARIO_PRESETS.keys()),
    index=1,  # default to Citrini Baseline
)

if preset_name != "Custom":
    preset = SCENARIO_PRESETS[preset_name]
else:
    preset = SimulationParams()

# AI Dynamics
with st.sidebar.expander("AI Dynamics", expanded=True):
    ai_growth = st.slider(
        "AI Capability Growth (%/quarter)",
        0, 30, int(preset.ai_capability_quarterly_growth * 100),
        help="How fast AI capabilities improve each quarter",
    )
    ai_cost = st.slider(
        "AI Cost Decline (%/quarter)",
        0, 30, int(preset.ai_cost_quarterly_decline * 100),
        help="How fast AI inference costs fall",
    )

# Sector automation
with st.sidebar.expander("Sector Automation Speed", expanded=False):
    st.caption("0 = no automation, 100 = maximum")
    sector_speeds = {}
    for sector in DEFAULT_SECTORS:
        default_speed = preset.sector_automation_speeds.get(
            sector.name,
            SimulationParams().sector_automation_speeds.get(sector.name, 30),
        )
        speed = st.slider(
            sector.name,
            0, 100, int(default_speed * 100),
            key=f"spd_{sector.name}",
        )
        sector_speeds[sector.name] = speed / 100.0

# Labor market
with st.sidebar.expander("Labor Market", expanded=False):
    redeployment = st.slider(
        "Worker Redeployment Rate (%/qtr)",
        0, 50, int(preset.worker_redeployment_rate * 100),
    )
    wage_penalty = st.slider(
        "Displaced Wage Penalty (%)",
        0, 80, int(preset.displaced_wage_penalty * 100),
    )
    layoff_speed = st.slider(
        "Layoff Speed (%/qtr)",
        5, 60, int(preset.layoff_speed * 100),
        help="How fast redundant workers are actually laid off",
    )

# Policy
with st.sidebar.expander("Policy Response", expanded=False):
    ubi = st.slider(
        "Monthly UBI ($/person)", 0, 2000,
        int(preset.ubi_monthly_per_person), step=100,
    )
    compute_tax = st.slider(
        "AI Compute Tax (%)", 0, 30,
        int(preset.compute_tax_rate * 100),
    )
    retraining = st.slider(
        "Retraining Investment ($B/yr)", 0, 200,
        int(preset.retraining_investment), step=10,
    )

# Advanced
with st.sidebar.expander("Advanced", expanded=False):
    num_quarters = st.slider("Simulation Length (quarters)", 8, 40, 24)
    confidence_sens = st.slider(
        "Consumer Confidence Sensitivity", 0.5, 3.0,
        preset.confidence_sensitivity, step=0.1,
    )
    feedback_strength = st.slider(
        "Feedback Loop Strength", 0.0, 1.0,
        preset.margin_pressure_ai_feedback, step=0.05,
        help="How much economic pain accelerates AI adoption",
    )
    agent_midpoint = st.slider(
        "Agent Commerce Adoption Midpoint (quarter)", 4, 20,
        int(preset.agent_adoption_midpoint_quarter),
        help="Quarter at which 50% of consumers use AI agents",
    )

# Build params
params = SimulationParams(
    num_quarters=num_quarters,
    ai_capability_quarterly_growth=ai_growth / 100,
    ai_cost_quarterly_decline=ai_cost / 100,
    sector_automation_speeds=sector_speeds,
    worker_redeployment_rate=redeployment / 100,
    displaced_wage_penalty=wage_penalty / 100,
    layoff_speed=layoff_speed / 100,
    ubi_monthly_per_person=ubi,
    compute_tax_rate=compute_tax / 100,
    retraining_investment=retraining,
    confidence_sensitivity=confidence_sens,
    margin_pressure_ai_feedback=feedback_strength,
    agent_adoption_midpoint_quarter=agent_midpoint,
)

# ── Run simulation ───────────────────────────────────────────────────
sim = EconomySimulator(params=params)
results = sim.run()
labels = results.labels
n = len(labels)

# ── Header ───────────────────────────────────────────────────────────
st.title("AI Economy Simulator")
st.markdown(
    "Interactive model of AI-driven economic transformation. "
    "Inspired by [Citrini Research's *2028 Global Intelligence Crisis*]"
    "(https://www.citriniresearch.com/p/2028gic)."
)

# Key metrics row
last = n - 1
gdp_chg = (results.gdp[last] - results.gdp[0]) / results.gdp[0] * 100
sp_chg = (results.sp500[last] - results.sp500[0]) / results.sp500[0] * 100
sp_peak = np.max(results.sp500)
sp_dd_max = (np.min(results.sp500[1:]) - sp_peak) / sp_peak * 100
emp_lost = results.total_employment[0] - results.total_employment[last]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("GDP Change", f"{gdp_chg:+.1f}%")
c2.metric(
    "Unemployment",
    f"{results.unemployment_rate[last] * 100:.1f}%",
    f"{(results.unemployment_rate[last] - results.unemployment_rate[0]) * 100:+.1f}pp",
    delta_color="inverse",
)
c3.metric("S&P 500", f"{results.sp500[last]:,.0f}", f"{sp_chg:+.1f}%")
c4.metric("Max S&P Drawdown", f"{sp_dd_max:.1f}%")
c5.metric("Jobs Lost (M)", f"{emp_lost:.1f}")
c6.metric(
    "Mortgage Delinq.",
    f"{results.mortgage_delinquency[last] * 100:.1f}%",
    f"{(results.mortgage_delinquency[last] - results.mortgage_delinquency[0]) * 100:+.1f}pp",
    delta_color="inverse",
)

# Milestones
if results.milestones:
    with st.expander(f"Simulation Milestones ({len(results.milestones)} events)", expanded=False):
        for qi, label in sorted(results.milestones):
            st.markdown(f"- **{labels[qi]}**: {label}")

# ── Tabs ─────────────────────────────────────────────────────────────
tab_overview, tab_sectors, tab_financial, tab_policy, tab_ai = st.tabs(
    ["Overview", "Sectors", "Financial", "Policy & Inequality", "AI Progress"]
)

# ── TAB: Overview ────────────────────────────────────────────────────
with tab_overview:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            line_chart(
                labels, results.gdp / 1000, "GDP ($T)", "Trillions",
                color="#2ca02c", milestones=results.milestones, labels=labels,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            line_chart(
                labels, results.sp500, "S&P 500", "Index",
                color="#1f77b4", milestones=results.milestones, labels=labels,
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            line_chart(
                labels, results.unemployment_rate * 100,
                "Unemployment Rate (%)", "%", color="#d62728",
                milestones=results.milestones, labels=labels,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            line_chart(
                labels, results.consumer_spending / 1000,
                "Consumer Spending ($T)", "Trillions", color="#ff7f0e",
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        line_chart(
            labels, results.gdp_growth_annualized * 100,
            "GDP Growth (Annualized %)", "%", color="#9467bd",
        ),
        use_container_width=True,
    )

# ── TAB: Sectors ─────────────────────────────────────────────────────
with tab_sectors:
    st.plotly_chart(
        area_chart_sectors(labels, results.employment, results.sector_names),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        # AI adoption by sector
        adopt_dict = {
            name: results.ai_adoption[:, i]
            for i, name in enumerate(results.sector_names)
        }
        st.plotly_chart(
            multi_line(
                labels, adopt_dict,
                "AI Adoption by Sector (fraction of tasks)", "Fraction",
            ),
            use_container_width=True,
        )

        # Total jobs displaced
        st.plotly_chart(
            line_chart(
                labels, results.total_jobs_displaced,
                "Cumulative Jobs Displaced (Millions)", "Millions",
                color="#d62728",
            ),
            use_container_width=True,
        )

    with col2:
        # Wages by sector
        wage_dict = {
            name: results.avg_wage[:, i]
            for i, name in enumerate(results.sector_names)
        }
        st.plotly_chart(
            multi_line(
                labels, wage_dict,
                "Average Wages by Sector ($K/year)", "$K",
            ),
            use_container_width=True,
        )

        # Employment change from baseline
        emp_chg = {}
        for i, name in enumerate(results.sector_names):
            pct = (results.employment[:, i] - results.employment[0, i]) / results.employment[0, i] * 100
            emp_chg[name] = pct
        st.plotly_chart(
            multi_line(
                labels, emp_chg,
                "Employment Change from Baseline (%)", "% Change",
            ),
            use_container_width=True,
        )

# ── TAB: Financial ──────────────────────────────────────────────────
with tab_financial:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            line_chart(
                labels, results.mortgage_delinquency * 100,
                "Mortgage Delinquency Rate (%)", "%", color="#d62728",
                milestones=results.milestones, labels=labels,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            line_chart(
                labels, results.sp500_drawdown * 100,
                "S&P 500 Drawdown from Peak (%)", "%", color="#e377c2",
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            line_chart(
                labels, results.home_price_index,
                "Home Price Index (100 = start)", "Index", color="#8c564b",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            line_chart(
                labels, results.total_labor_income / 1000,
                "Total Labor Income ($T)", "Trillions", color="#17becf",
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        line_chart(
            labels, results.savings_rate * 100,
            "Household Savings Rate (%)", "%", color="#bcbd22",
        ),
        use_container_width=True,
    )

# ── TAB: Policy & Inequality ────────────────────────────────────────
with tab_policy:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            line_chart(
                labels, results.federal_deficit_pct_gdp * 100,
                "Federal Deficit (% of GDP, negative=deficit)", "% of GDP",
                color="#d62728",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            line_chart(
                labels, results.gini,
                "Gini Coefficient", "Gini", color="#9467bd",
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            line_chart(
                labels, results.labor_share * 100,
                "Labor Share of GDP (%)", "%", color="#ff7f0e",
                milestones=results.milestones, labels=labels,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            line_chart(
                labels, results.agent_adoption * 100,
                "AI Agent Commerce Adoption (%)", "%", color="#2ca02c",
            ),
            use_container_width=True,
        )

    # Policy impact summary
    if params.ubi_monthly_per_person > 0 or params.compute_tax_rate > 0 or params.retraining_investment > 0:
        st.subheader("Policy Impact Summary")
        # Run baseline for comparison
        baseline_params = SimulationParams(
            num_quarters=params.num_quarters,
            ai_capability_quarterly_growth=params.ai_capability_quarterly_growth,
            ai_cost_quarterly_decline=params.ai_cost_quarterly_decline,
            sector_automation_speeds=params.sector_automation_speeds,
            worker_redeployment_rate=params.worker_redeployment_rate,
            displaced_wage_penalty=params.displaced_wage_penalty,
            layoff_speed=params.layoff_speed,
            confidence_sensitivity=params.confidence_sensitivity,
            margin_pressure_ai_feedback=params.margin_pressure_ai_feedback,
            agent_adoption_midpoint_quarter=params.agent_adoption_midpoint_quarter,
            # No policy
            ubi_monthly_per_person=0,
            compute_tax_rate=0,
            retraining_investment=0,
        )
        baseline = EconomySimulator(params=baseline_params).run()
        bl = len(baseline.labels) - 1

        pc1, pc2, pc3, pc4 = st.columns(4)
        ur_diff = (results.unemployment_rate[last] - baseline.unemployment_rate[bl]) * 100
        gdp_diff = (results.gdp[last] - baseline.gdp[bl]) / baseline.gdp[bl] * 100
        sp_diff = (results.sp500[last] - baseline.sp500[bl]) / baseline.sp500[bl] * 100
        md_diff = (results.mortgage_delinquency[last] - baseline.mortgage_delinquency[bl]) * 100

        pc1.metric("Unemployment vs No Policy", f"{ur_diff:+.1f}pp")
        pc2.metric("GDP vs No Policy", f"{gdp_diff:+.1f}%")
        pc3.metric("S&P vs No Policy", f"{sp_diff:+.1f}%")
        pc4.metric("Mortgage Delinq. vs No Policy", f"{md_diff:+.1f}pp")

# ── TAB: AI Progress ────────────────────────────────────────────────
with tab_ai:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            line_chart(
                labels, results.ai_capability,
                "AI Capability Index (1.0 = 2025)", "Index", color="#2ca02c",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            line_chart(
                labels, results.ai_effectiveness,
                "AI Cost-Effectiveness (capability / cost)", "Ratio",
                color="#ff7f0e",
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            line_chart(
                labels, results.ai_cost_index,
                "AI Cost Index (1.0 = 2025)", "Index", color="#d62728",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            line_chart(
                labels, results.ai_investment,
                "AI Investment ($B/year)", "$B", color="#9467bd",
            ),
            use_container_width=True,
        )

    # Adoption table
    st.subheader("AI Adoption Levels at End of Simulation")
    adopt_data = {
        "Sector": results.sector_names,
        "AI Adoption (%)": [f"{results.ai_adoption[last, i] * 100:.1f}" for i in range(len(results.sector_names))],
        "Employment (M)": [f"{results.employment[last, i]:.1f}" for i in range(len(results.sector_names))],
        "Employment Change (%)": [
            f"{(results.employment[last, i] - results.employment[0, i]) / results.employment[0, i] * 100:+.1f}"
            for i in range(len(results.sector_names))
        ],
        "Avg Wage ($K)": [f"{results.avg_wage[last, i]:.0f}" for i in range(len(results.sector_names))],
    }
    st.dataframe(pd.DataFrame(adopt_data), hide_index=True, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────
st.divider()
st.caption(
    "This is a simplified system dynamics model for educational exploration. "
    "It captures the qualitative feedback loops described in the Citrini report "
    "but should not be used for actual financial decisions. "
    "Parameters can be tuned in the sidebar to explore different scenarios."
)
