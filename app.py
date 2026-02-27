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
        # Use invisible scatter markers at milestone points so the
        # label only appears on hover — no permanent clutter.
        ms_x, ms_y, ms_text = [], [], []
        for qi, label in milestones:
            if 0 <= qi < len(y):
                ms_x.append(labels[qi])
                ms_y.append(y[qi])
                ms_text.append(label)
        if ms_x:
            fig.add_trace(
                go.Scatter(
                    x=ms_x, y=ms_y, mode="markers",
                    marker=dict(size=9, color="red", symbol="diamond",
                                line=dict(width=1, color="#333")),
                    text=ms_text,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                )
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
    new_job_rate = st.slider(
        "New AI Job Creation", 0, 100, int(preset.new_job_creation_rate * 100),
        help="How fast AI creates new job categories (trainers, AI-augmented roles, new businesses)",
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
    ai_deflation = st.slider(
        "AI Deflation Effect", 0, 20, int(preset.ai_deflation_rate * 1000),
        help="How much AI adoption reduces consumer prices (per mille/quarter)",
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
    new_job_creation_rate=new_job_rate / 100,
    ai_deflation_rate=ai_deflation / 1000,
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
c4.metric("Wellbeing Index", f"{results.wellbeing_index[last]:.0f}/100",
          f"{results.wellbeing_index[last] - results.wellbeing_index[0]:+.0f}")
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
tab_overview, tab_sectors, tab_financial, tab_policy, tab_ai, tab_method = st.tabs(
    ["Overview", "Sectors", "Financial", "Policy & Inequality", "AI Progress", "Methodology"]
)

# ── TAB: Overview ────────────────────────────────────────────────────
with tab_overview:
    st.plotly_chart(
        line_chart(
            labels, results.wellbeing_index,
            "Human Wellbeing Index (0-100)", "Index", color="#e45756",
            milestones=results.milestones, labels=labels,
        ),
        use_container_width=True,
    )

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

        # Jobs displaced vs created
        st.plotly_chart(
            multi_line(
                labels,
                {
                    "Jobs Displaced": results.total_jobs_displaced,
                    "New AI-Economy Jobs": results.new_ai_jobs,
                },
                "Jobs Displaced vs New AI Jobs (Millions)", "Millions",
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

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            line_chart(
                labels, results.savings_rate * 100,
                "Household Savings Rate (%)", "%", color="#bcbd22",
            ),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            line_chart(
                labels, results.purchasing_power * 100,
                "Purchasing Power (100 = 2025 dollar)", "%", color="#2ca02c",
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
                labels, results.fed_funds_rate * 100,
                "Fed Funds Rate (%)", "%", color="#2ca02c",
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
            ai_deflation_rate=params.ai_deflation_rate,
            new_job_creation_rate=params.new_job_creation_rate,
            # No policy
            ubi_monthly_per_person=0,
            compute_tax_rate=0,
            retraining_investment=0,
        )
        baseline = EconomySimulator(params=baseline_params).run()
        bl = len(baseline.labels) - 1

        pc0, pc1, pc2, pc3, pc4 = st.columns(5)
        hwi_diff = results.wellbeing_index[last] - baseline.wellbeing_index[bl]
        ur_diff = (results.unemployment_rate[last] - baseline.unemployment_rate[bl]) * 100
        gdp_diff = (results.gdp[last] - baseline.gdp[bl]) / baseline.gdp[bl] * 100
        sp_diff = (results.sp500[last] - baseline.sp500[bl]) / baseline.sp500[bl] * 100
        md_diff = (results.mortgage_delinquency[last] - baseline.mortgage_delinquency[bl]) * 100

        pc0.metric("Wellbeing vs No Policy", f"{hwi_diff:+.1f}")
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

# ── TAB: Methodology ─────────────────────────────────────────────────
with tab_method:
    st.header("Model Structure")
    st.markdown("""
This is a quarterly system dynamics model with nine interacting feedback loops:

1. **Intelligence Displacement Spiral** — AI improves → jobs displaced → savings reinvested in AI → AI improves
2. **Consumer Demand Loop** — Job losses → spending falls → revenues fall → more AI to cut costs
3. **Wage Compression** — Displaced workers flood service sector → economy-wide wage decline
4. **Financial Stress** — Income impairment → mortgage stress → wealth effect → spending decline
5. **Fiscal Pressure** — Tax receipts fall as labor share drops → deficit widens
6. **Intermediation Collapse** — AI agents remove friction → intermediation revenue collapses
7. **AI Deflation** — AI reduces production costs → consumer prices fall → purchasing power rises
8. **New Job Creation** — AI capability growth → new economic categories emerge
9. **Monetary Policy Response** — Fed cuts rates in response to unemployment/deflation → stabilizes investment and housing
""")

    st.header("Key Assumptions")
    st.markdown("""
- AI capability grows exponentially with S-curve saturation at physical limits
- Eight US economic sectors with distinct automation profiles (white-collar and blue-collar exposure)
- AI creates both cognitive and physical automation (physical lags cognitive by ~2-3 years as robotics matures)
- Consumer spending follows income with 2-3 quarter lag (savings buffer)
- Fed follows a Taylor-inspired rule (responds to unemployment gap and deflation)
- Supply-constrained economy: excess demand creates inflation, not output
- New AI-economy jobs emerge with a time lag (new industries take time to form)
- Inequality can improve when labor markets are tight and new jobs are broadly accessible
""")

    st.header("Calibration Sources")
    cal_data = {
        "Parameter": [
            "GDP (2025)", "Consumption share", "Compensation multiplier",
            "Initial AI adoption", "Natural unemployment rate",
            "Mortgage delinquency rate", "Gini coefficient",
            "S&P 500 level", "AI adoption rate",
            "Intermediation revenue", "AI productivity ceiling",
            "HWI employment scaling", "HWI calibration factor",
        ],
        "Value": [
            "$29T", "68% of GDP", "1.76x",
            "3%", "4%",
            "1.5%", "0.49",
            "6,000", "0.015/qtr base",
            "6% of GDP", "30% of GDP max",
            "667", "0.84",
        ],
        "Source / Reasoning": [
            "BEA: US nominal GDP ~$29T in 2025",
            "BEA: PCE is ~68% of GDP (stable 60-year average)",
            "BLS: total employer cost ~1.7-1.8x base wages (benefits, FICA, etc.)",
            "Pew/McKinsey 2024: ~3% of US workers actively use AI for core tasks",
            "CBO NAIRU estimate 2024",
            "MBA: current US mortgage delinquency rate ~1.5% (2024)",
            "Census Bureau: US Gini ~0.49 (2023)",
            "S&P 500 ~6,000 at 2025 baseline",
            "Calibrated: baseline reaches ~40-50% adoption in tech by 2030",
            "Calibrated to Citrini's '6% of GDP in intermediation revenue'",
            "Calibrated to Citrini: 'mid-to-high single-digit growth'",
            "Scaling: 19pp unemployment excess → score = 0 (Great Depression level)",
            "Chosen so 2025 baseline starts at ~75 ('good but not great')",
        ],
    }
    st.dataframe(pd.DataFrame(cal_data), hide_index=True, use_container_width=True)

    st.header("Known Limitations")
    st.markdown("""
- **No forward-looking expectations**: Agents react to current conditions, don't anticipate. No rational expectations or adaptive learning.
- **Closed economy with simplified trade**: No exchange rates or trade policy dynamics. Net exports respond to AI leadership but omit global competition effects.
- **No banking/credit channel**: Mortgage stress is modeled but there are no bank failures or credit crunch cascades.
- **No agent heterogeneity**: Single representative consumer per sector. No income distribution within sectors, no geographic variation.
- **No non-AI technological progress**: All productivity gains come from AI. Baseline economy has zero TFP growth.
- **Coefficients calibrated to plausibility, not estimated from data**: Parameters are chosen to produce reasonable trajectories, not fitted to historical data via regression or Bayesian estimation.
- **Directions more reliable than magnitudes**: The model captures whether policy X helps or hurts more reliably than by exactly how much.
""")

    # ── Sensitivity Analysis ──────────────────────────────────────
    st.header("Sensitivity Analysis")
    st.markdown(
        "Each parameter is varied **±20%** from current settings. "
        "Bars show how much the output metric changes from baseline."
    )

    @st.cache_data
    def run_sensitivity(_params_dict, num_quarters):
        """Run ±20% sweeps for key parameters and return deltas."""
        from dataclasses import asdict

        base_params = SimulationParams(**_params_dict, num_quarters=num_quarters)
        base_result = EconomySimulator(params=base_params).run()
        bl = len(base_result.labels) - 1
        base_hwi = base_result.wellbeing_index[bl]
        base_unemp = base_result.unemployment_rate[bl] * 100
        base_gdp_chg = (base_result.gdp[bl] - base_result.gdp[0]) / base_result.gdp[0] * 100

        sweep_params = [
            ("ai_capability_quarterly_growth", "AI Capability Growth"),
            ("ai_cost_quarterly_decline", "AI Cost Decline"),
            ("worker_redeployment_rate", "Worker Redeployment"),
            ("displaced_wage_penalty", "Displaced Wage Penalty"),
            ("layoff_speed", "Layoff Speed"),
            ("new_job_creation_rate", "New Job Creation"),
            ("confidence_sensitivity", "Confidence Sensitivity"),
            ("ai_deflation_rate", "AI Deflation Rate"),
        ]

        rows = []
        for attr, label in sweep_params:
            base_val = _params_dict[attr]
            for direction, mult in [("-20%", 0.8), ("+20%", 1.2)]:
                tweaked = dict(_params_dict)
                tweaked[attr] = base_val * mult
                tp = SimulationParams(**tweaked, num_quarters=num_quarters)
                r = EconomySimulator(params=tp).run()
                rl = len(r.labels) - 1
                rows.append({
                    "param": label,
                    "direction": direction,
                    "hwi_delta": r.wellbeing_index[rl] - base_hwi,
                    "unemp_delta": r.unemployment_rate[rl] * 100 - base_unemp,
                    "gdp_delta": (r.gdp[rl] - r.gdp[0]) / r.gdp[0] * 100 - base_gdp_chg,
                })
        return rows

    # Build a hashable dict of the params we sweep (exclude num_quarters
    # and non-swept fields so cache key stays stable)
    _sweep_keys = [
        "ai_capability_quarterly_growth", "ai_cost_quarterly_decline",
        "worker_redeployment_rate", "displaced_wage_penalty",
        "layoff_speed", "new_job_creation_rate",
        "confidence_sensitivity", "ai_deflation_rate",
        # Include other params that affect the simulation
        "ai_capability_ceiling", "base_savings_rate",
        "precautionary_savings_sensitivity", "wage_flexibility",
        "agent_adoption_midpoint_quarter", "agent_adoption_steepness",
        "intermediation_gdp_fraction", "mortgage_stress_sensitivity",
        "home_price_income_sensitivity", "equity_earnings_weight",
        "equity_risk_premium_sensitivity", "ubi_monthly_per_person",
        "compute_tax_rate", "retraining_investment",
        "fed_response_speed", "margin_pressure_ai_feedback",
        "credit_tightening_feedback",
    ]
    _params_for_sweep = {k: getattr(params, k) for k in _sweep_keys}
    # Sector automation speeds need special handling (dict → tuple for hashing)
    _params_for_sweep["sector_automation_speeds"] = params.sector_automation_speeds

    sensitivity_rows = run_sensitivity(_params_for_sweep, params.num_quarters)
    sens_df = pd.DataFrame(sensitivity_rows)

    def tornado_chart(df, metric_col, title, xaxis_label):
        """Build a horizontal tornado chart for one output metric."""
        low = df[df["direction"] == "-20%"][["param", metric_col]].rename(
            columns={metric_col: "low"}
        )
        high = df[df["direction"] == "+20%"][["param", metric_col]].rename(
            columns={metric_col: "high"}
        )
        merged = low.merge(high, on="param")
        # Sort by total absolute spread (most sensitive at top)
        merged["spread"] = merged["high"].abs() + merged["low"].abs()
        merged = merged.sort_values("spread", ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=merged["param"], x=merged["low"], name="-20%",
            orientation="h", marker_color="#4e79a7",
        ))
        fig.add_trace(go.Bar(
            y=merged["param"], x=merged["high"], name="+20%",
            orientation="h", marker_color="#e15759",
        ))
        fig.update_layout(
            **CHART_THEME,
            title=dict(text=title, font=dict(size=14)),
            xaxis_title=xaxis_label,
            barmode="overlay",
            height=340,
            margin=dict(l=160, r=20, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        return fig

    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.plotly_chart(
            tornado_chart(sens_df, "hwi_delta", "Wellbeing Index", "HWI change"),
            use_container_width=True,
        )
    with tc2:
        st.plotly_chart(
            tornado_chart(sens_df, "unemp_delta", "Unemployment Rate", "pp change"),
            use_container_width=True,
        )
    with tc3:
        st.plotly_chart(
            tornado_chart(sens_df, "gdp_delta", "GDP Change", "pp change"),
            use_container_width=True,
        )

# ── Footer ───────────────────────────────────────────────────────────
st.divider()
st.caption(
    "This is a simplified system dynamics model for educational exploration. "
    "It captures the qualitative feedback loops described in the Citrini report "
    "but should not be used for actual financial decisions. "
    "Parameters can be tuned in the sidebar to explore different scenarios."
)
