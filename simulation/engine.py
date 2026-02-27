"""
Economy simulation engine.

Models the key feedback loops from Citrini Research's
"2028 Global Intelligence Crisis":

1. Intelligence Displacement Spiral
   AI improves -> jobs displaced -> savings reinvested in AI -> AI improves

2. Consumer Demand Loop
   Job losses -> spending falls -> revenues fall -> more AI to cut costs

3. Wage Compression
   Displaced white-collar workers flood service sector -> economy-wide wage decline

4. Financial Stress
   Income impairment -> mortgage stress -> wealth effect -> spending decline

5. Fiscal Pressure
   Tax receipts fall as labor share drops -> deficit widens

6. Intermediation Collapse
   AI agents -> friction removal -> intermediation revenue collapses

And three positive channels that can offset the crisis:

7. AI Deflation
   AI reduces production costs -> consumer prices fall -> purchasing power rises

8. New Job Creation
   AI capability growth -> new economic categories emerge (AI trainers,
   AI-augmented services, prompt engineers, etc.)

9. Monetary Policy Response
   Fed cuts rates in response to unemployment/deflation -> stabilizes
   investment, housing, and asset prices
"""

import numpy as np
from dataclasses import dataclass
from typing import List

from .config import SectorConfig, SimulationParams, DEFAULT_SECTORS, quarter_labels

# BLS: total employer cost ~1.7-1.8x base wages (benefits, FICA, etc.)
# This maps displayed wages to the total labor share of GDP (~56%)
COMPENSATION_MULTIPLIER = 1.76


def _compute_hwi(income_ratio, unemployment_rate, gini_val,
                 mortgage_delinq, home_price_idx, savings_rate_val):
    """Human Wellbeing Index (0-100) — composite welfare metric."""
    clamp = lambda x: max(0.0, min(100.0, x))
    income      = max(0.0, min(150.0, income_ratio * 100))
    # 667 scaling: 19pp unemployment excess → score = 0 (Great Depression level)
    employment  = clamp(100 - max(0, unemployment_rate - 0.04) * 667)
    equality    = clamp((0.70 - gini_val) / 0.30 * 100)
    mort_sub    = clamp((0.15 - mortgage_delinq) / 0.135 * 100)
    hpi_sub     = clamp((home_price_idx - 40) / 60 * 100)
    housing     = 0.6 * mort_sub + 0.4 * hpi_sub
    security    = clamp(100 - abs(savings_rate_val - 0.05) * 500)
    raw = (0.30*income + 0.25*employment + 0.20*equality
         + 0.15*housing + 0.10*security)
    # 0.84 calibration: chosen so 2025 baseline starts at ~75 ("good but not great")
    return max(0.0, min(100.0, raw * 0.84))


@dataclass
class SimulationResults:
    """All output time series from the simulation."""

    labels: List[str]

    # AI state
    ai_capability: np.ndarray
    ai_cost_index: np.ndarray
    ai_effectiveness: np.ndarray

    # Per-sector arrays: shape (num_quarters+1, num_sectors)
    employment: np.ndarray
    avg_wage: np.ndarray
    ai_adoption: np.ndarray
    sector_names: List[str]

    # Aggregate time series
    total_employment: np.ndarray
    unemployment_rate: np.ndarray
    total_labor_income: np.ndarray
    consumer_spending: np.ndarray
    gdp: np.ndarray
    gdp_growth_annualized: np.ndarray
    sp500: np.ndarray
    sp500_drawdown: np.ndarray
    mortgage_delinquency: np.ndarray
    home_price_index: np.ndarray
    federal_deficit_pct_gdp: np.ndarray
    labor_share: np.ndarray
    gini: np.ndarray
    savings_rate: np.ndarray
    agent_adoption: np.ndarray

    # Derived
    total_jobs_displaced: np.ndarray
    ai_investment: np.ndarray
    wellbeing_index: np.ndarray

    # Positive channels
    price_level: np.ndarray
    purchasing_power: np.ndarray
    fed_funds_rate: np.ndarray
    new_ai_jobs: np.ndarray

    # Milestones: list of (quarter_index, label)
    milestones: List[tuple]


class EconomySimulator:
    """System dynamics simulator for AI-driven economic transformation."""

    def __init__(
        self,
        sectors: List[SectorConfig] = None,
        params: SimulationParams = None,
    ):
        self.sectors = sectors or list(DEFAULT_SECTORS)
        self.params = params or SimulationParams()

    def _sigmoid(self, x: float, midpoint: float, steepness: float) -> float:
        return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))

    def run(self) -> SimulationResults:
        p = self.params
        n = p.num_quarters + 1
        ns = len(self.sectors)

        # --- Allocate arrays ---
        ai_cap = np.zeros(n)
        ai_cost = np.zeros(n)
        ai_eff = np.zeros(n)

        emp = np.zeros((n, ns))
        wage = np.zeros((n, ns))
        adopt = np.zeros((n, ns))

        total_emp = np.zeros(n)
        unemp_rate = np.zeros(n)
        labor_inc = np.zeros(n)  # total compensation ($B)
        cons_spend = np.zeros(n)
        gdp_arr = np.zeros(n)
        gdp_growth = np.zeros(n)
        sp500 = np.zeros(n)
        sp500_dd = np.zeros(n)
        mort_delinq = np.zeros(n)
        home_px = np.zeros(n)
        deficit_pct = np.zeros(n)
        labor_share = np.zeros(n)
        gini = np.zeros(n)
        sav_rate = np.zeros(n)
        agent_adopt = np.zeros(n)
        jobs_displaced = np.zeros(n)
        ai_invest = np.zeros(n)
        hwi = np.zeros(n)
        price_lvl = np.zeros(n)
        fed_rate = np.zeros(n)
        new_ai_jobs = np.zeros(n)

        milestones = []
        milestone_flags = set()

        # --- Initial conditions (2025 US baseline) ---
        ai_cap[0] = 1.0
        ai_cost[0] = 1.0
        ai_eff[0] = 1.0

        for i, s in enumerate(self.sectors):
            emp[0, i] = s.employment_millions
            wage[0, i] = s.avg_annual_wage_k
            adopt[0, i] = 0.03  # Pew/McKinsey 2024: ~3% of US workers actively use AI for core tasks

        # 4% natural rate (CBO NAIRU estimate 2024)
        labor_force = sum(s.employment_millions for s in self.sectors) / 0.96
        total_emp[0] = sum(s.employment_millions for s in self.sectors)
        unemp_rate[0] = 1.0 - total_emp[0] / labor_force

        # Total labor compensation = wages * multiplier
        wage_income_0 = sum(
            s.employment_millions * s.avg_annual_wage_k for s in self.sectors
        )
        labor_inc[0] = wage_income_0 * COMPENSATION_MULTIPLIER  # ~$16.2T

        gdp_0 = 29_000.0               # BEA: US nominal GDP ~$29T in 2025
        cons_spend_0 = gdp_0 * 0.68    # BEA: PCE is ~68% of GDP (stable 60-year average)
        gov_spend_0 = gdp_0 * 0.20     # BEA: government consumption ~20% of GDP
        invest_0 = gdp_0 * 0.18        # BEA: gross private domestic investment ~18% of GDP
        net_exports_0 = gdp_0 * (-0.06)  # BEA: US trade deficit ~6% of GDP (2024)

        # Non-labor consumer income (capital gains, dividends, transfers, etc.)
        total_consumer_income_0 = cons_spend_0 / (1 - p.base_savings_rate)
        non_labor_income = total_consumer_income_0 - labor_inc[0]

        gdp_arr[0] = gdp_0
        cons_spend[0] = cons_spend_0
        sp500[0] = 6000.0  # S&P 500 ~6000 at 2025 baseline
        sp500_dd[0] = 0.0
        mort_delinq[0] = 0.015  # MBA: current US delinquency rate ~1.5% (2024)
        home_px[0] = 100.0
        deficit_pct[0] = -0.06
        labor_share[0] = labor_inc[0] / gdp_0  # ~0.56
        gini[0] = 0.49  # Census Bureau: US Gini ~0.49 (2023)
        sav_rate[0] = p.base_savings_rate
        agent_adopt[0] = 0.02
        gdp_growth[0] = 0.025
        price_lvl[0] = 1.0
        fed_rate[0] = 0.045  # 4.5% starting Fed funds rate

        hwi[0] = _compute_hwi(
            1.0, unemp_rate[0], gini[0],
            mort_delinq[0], home_px[0], sav_rate[0],
        )

        cumulative_displaced = 0.0

        # Smoothed labor income for consumer spending — models the lag
        # between job loss and spending cuts. The report describes this:
        # "High earners used their higher-than-average savings to maintain
        # the appearance of normalcy for two or three quarters."
        spending_labor_inc = labor_inc[0]

        # Smoothed UBI income — new policy takes 2-3 quarters to fully
        # affect spending (phase-in, behavioral adjustment, confidence).
        spending_ubi = 0.0

        # --- Simulation loop ---
        for t in range(p.num_quarters):

            # ============================================================
            # 1. AI CAPABILITY AND COST
            # ============================================================
            # S-curve saturation: growth slows as capability approaches
            # physical/algorithmic ceiling (inspired by GATE model's CMOS
            # limits). Barely noticeable over 6 years, dominant over 25+.
            saturation = max(0.0, 1.0 - (ai_cap[t] / p.ai_capability_ceiling) ** 2)
            ai_cap[t + 1] = ai_cap[t] * (1 + p.ai_capability_quarterly_growth * saturation)

            eff_cost_decline = p.ai_cost_quarterly_decline * (
                1 - p.compute_tax_rate * 0.5
            )
            ai_cost[t + 1] = max(0.005, ai_cost[t] * (1 - eff_cost_decline))

            ai_eff[t + 1] = ai_cap[t + 1] / ai_cost[t + 1]

            # ============================================================
            # 2. ECONOMIC FEEDBACK PRESSURE
            # ============================================================
            if t > 0:
                rev_growth = (gdp_arr[t] - gdp_arr[t - 1]) / max(gdp_arr[t - 1], 1)
            else:
                rev_growth = 0.005

            margin_pressure = max(0, -rev_growth * 4) * p.margin_pressure_ai_feedback

            fin_stress = max(0, mort_delinq[t] - 0.02) * p.credit_tightening_feedback

            # ============================================================
            # 2b. MONETARY POLICY (Fed responds to lagged data)
            # ============================================================
            # Simple Taylor-inspired rule: cut rates when unemployment
            # exceeds 4% target or when deflation appears.
            unemp_gap_fed = max(0, unemp_rate[t] - 0.04)
            if t > 0 and price_lvl[t] < price_lvl[t - 1]:
                deflation_signal = (price_lvl[t - 1] - price_lvl[t]) / price_lvl[t - 1]
            else:
                deflation_signal = 0
            # Fed tolerates mild deflation (~2%/yr = ~0.5%/qtr) before reacting
            deflation_excess = max(0, deflation_signal - 0.005)
            target_rate = max(0.0, 0.045 - unemp_gap_fed * 4.0 - deflation_excess * 10.0)
            fed_rate[t + 1] = fed_rate[t] + p.fed_response_speed * (target_rate - fed_rate[t])
            fed_rate[t + 1] = max(0.0, min(0.06, fed_rate[t + 1]))

            # Normalized stimulus: 0 at 4.5%, 1 at 0%
            rate_stimulus = max(0, (0.045 - fed_rate[t + 1]) / 0.045)

            # ============================================================
            # 3. AGENTIC COMMERCE ADOPTION
            # ============================================================
            # Time-based S-curve scaled by AI capability growth.
            # No AI improvement → no agentic commerce adoption.
            base_agent = self._sigmoid(
                t + 1, p.agent_adoption_midpoint_quarter, p.agent_adoption_steepness
            )
            # Capability gate: saturates at 1.0 once AI has doubled
            cap_gate = min(1.0, max(0, (ai_cap[t + 1] / ai_cap[0]) - 1.0) / 1.0)
            agent_adopt[t + 1] = base_agent * cap_gate

            # ============================================================
            # 4. PER-SECTOR DYNAMICS
            # ============================================================
            quarter_displaced = 0.0
            displaced_wage_total = 0.0

            for i, sector in enumerate(self.sectors):
                speed = p.sector_automation_speeds.get(sector.name, 0.3)

                # Adoption driven by AI IMPROVEMENT above baseline, not
                # absolute level — no improvement means no new adoption.
                eff_improvement = max(0, ai_eff[t + 1] - ai_eff[0])
                # 0.015: calibrated so baseline reaches ~40-50% adoption in tech by 2030
                base_rate = speed * 0.015 * np.log1p(eff_improvement)

                effective_rate = base_rate * (1 + margin_pressure)

                if sector.is_intermediation:
                    # 0.012: calibrated to Citrini's "6% of GDP in intermediation revenue"
                    effective_rate += (
                        agent_adopt[t + 1] * 0.012 * sector.automation_susceptibility
                    )

                new_adopt = min(
                    sector.automation_susceptibility,
                    adopt[t, i] + effective_rate,
                )
                adopt[t + 1, i] = new_adopt

                wc = sector.white_collar_pct
                # Use adoption DELTA from initial baseline so that the
                # starting 3% adoption doesn't create phantom job losses.
                # "No AI" scenario stays stable because delta ≈ 0.
                adopt_delta = max(0, new_adopt - adopt[0, i])

                # Physical automation lags cognitive by ~2-3 years as
                # robotics catches up (warehouses, delivery, cashiers).
                # Starts at 3x AI capability (~Q10), full maturity at 9x (~Q19).
                # At maturity, 50% of blue-collar workers in automatable roles exposed.
                physical_maturity = min(1.0, max(0, (ai_cap[t + 1] / ai_cap[0] - 3.0)) / 6.0)
                exposed_fraction = wc + (1 - wc) * physical_maturity * 0.5
                target = emp[0, i] * (1 - adopt_delta * exposed_fraction)

                gap = emp[t, i] - target
                if gap > 0:
                    layoffs = gap * p.layoff_speed
                    emp[t + 1, i] = emp[t, i] - layoffs
                    quarter_displaced += layoffs
                    displaced_wage_total += layoffs * wage[t, i]
                else:
                    emp[t + 1, i] = emp[t, i]

                surplus_ratio = max(
                    0,
                    (sector.employment_millions - emp[t + 1, i])
                    / sector.employment_millions,
                )
                w_pressure = min(
                    p.wage_flexibility,
                    surplus_ratio * 0.08 + adopt_delta * 0.004,
                )
                wage[t + 1, i] = wage[t, i] * (1 - w_pressure)

            # ============================================================
            # 5. WORKER REDEPLOYMENT
            # ============================================================
            if quarter_displaced > 0:
                avg_displaced_wage = displaced_wage_total / quarter_displaced
            else:
                avg_displaced_wage = 0

            remaining = quarter_displaced

            # Diminishing returns: sqrt scaling — $25B gets you 50% of max,
            # $100B gets ~100%, $400B gets ~200% (but still capped at 15%)
            retrain_boost = min(0.15, 0.01 * np.sqrt(max(0, p.retraining_investment)))
            eff_redeploy = p.worker_redeployment_rate + retrain_boost

            absorb_order = sorted(
                range(ns), key=lambda idx: self.sectors[idx].automation_susceptibility
            )
            for i in absorb_order:
                if remaining <= 0:
                    break
                s = self.sectors[i]
                if s.automation_susceptibility < 0.5:
                    capacity = emp[t + 1, i] * 0.015
                    absorbed = min(remaining * eff_redeploy, capacity)
                    if absorbed > 0:
                        old_bill = emp[t + 1, i] * wage[t + 1, i]
                        new_bill = absorbed * wage[t + 1, i] * (
                            1 - p.displaced_wage_penalty
                        )
                        emp[t + 1, i] += absorbed
                        wage[t + 1, i] = (old_bill + new_bill) / emp[t + 1, i]
                        remaining -= absorbed

            cumulative_displaced += quarter_displaced
            jobs_displaced[t + 1] = cumulative_displaced

            # ============================================================
            # 5b. NEW AI-ECONOMY JOB CREATION
            # ============================================================
            # AI creates new job categories that didn't exist before:
            # AI trainers, prompt engineers, AI-augmented service providers,
            # AI maintenance, new AI-enabled businesses, etc.
            # Like the internet creating Google/Amazon/etc. jobs.
            cap_ratio = ai_cap[t + 1] / ai_cap[0]
            # Potential scales with AI capability, caps at ~10% of workforce
            potential = min(labor_force * 0.10, max(0, (cap_ratio - 1.0) * 1.5))
            # Education/retraining helps workers fill new roles (sqrt diminishing returns)
            education_factor = 0.3 + min(0.7, 0.05 * np.sqrt(max(0, p.retraining_investment)))
            # Time lag: new industries take time to form (ramps mid-simulation)
            time_ramp = self._sigmoid(t + 1, 8, 0.3)
            new_ai_jobs[t + 1] = potential * education_factor * time_ramp * p.new_job_creation_rate

            # ============================================================
            # 6. AGGREGATE EMPLOYMENT & INCOME
            # ============================================================
            total_emp[t + 1] = emp[t + 1].sum() + new_ai_jobs[t + 1]
            unemp_rate[t + 1] = max(0.02, 1 - total_emp[t + 1] / labor_force)

            # Total labor compensation with multiplier
            # New AI-economy wages grow as the sector matures — AI-complementary
            # work gets more valuable with AI capability (like how tech salaries
            # grew as the internet matured). $75K base → up to ~$140K.
            wage_inc = sum(
                emp[t + 1, i] * wage[t + 1, i] for i in range(ns)
            )
            new_job_wage = 75.0 * (1 + min(1.0, (ai_cap[t + 1] / ai_cap[0] - 1) * 0.1))
            new_job_wage_inc = new_ai_jobs[t + 1] * new_job_wage
            labor_inc[t + 1] = (wage_inc + new_job_wage_inc) * COMPENSATION_MULTIPLIER

            # ============================================================
            # 7. CONSUMER SPENDING
            # ============================================================
            unemp_excess = max(0, unemp_rate[t + 1] - 0.04)

            sr = p.base_savings_rate + unemp_excess * p.precautionary_savings_sensitivity
            sr = min(0.20, sr)
            sav_rate[t + 1] = sr

            # Fiscal drag: extreme deficits erode confidence and crowd out
            # investment. Tolerate up to ~15% deficit/GDP (wartime levels);
            # beyond that, bond markets and inflation expectations push back.
            prior_deficit_ratio = -deficit_pct[t]  # positive = deficit
            fiscal_drag = max(0, (prior_deficit_ratio - 0.15) * 1.0)
            fiscal_drag = min(0.25, fiscal_drag)

            confidence = max(0.50, 1.0 - unemp_excess * p.confidence_sensitivity - fiscal_drag)

            # UBI: $/month * 12 months * 260M adults → $B/year
            ubi_income = p.ubi_monthly_per_person * 12.0 * 260.0 / 1000.0

            # Smoothed labor income: displaced workers draw down savings
            # before cutting spending, creating a 2-3 quarter lag
            smooth_alpha = 0.35
            spending_labor_inc = (
                smooth_alpha * labor_inc[t + 1]
                + (1 - smooth_alpha) * spending_labor_inc
            )

            # Smooth UBI: new transfers take time to fully affect spending
            spending_ubi = (
                smooth_alpha * ubi_income
                + (1 - smooth_alpha) * spending_ubi
            )

            gdp_ratio = gdp_arr[t] / gdp_0
            adj_non_labor = non_labor_income * (0.5 + 0.5 * max(0.4, gdp_ratio))

            total_consumer_income = spending_labor_inc + adj_non_labor + spending_ubi
            cons_spend[t + 1] = total_consumer_income * (1 - sr) * confidence

            # Supply constraint: the economy has finite productive capacity.
            # AI increases capacity, but you can't buy goods that don't exist.
            # Excess demand above capacity creates inflation, not real output.
            ai_supply_boost = 1 + min(1.0, (ai_cap[t + 1] / ai_cap[0] - 1) * 0.15)
            supply_cap = cons_spend_0 * ai_supply_boost * 1.3  # 30% above AI-augmented baseline
            demand_inflation_qtr = 0.0
            if cons_spend[t + 1] > supply_cap:
                excess = cons_spend[t + 1] - supply_cap
                demand_inflation_qtr = min(0.10, (excess / supply_cap) * 0.15)
                # Only tiny fraction of excess becomes real spending; rest is inflation
                cons_spend[t + 1] = supply_cap + excess * 0.05

            interm_loss = (
                agent_adopt[t + 1] * p.intermediation_gdp_fraction * gdp_arr[t]
            )

            # ============================================================
            # 8. GDP
            # ============================================================
            labor_savings = max(0, labor_inc[0] - labor_inc[t + 1])
            ai_inv = labor_savings * 0.25
            ai_invest[t + 1] = ai_inv

            # Business investment is highly cyclical — the "investment
            # accelerator" means capex swings 2-3x more than GDP.
            # Use prior-quarter GDP ratio + unemployment, not mild
            # consumer confidence (which barely moves).
            gdp_ratio_inv = gdp_arr[t] / gdp_0
            inv_sentiment = max(0.30, gdp_ratio_inv - unemp_excess * 1.5)
            # Fed rate cuts stimulate business investment; fiscal drag crowds out
            crowding_out = max(0, (prior_deficit_ratio - 0.15) * 0.8)
            crowding_out = min(0.20, crowding_out)
            other_inv = invest_0 * inv_sentiment * (1 + rate_stimulus * 0.10) * (1 - crowding_out)

            auto_stab = unemp_excess * gdp_0 * 0.02
            # Government PURCHASES (G in GDP = C+I+G+NX): excludes transfer
            # payments like UBI since those flow through consumer spending (C).
            gov_purchases = gov_spend_0 + auto_stab
            # Total fiscal outlay (for deficit calculation): includes transfers
            gov_total_spending = gov_purchases + ubi_income

            # AI PRODUCTIVITY BOOST ("Ghost GDP")
            # AI augments remaining workers AND autonomously replaces lost
            # output. This is the report's core insight: "Real output per
            # hour rose at rates not seen since the 1950s" and "nominal GDP
            # repeatedly printed mid-to-high single-digit growth" — even as
            # workers were being laid off. The output shows up in national
            # accounts as corporate profits but never circulates through
            # households.
            #
            # Weighted average AI adoption across sectors (by income share)
            weighted_adoption = sum(
                adopt[t + 1, i] * emp[t + 1, i] * wage[t + 1, i]
                for i in range(ns)
            ) / max(wage_inc, 1)
            # Net adoption above 3% baseline (so boost starts at zero)
            net_adoption = max(0, weighted_adoption - 0.03)
            # Saturating productivity boost: max 30% of GDP; S-curve shape.
            # 3.0 steepness + 0.30 ceiling calibrated to Citrini: "mid-to-high single-digit growth"
            raw_boost = (1 - np.exp(-net_adoption * 3.0)) * 0.30 * gdp_0
            # Demand dampening: can't sell output nobody can afford to buy
            demand_base = cons_spend[t + 1] + ai_inv + other_inv + gov_purchases
            demand_ratio = demand_base / (cons_spend_0 + invest_0 + gov_spend_0)
            demand_damper = min(1.0, max(0, demand_ratio) ** 1.2)
            ai_productivity_boost = raw_boost * demand_damper

            # Dynamic net exports: US can become dominant AI exporter.
            # AI leadership boosts service exports; weak demand reduces imports.
            # Small positive contribution (~$100-500B over 6 years).
            ai_export_boost = min(0.02 * gdp_0, max(0, ai_cap[t + 1] / ai_cap[0] - 1.0) * 0.003 * gdp_0)
            import_adjustment = max(0, 1 - demand_ratio) * 0.01 * gdp_0
            net_exports = net_exports_0 + ai_export_boost + import_adjustment

            gdp_arr[t + 1] = (
                cons_spend[t + 1]
                + ai_inv
                + other_inv
                + gov_purchases  # G = government purchases (not transfers)
                + net_exports
                + ai_productivity_boost
                - interm_loss
            )
            gdp_arr[t + 1] = max(gdp_arr[t + 1], gdp_0 * 0.4)

            if gdp_arr[t] > 0:
                gdp_growth[t + 1] = (gdp_arr[t + 1] / gdp_arr[t]) ** 4 - 1
            else:
                gdp_growth[t + 1] = 0

            # ============================================================
            # 8b. AI-DRIVEN DEFLATION
            # ============================================================
            # AI cost-effectiveness (capability/cost) compounds exponentially
            # and directly drives down production costs → consumer prices.
            # Log scaling: 2x efficiency → modest, 100x → very significant.
            # Gated by adoption (need actual deployment, not just capability).
            eff_ratio = ai_eff[t + 1] / ai_eff[0]
            deflation_potential = np.log(max(1.0, eff_ratio)) * 0.002
            adoption_gate = min(1.0, net_adoption / 0.20)
            ai_deflation_qtr = min(0.02, deflation_potential * adoption_gate
                                   * (p.ai_deflation_rate / 0.006))
            # Net price change: AI deflation vs demand-pull inflation
            price_lvl[t + 1] = max(0.60, price_lvl[t] * (1 - ai_deflation_qtr + demand_inflation_qtr))

            # ============================================================
            # 9. FINANCIAL INDICATORS
            # ============================================================
            labor_share[t + 1] = labor_inc[t + 1] / max(gdp_arr[t + 1], 1)

            # Mortgage delinquency
            # People pay mortgages from ALL income (labor + UBI + non-labor).
            # Deflation means fixed payments are easier to meet.
            # Use real total consumer income for mortgage stress, not just
            # nominal labor income — otherwise UBI and deflation are invisible.
            real_total_income_ratio = (total_consumer_income / total_consumer_income_0) / price_lvl[t + 1]
            real_inc_decline = max(0, 1 - real_total_income_ratio)
            # Keep labor-only decline for Gini (measures market income inequality)
            labor_inc_decline = max(
                0, (labor_inc[0] - labor_inc[t + 1]) / labor_inc[0]
            )
            md = 0.015 + real_inc_decline * p.mortgage_stress_sensitivity
            md += unemp_excess * 0.12
            # Fed rate cuts reduce mortgage stress (refinancing, lower payments)
            md -= rate_stimulus * 0.005
            mort_delinq[t + 1] = min(0.15, max(0.01, md))

            # Home prices — driven by what buyers can actually afford (real income)
            delinq_pressure = max(0, mort_delinq[t + 1] - 0.02)
            inc_factor = real_total_income_ratio
            qtr_price_chg = (
                (inc_factor - 1) * p.home_price_income_sensitivity * 0.15
                - delinq_pressure * 0.4  # 2008 was ~27% national; 0.8 was too aggressive
                + rate_stimulus * 0.005  # low rates support home prices
            )
            qtr_price_chg = max(-0.06, min(0.03, qtr_price_chg))
            home_px[t + 1] = max(40, home_px[t] * (1 + qtr_price_chg))

            # S&P 500 — models initial euphoria then crash
            # Corporate profits = GDP - labor compensation
            # Initially: AI cuts labor costs, GDP maintained by Ghost GDP → profits soar
            # Later: Demand destruction hits GDP → profits eventually fall
            corp_profits = gdp_arr[t + 1] - labor_inc[t + 1]
            init_profits = gdp_0 - labor_inc[0]
            # Discount profits that come from fiscal stimulus (UBI, auto-stabilizers)
            # — the market doesn't capitalize transfer-fueled revenue at full multiple
            fiscal_stimulus = auto_stab + ubi_income
            organic_profits = corp_profits - fiscal_stimulus * 0.5
            earnings_ratio = max(0.3, organic_profits / max(init_profits, 1))

            # AI euphoria: builds with actual AI capability growth, not just
            # time. "No AI" scenario (flat capability) gets no euphoria.
            cap_growth = (ai_cap[t + 1] / ai_cap[0]) - 1.0  # cumulative growth
            raw_euphoria = min(1.30, 1.0 + cap_growth * 0.08)
            # Euphoria erodes with unemployment and mortgage stress
            euphoria_erosion = (
                unemp_excess * 3.0
                + max(0, mort_delinq[t + 1] - 0.03) * 5
            )
            ai_euphoria = max(1.0, raw_euphoria - euphoria_erosion)

            # Macro risk premium: rises with unemployment, mortgage stress,
            # and unsustainable fiscal deficits
            fiscal_risk = max(0, -deficit_pct[t] - 0.08) * 2  # risk above 8% deficit
            stress = (
                unemp_excess * 2.0
                + fin_stress
                + max(0, mort_delinq[t + 1] - 0.02) * 6
                + fiscal_risk
            )
            # Net sentiment: euphoria vs fear
            sentiment = max(0.35, ai_euphoria - stress * p.equity_risk_premium_sensitivity)

            sp500[t + 1] = 6000 * earnings_ratio * sentiment
            sp500[t + 1] = max(1500, min(12000, sp500[t + 1]))

            # Drawdown from peak
            peak = np.max(sp500[: t + 2])
            sp500_dd[t + 1] = (sp500[t + 1] - peak) / peak if peak > 0 else 0

            # Federal deficit (uses total fiscal outlay including transfers)
            corp_profits = gdp_arr[t + 1] - labor_inc[t + 1]
            income_tax = labor_inc[t + 1] * 0.14
            corp_tax = max(0, corp_profits) * 0.08
            compute_tax_rev = ai_inv * p.compute_tax_rate
            total_receipts = income_tax + corp_tax + compute_tax_rev
            deficit = gov_total_spending - total_receipts
            deficit_pct[t + 1] = -deficit / max(gdp_arr[t + 1], 1)

            # Gini — Census Bureau: US Gini ~0.49 (2023)
            # Uses labor income decline (market inequality), not total income.
            # UBI directly reduces inequality (income redistribution).
            # New AI jobs and tight labor markets can also reduce inequality.
            ubi_equality_boost = min(0.04, (ubi_income / total_consumer_income_0) * 0.2)
            # Broadly accessible new AI-economy jobs reduce inequality
            new_job_equality = min(0.03, new_ai_jobs[t + 1] / labor_force * 0.5)
            # Below-4% unemployment gives workers bargaining power
            tight_labor_bonus = max(0, 0.04 - unemp_rate[t + 1]) * 0.8
            g = (0.49 + labor_inc_decline * 0.4 + unemp_excess * 0.5
                 - ubi_equality_boost - new_job_equality - tight_labor_bonus)
            gini[t + 1] = min(0.70, max(0.35, g))  # floor 0.35 (Nordic-level)

            # ============================================================
            # 10. HUMAN WELLBEING INDEX
            # ============================================================
            # Use REAL income: nominal income / price level. If AI makes
            # everything cheaper, each dollar of income buys more.
            real_income_ratio = (total_consumer_income / total_consumer_income_0) / price_lvl[t + 1]
            hwi[t + 1] = _compute_hwi(
                real_income_ratio,
                unemp_rate[t + 1], gini[t + 1],
                mort_delinq[t + 1], home_px[t + 1], sav_rate[t + 1],
            )

            # ============================================================
            # 11. MILESTONE EVENTS
            # ============================================================
            ur = unemp_rate[t + 1]
            dd = sp500_dd[t + 1]

            if ur >= 0.05 and "unemp_5" not in milestone_flags:
                milestones.append((t + 1, "Unemployment crosses 5%"))
                milestone_flags.add("unemp_5")
            if ur >= 0.07 and "unemp_7" not in milestone_flags:
                milestones.append((t + 1, "Unemployment crosses 7%"))
                milestone_flags.add("unemp_7")
            if ur >= 0.10 and "unemp_10" not in milestone_flags:
                milestones.append((t + 1, "Unemployment crosses 10%"))
                milestone_flags.add("unemp_10")
            if dd <= -0.20 and "bear" not in milestone_flags:
                milestones.append((t + 1, "S&P 500 enters bear market (-20%)"))
                milestone_flags.add("bear")
            if dd <= -0.38 and "crisis" not in milestone_flags:
                milestones.append(
                    (t + 1, "Global Intelligence Crisis level (-38%)")
                )
                milestone_flags.add("crisis")
            if mort_delinq[t + 1] >= 0.03 and "mort_stress" not in milestone_flags:
                milestones.append((t + 1, "Prime mortgage stress emerging"))
                milestone_flags.add("mort_stress")
            if mort_delinq[t + 1] >= 0.05 and "mort_crisis" not in milestone_flags:
                milestones.append((t + 1, "Mortgage delinquencies accelerating"))
                milestone_flags.add("mort_crisis")
            if labor_share[t + 1] <= 0.46 and "labor_46" not in milestone_flags:
                milestones.append(
                    (t + 1, "Labor share of GDP falls to report's 46% level")
                )
                milestone_flags.add("labor_46")
            if hwi[t + 1] < 65 and "hwi_65" not in milestone_flags:
                milestones.append((t + 1, "Wellbeing Index: notable decline (<65)"))
                milestone_flags.add("hwi_65")
            if hwi[t + 1] < 50 and "hwi_50" not in milestone_flags:
                milestones.append((t + 1, "Wellbeing Index: crisis level (<50)"))
                milestone_flags.add("hwi_50")
            if hwi[t + 1] < 40 and "hwi_40" not in milestone_flags:
                milestones.append((t + 1, "Wellbeing Index: severe crisis (<40)"))
                milestone_flags.add("hwi_40")

        # --- Build results ---
        labels = quarter_labels(p.num_quarters, p.start_year)

        return SimulationResults(
            labels=labels,
            ai_capability=ai_cap,
            ai_cost_index=ai_cost,
            ai_effectiveness=ai_eff,
            employment=emp,
            avg_wage=wage,
            ai_adoption=adopt,
            sector_names=[s.name for s in self.sectors],
            total_employment=total_emp,
            unemployment_rate=unemp_rate,
            total_labor_income=labor_inc,
            consumer_spending=cons_spend,
            gdp=gdp_arr,
            gdp_growth_annualized=gdp_growth,
            sp500=sp500,
            sp500_drawdown=sp500_dd,
            mortgage_delinquency=mort_delinq,
            home_price_index=home_px,
            federal_deficit_pct_gdp=deficit_pct,
            labor_share=labor_share,
            gini=gini,
            savings_rate=sav_rate,
            agent_adoption=agent_adopt,
            total_jobs_displaced=jobs_displaced,
            ai_investment=ai_invest,
            wellbeing_index=hwi,
            price_level=price_lvl,
            purchasing_power=1.0 / price_lvl,
            fed_funds_rate=fed_rate,
            new_ai_jobs=new_ai_jobs,
            milestones=milestones,
        )
