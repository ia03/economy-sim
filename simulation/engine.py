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
"""

import numpy as np
from dataclasses import dataclass
from typing import List

from .config import SectorConfig, SimulationParams, DEFAULT_SECTORS, quarter_labels

# Total compensation is ~1.76x base salary (benefits, employer taxes, etc.)
# This maps displayed wages to the total labor share of GDP (~56%)
COMPENSATION_MULTIPLIER = 1.76


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

        milestones = []
        milestone_flags = set()

        # --- Initial conditions (2025 US baseline) ---
        ai_cap[0] = 1.0
        ai_cost[0] = 1.0
        ai_eff[0] = 1.0

        for i, s in enumerate(self.sectors):
            emp[0, i] = s.employment_millions
            wage[0, i] = s.avg_annual_wage_k
            adopt[0, i] = 0.03

        labor_force = sum(s.employment_millions for s in self.sectors) / 0.96
        total_emp[0] = sum(s.employment_millions for s in self.sectors)
        unemp_rate[0] = 1.0 - total_emp[0] / labor_force

        # Total labor compensation = wages * multiplier
        wage_income_0 = sum(
            s.employment_millions * s.avg_annual_wage_k for s in self.sectors
        )
        labor_inc[0] = wage_income_0 * COMPENSATION_MULTIPLIER  # ~$16.2T

        gdp_0 = 29_000.0
        cons_spend_0 = gdp_0 * 0.68
        gov_spend_0 = gdp_0 * 0.20
        invest_0 = gdp_0 * 0.18
        net_exports_0 = gdp_0 * (-0.06)

        # Non-labor consumer income (capital gains, dividends, transfers, etc.)
        total_consumer_income_0 = cons_spend_0 / (1 - p.base_savings_rate)
        non_labor_income = total_consumer_income_0 - labor_inc[0]

        gdp_arr[0] = gdp_0
        cons_spend[0] = cons_spend_0
        sp500[0] = 6000.0
        sp500_dd[0] = 0.0
        mort_delinq[0] = 0.015
        home_px[0] = 100.0
        deficit_pct[0] = -0.06
        labor_share[0] = labor_inc[0] / gdp_0  # ~0.56
        gini[0] = 0.49
        sav_rate[0] = p.base_savings_rate
        agent_adopt[0] = 0.02
        gdp_growth[0] = 0.025

        cumulative_displaced = 0.0

        # --- Simulation loop ---
        for t in range(p.num_quarters):

            # ============================================================
            # 1. AI CAPABILITY AND COST
            # ============================================================
            ai_cap[t + 1] = ai_cap[t] * (1 + p.ai_capability_quarterly_growth)

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
            # 3. AGENTIC COMMERCE ADOPTION
            # ============================================================
            agent_adopt[t + 1] = self._sigmoid(
                t + 1, p.agent_adoption_midpoint_quarter, p.agent_adoption_steepness
            )

            # ============================================================
            # 4. PER-SECTOR DYNAMICS
            # ============================================================
            quarter_displaced = 0.0
            displaced_wage_total = 0.0

            for i, sector in enumerate(self.sectors):
                speed = p.sector_automation_speeds.get(sector.name, 0.3)

                base_rate = speed * 0.015 * np.log1p(ai_eff[t + 1])

                effective_rate = base_rate * (1 + margin_pressure)

                if sector.is_intermediation:
                    effective_rate += (
                        agent_adopt[t + 1] * 0.012 * sector.automation_susceptibility
                    )

                new_adopt = min(
                    sector.automation_susceptibility,
                    adopt[t, i] + effective_rate,
                )
                adopt[t + 1, i] = new_adopt

                wc = sector.white_collar_pct
                target = sector.employment_millions * (1 - new_adopt * wc)

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
                    surplus_ratio * 0.08 + new_adopt * 0.004,
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

            retrain_boost = min(0.10, p.retraining_investment / 100.0)
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
            # 6. AGGREGATE EMPLOYMENT & INCOME
            # ============================================================
            total_emp[t + 1] = emp[t + 1].sum()
            unemp_rate[t + 1] = max(0.02, 1 - total_emp[t + 1] / labor_force)

            # Total labor compensation with multiplier
            wage_inc = sum(
                emp[t + 1, i] * wage[t + 1, i] for i in range(ns)
            )
            labor_inc[t + 1] = wage_inc * COMPENSATION_MULTIPLIER

            # ============================================================
            # 7. CONSUMER SPENDING
            # ============================================================
            unemp_excess = max(0, unemp_rate[t + 1] - 0.04)

            sr = p.base_savings_rate + unemp_excess * p.precautionary_savings_sensitivity
            sr = min(0.20, sr)
            sav_rate[t + 1] = sr

            confidence = max(0.50, 1.0 - unemp_excess * p.confidence_sensitivity)

            ubi_income = p.ubi_monthly_per_person * 12.0 * 260.0 / 1e6

            gdp_ratio = gdp_arr[t] / gdp_0
            adj_non_labor = non_labor_income * (0.5 + 0.5 * max(0.4, gdp_ratio))

            total_consumer_income = labor_inc[t + 1] + adj_non_labor + ubi_income
            cons_spend[t + 1] = total_consumer_income * (1 - sr) * confidence

            interm_loss = (
                agent_adopt[t + 1] * p.intermediation_gdp_fraction * gdp_arr[t]
            )

            # ============================================================
            # 8. GDP
            # ============================================================
            labor_savings = max(0, labor_inc[0] - labor_inc[t + 1])
            ai_inv = labor_savings * 0.25
            ai_invest[t + 1] = ai_inv

            other_inv = invest_0 * (0.7 + 0.3 * confidence)

            auto_stab = unemp_excess * gdp_0 * 0.02
            gov_spend = gov_spend_0 + auto_stab + ubi_income

            # "Ghost GDP": AI maintains output even as workers are cut.
            # This output shows up in national accounts (corporate profits)
            # but doesn't circulate through households as income.
            # It's what the report calls the disconnect between productivity
            # and the real consumer economy.
            ai_output = labor_savings * 0.45

            gdp_arr[t + 1] = (
                cons_spend[t + 1]
                + ai_inv
                + other_inv
                + gov_spend
                + net_exports_0
                + ai_output
                - interm_loss
            )
            gdp_arr[t + 1] = max(gdp_arr[t + 1], gdp_0 * 0.4)

            if gdp_arr[t] > 0:
                gdp_growth[t + 1] = (gdp_arr[t + 1] / gdp_arr[t]) ** 4 - 1
            else:
                gdp_growth[t + 1] = 0

            # ============================================================
            # 9. FINANCIAL INDICATORS
            # ============================================================
            labor_share[t + 1] = labor_inc[t + 1] / max(gdp_arr[t + 1], 1)

            # Mortgage delinquency
            inc_decline = max(
                0, (labor_inc[0] - labor_inc[t + 1]) / labor_inc[0]
            )
            md = 0.015 + inc_decline * p.mortgage_stress_sensitivity
            md += unemp_excess * 0.12
            mort_delinq[t + 1] = min(0.15, max(0.01, md))

            # Home prices
            delinq_pressure = max(0, mort_delinq[t + 1] - 0.02)
            inc_factor = labor_inc[t + 1] / labor_inc[0]
            qtr_price_chg = (
                (inc_factor - 1) * p.home_price_income_sensitivity * 0.15
                - delinq_pressure * 0.8
            )
            qtr_price_chg = max(-0.06, min(0.03, qtr_price_chg))
            home_px[t + 1] = max(40, home_px[t] * (1 + qtr_price_chg))

            # S&P 500 — models initial euphoria then crash
            # Corporate profits = GDP - labor compensation
            # Initially: AI cuts labor costs, GDP maintained by Ghost GDP → profits soar
            # Later: Demand destruction hits GDP → profits eventually fall
            corp_profits = gdp_arr[t + 1] - labor_inc[t + 1]
            init_profits = gdp_0 - labor_inc[0]
            earnings_ratio = max(0.3, corp_profits / max(init_profits, 1))

            # AI euphoria: builds as capability grows, peaked by confidence in AI narrative
            # Fades as macro reality sets in (unemployment, financial stress)
            quarters_in = t + 1
            ai_euphoria = min(1.30, 1.0 + 0.04 * min(quarters_in, 8))

            # Macro risk premium: rises with unemployment, mortgage stress, etc.
            stress = (
                unemp_excess * 2.0
                + fin_stress
                + max(0, mort_delinq[t + 1] - 0.02) * 6
            )
            # Net sentiment: euphoria vs fear
            sentiment = max(0.35, ai_euphoria - stress * p.equity_risk_premium_sensitivity)

            sp500[t + 1] = 6000 * earnings_ratio * sentiment
            sp500[t + 1] = max(1500, min(12000, sp500[t + 1]))

            # Drawdown from peak
            peak = np.max(sp500[: t + 2])
            sp500_dd[t + 1] = (sp500[t + 1] - peak) / peak if peak > 0 else 0

            # Federal deficit
            corp_profits = gdp_arr[t + 1] - labor_inc[t + 1]
            income_tax = labor_inc[t + 1] * 0.14
            corp_tax = max(0, corp_profits) * 0.08
            compute_tax_rev = ai_inv * p.compute_tax_rate
            total_receipts = income_tax + corp_tax + compute_tax_rev
            deficit = gov_spend - total_receipts
            deficit_pct[t + 1] = -deficit / max(gdp_arr[t + 1], 1)

            # Gini
            g = 0.49 + inc_decline * 0.8 + unemp_excess * 1.0
            gini[t + 1] = min(0.70, max(0.40, g))

            # ============================================================
            # 10. MILESTONE EVENTS
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
            milestones=milestones,
        )
