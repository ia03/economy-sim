"""
Configuration for the AI Economy Simulator.

Defines economic sectors, simulation parameters, and scenario presets
based on the dynamics described in the Citrini Research
"2028 Global Intelligence Crisis" report.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SectorConfig:
    """An economic sector with employment, wage, and automation characteristics."""

    name: str
    employment_millions: float
    avg_annual_wage_k: float  # $thousands/year
    automation_susceptibility: float  # 0-1, theoretical max automatable fraction
    white_collar_pct: float  # fraction of workforce that is white-collar
    is_intermediation: bool = False  # extra exposure to agentic commerce

    @property
    def labor_income_billions(self) -> float:
        return self.employment_millions * self.avg_annual_wage_k / 1000


# Eight sectors approximating the US economy (~158M total employment)
DEFAULT_SECTORS: List[SectorConfig] = [
    SectorConfig("Technology & Software", 5.0, 140.0, 0.90, 0.90),
    SectorConfig("Finance & Insurance", 7.0, 105.0, 0.75, 0.85, is_intermediation=True),
    SectorConfig("Professional Services", 23.0, 80.0, 0.80, 0.75),
    SectorConfig("Retail & Hospitality", 27.0, 32.0, 0.30, 0.25, is_intermediation=True),
    SectorConfig("Healthcare", 21.0, 58.0, 0.25, 0.40),
    SectorConfig("Manufacturing & Construction", 20.0, 55.0, 0.20, 0.25),
    SectorConfig("Government & Education", 25.0, 60.0, 0.25, 0.50),
    SectorConfig("Other Services", 30.0, 42.0, 0.35, 0.30, is_intermediation=True),
]


@dataclass
class SimulationParams:
    """All tunable parameters for the economy simulation."""

    # --- Simulation Horizon ---
    num_quarters: int = 24  # Q1 2025 to Q4 2030
    start_year: float = 2025.0

    # --- AI Dynamics ---
    ai_capability_quarterly_growth: float = 0.12  # 12%/quarter (~60%/year)
    ai_cost_quarterly_decline: float = 0.15  # 15% cost reduction/quarter
    ai_capability_ceiling: float = 100.0  # theoretical max (multiple of initial); physical/algorithmic limits

    # --- Per-Sector Automation Speed ---
    # 0.0 = no automation, 1.0 = maximum speed
    sector_automation_speeds: Dict[str, float] = field(default_factory=lambda: {
        "Technology & Software": 0.70,
        "Finance & Insurance": 0.50,
        "Professional Services": 0.60,
        "Retail & Hospitality": 0.30,
        "Healthcare": 0.20,
        "Manufacturing & Construction": 0.15,
        "Government & Education": 0.15,
        "Other Services": 0.25,
    })

    # --- Labor Market ---
    worker_redeployment_rate: float = 0.15  # fraction finding new work/quarter
    displaced_wage_penalty: float = 0.40  # wage cut on sector switch
    layoff_speed: float = 0.25  # fraction of redundant workers laid off/quarter
    wage_flexibility: float = 0.03  # max quarterly downward wage adjustment

    # --- Consumer Behavior ---
    base_savings_rate: float = 0.05
    precautionary_savings_sensitivity: float = 0.8
    confidence_sensitivity: float = 0.5

    # --- Agentic Commerce ---
    agent_adoption_midpoint_quarter: float = 10.0  # quarter when 50% adopt
    agent_adoption_steepness: float = 0.4
    intermediation_gdp_fraction: float = 0.06

    # --- Financial System ---
    mortgage_stress_sensitivity: float = 0.25
    home_price_income_sensitivity: float = 0.4
    equity_earnings_weight: float = 0.6
    equity_risk_premium_sensitivity: float = 0.4

    # --- Policy ---
    ubi_monthly_per_person: float = 0.0
    compute_tax_rate: float = 0.0
    retraining_investment: float = 0.0  # $B/year

    # --- AI Positive Channels ---
    ai_deflation_rate: float = 0.006  # max quarterly price decline from AI adoption
    new_job_creation_rate: float = 0.5  # 0-1: how fast AI economy creates new roles
    fed_response_speed: float = 0.25  # how aggressively the Fed adjusts rates

    # --- Feedback Loop Strength ---
    margin_pressure_ai_feedback: float = 0.3
    credit_tightening_feedback: float = 0.15


def _default_speeds():
    return SimulationParams().sector_automation_speeds


# Named scenario presets
SCENARIO_PRESETS: Dict[str, SimulationParams] = {
    "Citrini Baseline": SimulationParams(),
    "Optimistic (Slower AI)": SimulationParams(
        ai_capability_quarterly_growth=0.06,
        ai_cost_quarterly_decline=0.08,
        worker_redeployment_rate=0.25,
        displaced_wage_penalty=0.25,
        confidence_sensitivity=0.3,
    ),
    "Aggressive Policy Response": SimulationParams(
        ubi_monthly_per_person=1000,
        compute_tax_rate=0.10,
        retraining_investment=50,
    ),
    "Accelerated Disruption": SimulationParams(
        ai_capability_quarterly_growth=0.18,
        ai_cost_quarterly_decline=0.22,
        sector_automation_speeds={
            "Technology & Software": 0.90,
            "Finance & Insurance": 0.70,
            "Professional Services": 0.80,
            "Retail & Hospitality": 0.45,
            "Healthcare": 0.30,
            "Manufacturing & Construction": 0.25,
            "Government & Education": 0.20,
            "Other Services": 0.40,
        },
    ),
    "AI Prosperity": SimulationParams(
        ai_capability_quarterly_growth=0.10,
        ai_cost_quarterly_decline=0.12,
        worker_redeployment_rate=0.25,
        displaced_wage_penalty=0.25,
        ubi_monthly_per_person=1500,
        compute_tax_rate=0.10,
        retraining_investment=150,
        new_job_creation_rate=1.0,
        ai_deflation_rate=0.010,
        margin_pressure_ai_feedback=0.15,
    ),
    "Soft Landing": SimulationParams(
        ai_capability_quarterly_growth=0.08,
        ai_cost_quarterly_decline=0.10,
        worker_redeployment_rate=0.30,
        displaced_wage_penalty=0.20,
        ubi_monthly_per_person=500,
        compute_tax_rate=0.05,
        retraining_investment=30,
        margin_pressure_ai_feedback=0.15,
    ),
}


def quarter_labels(num_quarters: int, start_year: float = 2025.0) -> List[str]:
    """Generate quarter labels like 'Q1 2025', 'Q2 2025', etc."""
    labels = []
    for q in range(num_quarters + 1):
        year = int(start_year + q // 4)
        quarter = (q % 4) + 1
        labels.append(f"Q{quarter} {year}")
    return labels
