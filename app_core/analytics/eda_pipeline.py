# eda_pipeline.py — EDA foundation (merged-only + options UI ready)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple

import pandas as pd
# Add these imports at the top
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


# ...existing code...

# ---------------- Config & Results ----------------

@dataclass
class EDAConfig:
    """
    Configuration for the EDA pipeline.
    NOTE: We are locked to the 'merged' dataset by design for this phase.
    """
    dataset_source: str = "merged"     # fixed
    sample_rows: Optional[int] = 1000  # reserved for later (no-op for now)

    # --- Feature options (UI only; no computation yet) ---
    show_avg_by_dow_table: bool = True        # Option 1
    show_avg_by_dow_pie: bool = True          # Option 2
    pie_color_rule: str = "busiest_red_least_green"  # fixed rule by spec
    pie_legend_side: str = "right"            # by spec

    # Keep these as future/general toggles if needed
    include_overview: bool = True
    include_quality_checks: bool = False
    include_types_inference: bool = False
    include_date_checks: bool = False


@dataclass
class EDAResult:
    """Container for pipeline outputs (placeholders only, no data yet)."""
    status: str
    messages: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)


PipelineStep = Callable[[Optional[pd.DataFrame], EDAConfig, EDAResult], Tuple[Optional[pd.DataFrame], EDAResult]]

# ---------------- Step placeholders ----------------

def step_overview(df: Optional[pd.DataFrame], cfg: EDAConfig, res: EDAResult):
    res.messages.append("📌 Overview step registered (no data executed).")
    return df, res

def step_avg_by_dow_table(df: Optional[pd.DataFrame], cfg: EDAConfig, res: EDAResult):
    res.messages.append("🧮 Avg arrivals by Day-of-Week table planned (will add 'Total average' row).")
    return df, res

def step_avg_by_dow_pie(df: Optional[pd.DataFrame], cfg: EDAConfig, res: EDAResult):
    res.messages.append(
        f"🥧 Pie chart planned (legend: {cfg.pie_legend_side}; colors: {cfg.pie_color_rule})."
    )
    return df, res

# ---------------- Orchestrator ----------------

class EDAPipeline:
    """
    Lightweight EDA pipeline. Right now it only wires steps and returns a plan/log.
    Later we’ll attach real computations on the merged dataset.
    """
    def __init__(self, config: Optional[EDAConfig] = None):
        self.config = config or EDAConfig()
        self.steps: List[PipelineStep] = []
        self._register_steps()

    def _register_steps(self):
        # Always include a minimal overview line (log-level only for now)
        if self.config.include_overview:
            self.steps.append(step_overview)
        # Register the specific requested features (UI-driven)
        if self.config.show_avg_by_dow_table:
            self.steps.append(step_avg_by_dow_table)
        if self.config.show_avg_by_dow_pie:
            self.steps.append(step_avg_by_dow_pie)

    def planned_steps(self) -> List[str]:
        names = []
        for fn in self.steps:
            if fn is step_overview: names.append("Overview (log only)")
            elif fn is step_avg_by_dow_table: names.append("Avg arrivals by DOW — table (+ total row)")
            elif fn is step_avg_by_dow_pie: names.append("Avg arrivals by DOW — pie chart")
            else: names.append(fn.__name__)
        return names

    def dry_run(self) -> EDAResult:
        res = EDAResult(status="DRY_RUN")
        res.messages.append("✅ EDA initialized for merged dataset (dry run; no data needed).")
        res.messages.append(f"• Planned steps: {', '.join(self.planned_steps()) or 'None'}")
        # Echo key UI choices
        res.messages.append(f"• Pie legend: {self.config.pie_legend_side}")
        res.messages.append(f"• Pie color rule: {self.config.pie_color_rule}")
        return res

    def run(self, df: Optional[pd.DataFrame]) -> EDAResult:
        # For now, behave like dry_run; real data work arrives next step.
        res = self.dry_run()
        res.messages.append("ℹ️ Execution deferred (UI-only phase).")
        return res

