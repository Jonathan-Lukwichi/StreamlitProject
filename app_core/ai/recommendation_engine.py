# =============================================================================
# app_core/ai/recommendation_engine.py
# AI Recommendation Engine for HealthForecast AI
# Supports both Rule-Based (Option A) and LLM-Powered (Option B) insights
# =============================================================================
"""
AI Recommendation Engine with dual-mode support:
- Option A (rule_based): Fast, deterministic, no API required
- Option B (llm): Natural language generation using OpenAI/Anthropic API

The engine analyzes optimization results and generates:
1. Smart KPI Cards with contextual styling
2. Key Insights explaining what happened and why
3. Actionable Recommendations with priorities
4. Executive Summary for decision-makers
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class InsightMode(Enum):
    """Available insight generation modes."""
    RULE_BASED = "rule_based"  # Option A: Fast, free, deterministic
    LLM = "llm"                 # Option B: Natural language, requires API


class InsightType(Enum):
    """Types of insights that can be generated."""
    COST_INCREASE = "cost_increase"
    COST_DECREASE = "cost_decrease"
    STAFF_SHORTAGE = "staff_shortage"
    STAFF_SURPLUS = "staff_surplus"
    OVERTIME_ELIMINATED = "overtime_eliminated"
    UTILIZATION_IMPROVED = "utilization_improved"
    UTILIZATION_CONCERN = "utilization_concern"
    STOCKOUT_ELIMINATED = "stockout_eliminated"
    SERVICE_LEVEL_IMPROVED = "service_level_improved"
    GENERAL = "general"


class Priority(Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Insight:
    """A single insight with type, message, and styling."""
    type: InsightType
    title: str
    message: str
    icon: str = ""
    color: str = "blue"  # blue, green, red, yellow, purple
    details: List[str] = field(default_factory=list)


@dataclass
class Recommendation:
    """A single actionable recommendation."""
    priority: Priority
    action: str
    impact: str
    details: str = ""
    status: str = "pending"  # pending, in_progress, completed


@dataclass
class KPICard:
    """A KPI card for display."""
    label: str
    value: str
    delta: str = ""
    delta_type: str = "neutral"  # positive, negative, neutral
    color: str = "blue"  # blue, green, red, yellow, purple
    icon: str = ""
    tooltip: str = ""


@dataclass
class InsightResult:
    """Complete result from insight generation."""
    mode: InsightMode
    kpi_cards: List[KPICard]
    key_insight: Insight
    recommendations: List[Recommendation]
    summary_table: Dict[str, Dict[str, Any]]
    executive_summary: str
    raw_llm_response: Optional[str] = None  # Only for LLM mode


# =============================================================================
# BASE ENGINE
# =============================================================================

class BaseRecommendationEngine:
    """Base class for recommendation engines."""

    def __init__(self, mode: str = "rule_based", api_key: Optional[str] = None):
        """
        Initialize the recommendation engine.

        Args:
            mode: "rule_based" (Option A) or "llm" (Option B)
            api_key: API key for LLM mode (OpenAI or Anthropic)
        """
        self.mode = InsightMode(mode)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    def _format_currency(self, value: float) -> str:
        """Format a number as currency."""
        if abs(value) >= 1_000_000:
            return f"${value/1_000_000:,.1f}M"
        elif abs(value) >= 1_000:
            return f"${value/1_000:,.1f}K"
        else:
            return f"${value:,.0f}"

    def _format_percent(self, value: float) -> str:
        """Format a number as percentage."""
        return f"{value:.1f}%"

    def _get_priority_from_impact(self, impact_pct: float) -> Priority:
        """Determine priority based on impact percentage."""
        if abs(impact_pct) > 20:
            return Priority.CRITICAL
        elif abs(impact_pct) > 10:
            return Priority.HIGH
        elif abs(impact_pct) > 5:
            return Priority.MEDIUM
        else:
            return Priority.LOW


# =============================================================================
# STAFF RECOMMENDATION ENGINE
# =============================================================================

class StaffRecommendationEngine(BaseRecommendationEngine):
    """
    Recommendation engine for Staff Planner.

    Analyzes staffing optimization results and generates insights about:
    - Cost changes (increases or savings)
    - Staff level adjustments (doctors, nurses, support)
    - Utilization improvements
    - Overtime and shortage elimination
    """

    def generate_insights(
        self,
        before_stats: Dict[str, Any],
        after_stats: Dict[str, Any],
        optimization_results: Dict[str, Any],
        cost_params: Optional[Dict[str, Any]] = None
    ) -> InsightResult:
        """
        Generate insights from optimization results.

        Args:
            before_stats: Current/historical staff statistics
            after_stats: Optimized staff statistics
            optimization_results: Full optimization results dict
            cost_params: Cost parameters used in optimization

        Returns:
            InsightResult with KPIs, insights, recommendations, and summary
        """
        if self.mode == InsightMode.LLM:
            return self._generate_llm_insights(before_stats, after_stats, optimization_results, cost_params)
        else:
            return self._generate_rule_based_insights(before_stats, after_stats, optimization_results, cost_params)

    def _generate_rule_based_insights(
        self,
        before_stats: Dict[str, Any],
        after_stats: Dict[str, Any],
        opt: Dict[str, Any],
        cost_params: Optional[Dict[str, Any]] = None
    ) -> InsightResult:
        """Generate insights using rule-based logic (Option A)."""

        # Extract key metrics
        cost_before = opt.get("current_weekly_cost", 0)
        cost_after = opt.get("optimized_weekly_cost", 0)
        cost_diff = cost_after - cost_before
        cost_pct = (cost_diff / cost_before * 100) if cost_before > 0 else 0

        doctors_before = before_stats.get("avg_doctors", 0)
        doctors_after = opt.get("avg_opt_doctors", 0)
        doctors_diff = doctors_after - doctors_before

        nurses_before = before_stats.get("avg_nurses", 0)
        nurses_after = opt.get("avg_opt_nurses", 0)
        nurses_diff = nurses_after - nurses_before

        support_before = before_stats.get("avg_support", 0)
        support_after = opt.get("avg_opt_support", 0)
        support_diff = support_after - support_before

        util_before = before_stats.get("avg_utilization", 0)
        util_after = 95  # Target utilization
        util_diff = util_after - util_before

        overtime_before = before_stats.get("total_overtime", 0)
        shortage_before = before_stats.get("shortage_days", 0)

        # Build KPI Cards
        kpi_cards = []

        # Cost Change Card
        if cost_diff > 0:
            kpi_cards.append(KPICard(
                label="Weekly Investment Needed",
                value=self._format_currency(abs(cost_diff)),
                delta=f"+{abs(cost_pct):.1f}% to staff properly",
                delta_type="negative",
                color="yellow",
                icon="",
                tooltip="Cost increases because proper staffing requires more resources"
            ))
        else:
            kpi_cards.append(KPICard(
                label="Weekly Savings",
                value=self._format_currency(abs(cost_diff)),
                delta=f"-{abs(cost_pct):.1f}% reduction",
                delta_type="positive",
                color="green",
                icon="",
                tooltip="Cost savings from optimized staffing"
            ))

        # Doctor Change Card
        if doctors_diff > 0:
            kpi_cards.append(KPICard(
                label="Doctors Needed",
                value=f"+{doctors_diff:.0f}",
                delta=f"{doctors_before:.0f} -> {doctors_after:.0f}",
                delta_type="negative",
                color="red",
                icon="",
                tooltip="More doctors needed to meet patient demand"
            ))
        elif doctors_diff < 0:
            kpi_cards.append(KPICard(
                label="Doctors Surplus",
                value=f"{abs(doctors_diff):.0f}",
                delta=f"{doctors_before:.0f} -> {doctors_after:.0f}",
                delta_type="positive",
                color="green",
                icon="",
                tooltip="Can reduce doctor count while maintaining care"
            ))

        # Overtime Card
        if overtime_before > 0:
            kpi_cards.append(KPICard(
                label="Overtime Eliminated",
                value=f"{overtime_before:,.0f}h",
                delta="0h after optimization",
                delta_type="positive",
                color="green",
                icon="",
                tooltip="All overtime hours eliminated with proper staffing"
            ))

        # Shortage Card
        if shortage_before > 0:
            kpi_cards.append(KPICard(
                label="Shortages Eliminated",
                value=f"{shortage_before} days",
                delta="0 days after optimization",
                delta_type="positive",
                color="green",
                icon="",
                tooltip="All shortage days eliminated"
            ))

        # Determine Key Insight
        if cost_diff > 0:
            # Cost is INCREASING - need to explain why
            insight_type = InsightType.COST_INCREASE
            insight_title = "Investment Required for Proper Staffing"

            details = []
            if doctors_diff > 0:
                details.append(f"Hire {doctors_diff:.0f} more doctors (+{doctors_diff/doctors_before*100:.0f}%)")
            if nurses_diff < 0:
                details.append(f"Reduce {abs(nurses_diff):.0f} nurses (-{abs(nurses_diff)/nurses_before*100:.0f}%)")
            if support_diff != 0:
                if support_diff > 0:
                    details.append(f"Add {support_diff:.0f} support staff")
                else:
                    details.append(f"Reduce {abs(support_diff):.0f} support staff")

            if util_before > 100:
                insight_message = (
                    f"Your hospital was understaffed with {util_before:.1f}% utilization (staff overworked). "
                    f"The optimization shows the TRUE cost of proper staffing: "
                    f"{self._format_currency(abs(cost_diff))}/week more to achieve healthy {util_after}% utilization."
                )
            else:
                insight_message = (
                    f"To meet forecast patient demand, staffing levels need adjustment. "
                    f"This requires an additional {self._format_currency(abs(cost_diff))}/week investment."
                )

            key_insight = Insight(
                type=insight_type,
                title=insight_title,
                message=insight_message,
                icon="",
                color="yellow",
                details=details
            )
        else:
            # Cost is DECREASING - savings
            insight_type = InsightType.COST_DECREASE
            insight_title = "Optimization Achieves Cost Savings"
            insight_message = (
                f"By aligning staff levels with forecast demand, you can save "
                f"{self._format_currency(abs(cost_diff))}/week ({abs(cost_pct):.1f}% reduction) "
                f"while maintaining optimal {util_after}% utilization."
            )
            key_insight = Insight(
                type=insight_type,
                title=insight_title,
                message=insight_message,
                icon="",
                color="green",
                details=[]
            )

        # Build Recommendations
        recommendations = []

        if doctors_diff > 0:
            recommendations.append(Recommendation(
                priority=Priority.HIGH if doctors_diff > 3 else Priority.MEDIUM,
                action=f"Hire {doctors_diff:.0f} additional doctors",
                impact=f"Eliminates understaffing and {overtime_before:,.0f}h overtime",
                details="Critical for patient care quality and staff wellbeing"
            ))

        if nurses_diff < 0:
            recommendations.append(Recommendation(
                priority=Priority.MEDIUM,
                action=f"Reduce nursing staff by {abs(nurses_diff):.0f}",
                impact=f"Saves approximately {self._format_currency(abs(nurses_diff) * (cost_params.get('nurse_hourly', 50) if cost_params else 50) * 8 * 7)}/week",
                details="Through attrition or redeployment to other departments"
            ))

        if overtime_before > 0:
            recommendations.append(Recommendation(
                priority=Priority.HIGH,
                action="Eliminate all overtime hours",
                impact=f"Removes {overtime_before:,.0f}h of overtime, improves staff wellbeing",
                details="Proper staffing levels prevent the need for overtime"
            ))

        if shortage_before > 0:
            recommendations.append(Recommendation(
                priority=Priority.CRITICAL,
                action="Eliminate staff shortage days",
                impact=f"Zero shortage days = better patient care",
                details=f"Currently {shortage_before} days with staff shortages"
            ))

        # Summary Table
        summary_table = {
            "Weekly Cost": {
                "before": self._format_currency(cost_before),
                "after": self._format_currency(cost_after),
                "change": f"{'+' if cost_diff > 0 else ''}{self._format_currency(cost_diff)} ({cost_pct:+.1f}%)"
            },
            "Annual Cost": {
                "before": self._format_currency(cost_before * 52),
                "after": self._format_currency(cost_after * 52),
                "change": f"{'+' if cost_diff > 0 else ''}{self._format_currency(cost_diff * 52)}"
            },
            "Doctors/Day": {
                "before": f"{doctors_before:.0f}",
                "after": f"{doctors_after:.0f}",
                "change": f"{doctors_diff:+.0f}"
            },
            "Nurses/Day": {
                "before": f"{nurses_before:.0f}",
                "after": f"{nurses_after:.0f}",
                "change": f"{nurses_diff:+.0f}"
            },
            "Utilization": {
                "before": f"{util_before:.1f}%",
                "after": f"{util_after}%",
                "change": f"{util_diff:+.1f}%"
            },
            "Overtime": {
                "before": f"{overtime_before:,.0f}h",
                "after": "0h",
                "change": "Eliminated"
            },
            "Shortages": {
                "before": f"{shortage_before} days",
                "after": "0 days",
                "change": "Eliminated"
            }
        }

        # Executive Summary
        if cost_diff > 0:
            executive_summary = (
                f"**Investment Required:** Your hospital needs {self._format_currency(abs(cost_diff))}/week more "
                f"to staff properly.\n\n"
                f"**Why?** Historical staffing ({doctors_before:.0f} doctors, {nurses_before:.0f} nurses) was insufficient, "
                f"leading to {util_before:.1f}% utilization and {overtime_before:,.0f}h overtime.\n\n"
                f"**Benefits:** Proper staffing ({doctors_after:.0f} doctors, {nurses_after:.0f} nurses) eliminates "
                f"overtime, shortages, and improves patient care quality."
            )
        else:
            executive_summary = (
                f"**Savings Available:** Optimization saves {self._format_currency(abs(cost_diff))}/week "
                f"({abs(cost_pct):.1f}% reduction).\n\n"
                f"**How?** By right-sizing staff to match forecast demand while maintaining {util_after}% utilization.\n\n"
                f"**Annual Impact:** {self._format_currency(abs(cost_diff) * 52)} in savings."
            )

        return InsightResult(
            mode=self.mode,
            kpi_cards=kpi_cards,
            key_insight=key_insight,
            recommendations=recommendations,
            summary_table=summary_table,
            executive_summary=executive_summary
        )

    def _generate_llm_insights(
        self,
        before_stats: Dict[str, Any],
        after_stats: Dict[str, Any],
        opt: Dict[str, Any],
        cost_params: Optional[Dict[str, Any]] = None
    ) -> InsightResult:
        """Generate insights using LLM (Option B)."""

        # First generate rule-based as fallback
        rule_based = self._generate_rule_based_insights(before_stats, after_stats, opt, cost_params)

        if not self.api_key:
            # No API key, return rule-based with note
            rule_based.executive_summary += "\n\n*Note: LLM mode selected but no API key configured. Using rule-based insights.*"
            return rule_based

        try:
            # Try OpenAI first
            import openai
            client = openai.OpenAI(api_key=self.api_key)

            # Build context for LLM
            context = self._build_llm_context(before_stats, opt, cost_params)

            # First API call: Get enhanced executive summary and key insight
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use smaller model for cost efficiency
                messages=[
                    {"role": "system", "content": """You are a healthcare operations analyst expert.
                    Analyze staffing optimization results and provide sophisticated, actionable insights.
                    Be direct and focus on what matters to hospital administrators.
                    Always explain WHY costs are changing, not just what changed.
                    Use professional healthcare industry terminology.
                    Focus on patient care quality, staff wellbeing, and operational efficiency."""},
                    {"role": "user", "content": context}
                ],
                max_tokens=700,
                temperature=0.3
            )

            llm_summary = response.choices[0].message.content

            # Second API call: Get an enhanced key insight message
            key_insight_prompt = self._build_key_insight_prompt(before_stats, opt)
            key_insight_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are a healthcare operations analyst.
                    Write a concise but impactful 2-3 sentence explanation of the key finding.
                    Be specific about numbers and percentages. Focus on the business impact.
                    Do NOT use markdown formatting. Just plain text."""},
                    {"role": "user", "content": key_insight_prompt}
                ],
                max_tokens=150,
                temperature=0.2
            )

            enhanced_key_insight = key_insight_response.choices[0].message.content

            # Update the key insight with LLM-generated message
            rule_based.key_insight.message = enhanced_key_insight

            # Use rule-based KPIs but LLM executive summary
            rule_based.executive_summary = llm_summary
            rule_based.raw_llm_response = llm_summary
            rule_based.mode = InsightMode.LLM

            return rule_based

        except ImportError:
            rule_based.executive_summary += "\n\n*Note: OpenAI package not installed. Run `pip install openai` to enable LLM mode.*"
            return rule_based
        except Exception as e:
            rule_based.executive_summary += f"\n\n*Note: LLM error: {str(e)}. Using rule-based insights.*"
            return rule_based

    def _build_key_insight_prompt(
        self,
        before_stats: Dict[str, Any],
        opt: Dict[str, Any]
    ) -> str:
        """Build prompt for key insight generation."""
        cost_before = opt.get("current_weekly_cost", 0)
        cost_after = opt.get("optimized_weekly_cost", 0)
        cost_diff = cost_after - cost_before
        cost_pct = (cost_diff / cost_before * 100) if cost_before > 0 else 0

        doctors_before = before_stats.get("avg_doctors", 0)
        doctors_after = opt.get("avg_opt_doctors", 0)
        util_before = before_stats.get("avg_utilization", 0)

        return f"""
Based on this staffing optimization:
- Cost change: ${cost_diff:,.0f}/week ({cost_pct:+.1f}%)
- Doctor change: {doctors_before:.0f} → {doctors_after:.0f}
- Current utilization: {util_before:.1f}%
- Target utilization: 95%
- Current overtime: {before_stats.get('total_overtime', 0):,.0f} hours
- Current shortage days: {before_stats.get('shortage_days', 0)}

Write a 2-3 sentence key finding that explains the IMPACT of this optimization.
Focus on patient care and operational efficiency. Be specific with numbers.
"""

    def _build_llm_context(
        self,
        before_stats: Dict[str, Any],
        opt: Dict[str, Any],
        cost_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build context string for LLM."""
        return f"""
Analyze this hospital staffing optimization:

BEFORE (Current Situation):
- Doctors per day: {before_stats.get('avg_doctors', 0):.0f}
- Nurses per day: {before_stats.get('avg_nurses', 0):.0f}
- Support staff per day: {before_stats.get('avg_support', 0):.0f}
- Staff utilization: {before_stats.get('avg_utilization', 0):.1f}%
- Total overtime hours: {before_stats.get('total_overtime', 0):,.0f}
- Shortage days: {before_stats.get('shortage_days', 0)}
- Weekly cost: ${opt.get('current_weekly_cost', 0):,.0f}

AFTER (Optimized):
- Doctors per day: {opt.get('avg_opt_doctors', 0):.0f}
- Nurses per day: {opt.get('avg_opt_nurses', 0):.0f}
- Support staff per day: {opt.get('avg_opt_support', 0):.0f}
- Target utilization: 95%
- Expected overtime: 0 hours
- Expected shortages: 0 days
- Weekly cost: ${opt.get('optimized_weekly_cost', 0):,.0f}

CHANGES:
- Weekly cost difference: ${opt.get('optimized_weekly_cost', 0) - opt.get('current_weekly_cost', 0):,.0f}
- Doctor change: {opt.get('avg_opt_doctors', 0) - before_stats.get('avg_doctors', 0):+.0f}
- Nurse change: {opt.get('avg_opt_nurses', 0) - before_stats.get('avg_nurses', 0):+.0f}

Provide a 2-3 paragraph executive summary explaining:
1. What the key finding is (is cost increasing or decreasing?)
2. WHY this is happening (understaffing, overstaffing, etc.)
3. What are the main benefits of implementing this optimization?

Be direct and business-focused. Use bullet points for key actions.
"""


# =============================================================================
# SUPPLY RECOMMENDATION ENGINE
# =============================================================================

class SupplyRecommendationEngine(BaseRecommendationEngine):
    """
    Recommendation engine for Supply Planner.

    Analyzes inventory optimization results and generates insights about:
    - Cost changes (savings from optimized ordering)
    - Service level improvements
    - Stockout risk elimination
    - Optimal order quantities
    """

    def generate_insights(
        self,
        before_stats: Dict[str, Any],
        after_stats: Dict[str, Any],
        optimization_results: Dict[str, Any],
        cost_params: Optional[Dict[str, Any]] = None
    ) -> InsightResult:
        """
        Generate insights from optimization results.

        Args:
            before_stats: Current/historical inventory statistics
            after_stats: Optimized inventory parameters
            optimization_results: Full optimization results dict
            cost_params: Cost parameters used in optimization

        Returns:
            InsightResult with KPIs, insights, recommendations, and summary
        """
        if self.mode == InsightMode.LLM:
            return self._generate_llm_insights(before_stats, after_stats, optimization_results, cost_params)
        else:
            return self._generate_rule_based_insights(before_stats, after_stats, optimization_results, cost_params)

    def _generate_rule_based_insights(
        self,
        before_stats: Dict[str, Any],
        after_stats: Dict[str, Any],
        opt: Dict[str, Any],
        cost_params: Optional[Dict[str, Any]] = None
    ) -> InsightResult:
        """Generate insights using rule-based logic (Option A)."""

        # Extract key metrics
        cost_before = opt.get("current_weekly_cost", 0)
        cost_after = opt.get("optimized_weekly_cost", 0)
        cost_diff = cost_after - cost_before
        cost_pct = (cost_diff / cost_before * 100) if cost_before > 0 else 0

        stockout_risk_before = before_stats.get("avg_stockout_risk", 0)
        service_level_before = 100 - stockout_risk_before
        service_level_after = 100  # Target

        restock_events = before_stats.get("restock_events", 0)

        # Build KPI Cards
        kpi_cards = []

        # Cost Change Card
        if cost_diff > 0:
            kpi_cards.append(KPICard(
                label="Weekly Investment Needed",
                value=self._format_currency(abs(cost_diff)),
                delta=f"+{abs(cost_pct):.1f}% for optimal inventory",
                delta_type="negative",
                color="yellow",
                icon=""
            ))
        else:
            kpi_cards.append(KPICard(
                label="Weekly Savings",
                value=self._format_currency(abs(cost_diff)),
                delta=f"-{abs(cost_pct):.1f}% reduction",
                delta_type="positive",
                color="green",
                icon=""
            ))

        # Service Level Card
        service_improvement = service_level_after - service_level_before
        kpi_cards.append(KPICard(
            label="Service Level",
            value=f"{service_level_after}%",
            delta=f"+{service_improvement:.1f}% improvement",
            delta_type="positive",
            color="green",
            icon=""
        ))

        # Stockout Risk Card
        if stockout_risk_before > 0:
            kpi_cards.append(KPICard(
                label="Stockout Risk Eliminated",
                value=f"{stockout_risk_before:.1f}%",
                delta="0% after optimization",
                delta_type="positive",
                color="green",
                icon=""
            ))

        # Restock Efficiency Card
        if restock_events > 0:
            kpi_cards.append(KPICard(
                label="Restock Events",
                value=f"{restock_events}",
                delta="Optimized ordering",
                delta_type="positive",
                color="blue",
                icon=""
            ))

        # Determine Key Insight
        if cost_diff > 0:
            insight_type = InsightType.COST_INCREASE
            insight_title = "Investment Required for Optimal Inventory"
            insight_message = (
                f"To achieve 100% service level and eliminate stockout risk, "
                f"an additional {self._format_currency(abs(cost_diff))}/week investment is needed. "
                f"This ensures supplies are always available when needed."
            )
            insight_color = "yellow"
        else:
            insight_type = InsightType.COST_DECREASE
            insight_title = "Optimization Achieves Cost Savings"
            insight_message = (
                f"By aligning inventory orders with forecast demand, you can save "
                f"{self._format_currency(abs(cost_diff))}/week ({abs(cost_pct):.1f}% reduction) "
                f"while improving service level to 100%."
            )
            insight_color = "green"

        key_insight = Insight(
            type=insight_type,
            title=insight_title,
            message=insight_message,
            icon="" if cost_diff > 0 else "",
            color=insight_color,
            details=[]
        )

        # Build Recommendations
        recommendations = []

        if stockout_risk_before > 5:
            recommendations.append(Recommendation(
                priority=Priority.HIGH,
                action="Increase safety stock levels",
                impact=f"Reduces stockout risk from {stockout_risk_before:.1f}% to 0%",
                details="Use forecast-driven reorder points"
            ))

        if service_level_before < 98:
            recommendations.append(Recommendation(
                priority=Priority.MEDIUM,
                action="Implement forecast-based ordering",
                impact=f"Improves service level from {service_level_before:.1f}% to 100%",
                details="Align orders with patient demand forecasts"
            ))

        if restock_events > 50:
            recommendations.append(Recommendation(
                priority=Priority.LOW,
                action="Consolidate restock orders",
                impact="Reduces ordering costs and handling time",
                details="Use optimal order quantities from forecast"
            ))

        # Summary Table
        summary_table = {
            "Weekly Cost": {
                "before": self._format_currency(cost_before),
                "after": self._format_currency(cost_after),
                "change": f"{'+' if cost_diff > 0 else ''}{self._format_currency(cost_diff)} ({cost_pct:+.1f}%)"
            },
            "Annual Cost": {
                "before": self._format_currency(cost_before * 52),
                "after": self._format_currency(cost_after * 52),
                "change": f"{'+' if cost_diff > 0 else ''}{self._format_currency(cost_diff * 52)}"
            },
            "Service Level": {
                "before": f"{service_level_before:.1f}%",
                "after": f"{service_level_after}%",
                "change": f"+{service_improvement:.1f}%"
            },
            "Stockout Risk": {
                "before": f"{stockout_risk_before:.1f}%",
                "after": "0%",
                "change": "Eliminated"
            }
        }

        # Executive Summary
        if cost_diff > 0:
            executive_summary = (
                f"**Investment Required:** {self._format_currency(abs(cost_diff))}/week more "
                f"to maintain optimal inventory levels.\n\n"
                f"**Why?** Current inventory levels carry {stockout_risk_before:.1f}% stockout risk. "
                f"Proper safety stock requires additional investment.\n\n"
                f"**Benefits:** 100% service level, zero stockouts, reliable supply chain."
            )
        else:
            executive_summary = (
                f"**Savings Available:** {self._format_currency(abs(cost_diff))}/week "
                f"({abs(cost_pct):.1f}% reduction) through optimized ordering.\n\n"
                f"**How?** Forecast-driven ordering reduces excess inventory while maintaining service levels.\n\n"
                f"**Annual Impact:** {self._format_currency(abs(cost_diff) * 52)} in savings."
            )

        return InsightResult(
            mode=self.mode,
            kpi_cards=kpi_cards,
            key_insight=key_insight,
            recommendations=recommendations,
            summary_table=summary_table,
            executive_summary=executive_summary
        )

    def _generate_llm_insights(
        self,
        before_stats: Dict[str, Any],
        after_stats: Dict[str, Any],
        opt: Dict[str, Any],
        cost_params: Optional[Dict[str, Any]] = None
    ) -> InsightResult:
        """Generate insights using LLM (Option B)."""

        # First generate rule-based as fallback
        rule_based = self._generate_rule_based_insights(before_stats, after_stats, opt, cost_params)

        if not self.api_key:
            rule_based.executive_summary += "\n\n*Note: LLM mode selected but no API key configured. Using rule-based insights.*"
            return rule_based

        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)

            context = self._build_llm_context(before_stats, opt, cost_params)

            # First API call: Get enhanced executive summary
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are a healthcare supply chain analyst expert.
                    Analyze inventory optimization results and provide sophisticated, actionable insights.
                    Focus on service levels, stockout prevention, and cost efficiency.
                    Use professional supply chain terminology.
                    Emphasize patient care continuity and operational resilience."""},
                    {"role": "user", "content": context}
                ],
                max_tokens=700,
                temperature=0.3
            )

            llm_summary = response.choices[0].message.content

            # Second API call: Get an enhanced key insight message
            key_insight_prompt = self._build_key_insight_prompt(before_stats, opt)
            key_insight_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are a healthcare supply chain analyst.
                    Write a concise but impactful 2-3 sentence explanation of the key finding.
                    Be specific about numbers and percentages. Focus on patient care impact.
                    Do NOT use markdown formatting. Just plain text."""},
                    {"role": "user", "content": key_insight_prompt}
                ],
                max_tokens=150,
                temperature=0.2
            )

            enhanced_key_insight = key_insight_response.choices[0].message.content

            # Update the key insight with LLM-generated message
            rule_based.key_insight.message = enhanced_key_insight

            rule_based.executive_summary = llm_summary
            rule_based.raw_llm_response = llm_summary
            rule_based.mode = InsightMode.LLM

            return rule_based

        except ImportError:
            rule_based.executive_summary += "\n\n*Note: OpenAI package not installed. Run `pip install openai` to enable LLM mode.*"
            return rule_based
        except Exception as e:
            rule_based.executive_summary += f"\n\n*Note: LLM error: {str(e)}. Using rule-based insights.*"
            return rule_based

    def _build_key_insight_prompt(
        self,
        before_stats: Dict[str, Any],
        opt: Dict[str, Any]
    ) -> str:
        """Build prompt for key insight generation."""
        cost_before = opt.get("current_weekly_cost", 0)
        cost_after = opt.get("optimized_weekly_cost", 0)
        cost_diff = cost_after - cost_before
        cost_pct = (cost_diff / cost_before * 100) if cost_before > 0 else 0

        stockout_risk = before_stats.get("avg_stockout_risk", 0)
        service_level = 100 - stockout_risk

        return f"""
Based on this inventory optimization:
- Cost change: ${cost_diff:,.0f}/week ({cost_pct:+.1f}%)
- Current stockout risk: {stockout_risk:.1f}%
- Current service level: {service_level:.1f}%
- Target service level: 100%
- Restock events: {before_stats.get('restock_events', 0)}

Write a 2-3 sentence key finding that explains the IMPACT of this optimization.
Focus on patient care continuity and supply chain reliability. Be specific with numbers.
"""

    def _build_llm_context(
        self,
        before_stats: Dict[str, Any],
        opt: Dict[str, Any],
        cost_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build context string for LLM."""
        return f"""
Analyze this hospital inventory optimization:

BEFORE (Current Situation):
- Average stockout risk: {before_stats.get('avg_stockout_risk', 0):.1f}%
- Service level: {100 - before_stats.get('avg_stockout_risk', 0):.1f}%
- Restock events: {before_stats.get('restock_events', 0)}
- Weekly cost: ${opt.get('current_weekly_cost', 0):,.0f}

AFTER (Optimized):
- Target stockout risk: 0%
- Target service level: 100%
- Optimized ordering based on demand forecast
- Weekly cost: ${opt.get('optimized_weekly_cost', 0):,.0f}

CHANGES:
- Weekly cost difference: ${opt.get('optimized_weekly_cost', 0) - opt.get('current_weekly_cost', 0):,.0f}
- Service level improvement: +{100 - (100 - before_stats.get('avg_stockout_risk', 0)):.1f}%

Provide a 2-3 paragraph executive summary explaining:
1. What the key finding is (cost savings or investment needed?)
2. WHY this change is beneficial
3. Key actions to implement

Be direct and business-focused.
"""
