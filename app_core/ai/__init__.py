# =============================================================================
# app_core/ai/__init__.py
# AI Module for HealthForecast AI
# =============================================================================
"""
AI-powered recommendation and insight generation module.

This module provides:
- Rule-based insights (Option A): Fast, free, no API required
- LLM-powered insights (Option B): Natural language, requires API key

Usage:
    from app_core.ai import StaffRecommendationEngine, SupplyRecommendationEngine

    # For Staff Planner
    engine = StaffRecommendationEngine(mode="rule_based")  # or "llm"
    insights = engine.generate_insights(before_stats, after_stats, optimization_results)

    # For Supply Planner
    engine = SupplyRecommendationEngine(mode="rule_based")
    insights = engine.generate_insights(before_stats, after_stats, optimization_results)
"""

from .recommendation_engine import (
    StaffRecommendationEngine,
    SupplyRecommendationEngine,
    InsightMode,
    APIProvider,
    Insight,
    Recommendation,
)

__all__ = [
    "StaffRecommendationEngine",
    "SupplyRecommendationEngine",
    "InsightMode",
    "APIProvider",
    "Insight",
    "Recommendation",
]
