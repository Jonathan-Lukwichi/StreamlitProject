# =============================================================================
# app_core/ui/unified_dashboard/styles.py
# Premium CSS Styles for the Unified Dashboard
# =============================================================================
"""
Premium CSS styles with glassmorphism, animations, and responsive design.
"""

# =============================================================================
# ANIMATION KEYFRAMES
# =============================================================================

ANIMATION_KEYFRAMES = """
/* ========================================
   PREMIUM ANIMATION LIBRARY
   ======================================== */

/* Shimmer Loading Effect */
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* Fade In with Slide */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Fade In Scale */
@keyframes fadeInScale {
    from {
        opacity: 0;
        transform: scale(0.95);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

/* Pulse Effect for Alerts */
@keyframes pulse-alert {
    0%, 100% {
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4);
    }
    50% {
        box-shadow: 0 0 0 12px rgba(239, 68, 68, 0);
    }
}

/* Pulse Sync Indicator */
@keyframes pulse-sync {
    0%, 100% {
        box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.4);
    }
    50% {
        box-shadow: 0 0 0 8px rgba(34, 197, 94, 0);
    }
}

/* Gentle Glow */
@keyframes glow {
    0%, 100% {
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
    }
    50% {
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
    }
}

/* Bounce Icon */
@keyframes bounce {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-8px);
    }
}

/* Float Effect */
@keyframes float {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-5px);
    }
}

/* Counter Animation */
@keyframes countUp {
    from {
        opacity: 0;
        transform: scale(0.8);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

/* Slide In from Right */
@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}
"""

# =============================================================================
# NORTHSTAR HEADER CSS
# =============================================================================

NORTHSTAR_HEADER_CSS = """
/* ========================================
   NORTH STAR HEADER (Sticky Hero KPIs)
   ======================================== */

.northstar-strip {
    position: sticky;
    top: 0;
    z-index: 100;
    background: linear-gradient(135deg, rgba(11, 17, 32, 0.98), rgba(5, 8, 22, 0.95));
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(59, 130, 246, 0.2);
    padding: 1rem 1.5rem;
    margin: -1rem -1rem 1.5rem -1rem;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
}

.northstar-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1.5rem;
}

.northstar-kpi-grid {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}

.northstar-kpi {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 0.75rem 1.25rem;
    min-width: 120px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeIn 0.4s ease-out forwards;
}

.northstar-kpi:hover {
    transform: translateY(-3px);
    border-color: rgba(59, 130, 246, 0.4);
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.2);
}

.northstar-kpi:nth-child(1) { animation-delay: 0.1s; }
.northstar-kpi:nth-child(2) { animation-delay: 0.15s; }
.northstar-kpi:nth-child(3) { animation-delay: 0.2s; }
.northstar-kpi:nth-child(4) { animation-delay: 0.25s; }
.northstar-kpi:nth-child(5) { animation-delay: 0.3s; }

.northstar-value {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}

.northstar-label {
    font-size: 0.7rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
}

.northstar-actions {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
"""

# =============================================================================
# GLASSMORPHIC CARD CSS
# =============================================================================

GLASSMORPHIC_CARD_CSS = """
/* ========================================
   GLASSMORPHIC DESIGN SYSTEM
   ======================================== */

/* Base Glass Card */
.glass-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow:
        0 4px 30px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeInScale 0.4s ease-out forwards;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow:
        0 8px 40px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.08);
    border-color: rgba(59, 130, 246, 0.3);
}

/* Glass Card with Accent Border */
.glass-card-accent {
    position: relative;
    overflow: hidden;
}

.glass-card-accent::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--accent-start, #3b82f6), var(--accent-end, #22d3ee));
    border-radius: 16px 16px 0 0;
}

/* Accent Color Variants */
.glass-blue { --accent-start: #3b82f6; --accent-end: #22d3ee; }
.glass-green { --accent-start: #22c55e; --accent-end: #4ade80; }
.glass-purple { --accent-start: #a855f7; --accent-end: #ec4899; }
.glass-orange { --accent-start: #f97316; --accent-end: #fbbf24; }
.glass-red { --accent-start: #ef4444; --accent-end: #f87171; }

/* KPI Card Styles */
.kpi-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 14px;
    padding: 1.25rem;
    text-align: center;
    transition: all 0.3s ease;
}

.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.25);
    border-color: rgba(59, 130, 246, 0.4);
}

.kpi-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}

.kpi-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.25rem;
}

.kpi-delta {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.5rem;
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
}

.kpi-delta.positive {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
}

.kpi-delta.negative {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
}
"""

# =============================================================================
# STATUS INDICATORS CSS
# =============================================================================

STATUS_INDICATORS_CSS = """
/* ========================================
   STATUS INDICATORS
   ======================================== */

/* Status Badges */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.status-good {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #22c55e;
}

.status-warning {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #f59e0b;
}

.status-danger {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #ef4444;
}

/* Sync Status */
.sync-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.sync-complete {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #22c55e;
    animation: pulse-sync 2s ease-in-out infinite;
}

.sync-partial {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #f59e0b;
}

.sync-none {
    background: rgba(100, 116, 139, 0.15);
    border: 1px solid rgba(100, 116, 139, 0.4);
    color: #94a3b8;
}

.sync-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
}
"""

# =============================================================================
# ACTION CARDS CSS
# =============================================================================

ACTION_CARDS_CSS = """
/* ========================================
   ACTION CARDS
   ======================================== */

.action-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border-radius: 12px;
    padding: 1rem 1.25rem;
    border-left: 4px solid;
    margin-bottom: 0.75rem;
    transition: all 0.3s ease;
}

.action-card:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.action-urgent {
    border-left-color: #ef4444;
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(15, 23, 42, 0.95));
}

.action-urgent:hover {
    box-shadow: 0 4px 20px rgba(239, 68, 68, 0.2);
}

.action-important {
    border-left-color: #f59e0b;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(15, 23, 42, 0.95));
}

.action-important:hover {
    box-shadow: 0 4px 20px rgba(245, 158, 11, 0.2);
}

.action-normal {
    border-left-color: #3b82f6;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(15, 23, 42, 0.95));
}

.action-normal:hover {
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.2);
}

.action-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}

.action-icon {
    font-size: 1.25rem;
}

.action-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #f1f5f9;
}

.action-description {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.4;
}

.action-category {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    background: rgba(100, 116, 139, 0.2);
    color: #94a3b8;
    margin-top: 0.5rem;
}

/* Pulsing Alert for Urgent */
.action-urgent.pulse {
    animation: pulse-alert 2s ease-in-out infinite;
}
"""

# =============================================================================
# EMPTY STATE CSS
# =============================================================================

EMPTY_STATE_CSS = """
/* ========================================
   EMPTY STATE
   ======================================== */

.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.6));
    border: 2px dashed rgba(100, 116, 139, 0.3);
    border-radius: 16px;
    margin: 2rem 0;
}

.empty-state-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.7;
    animation: bounce 2s ease-in-out infinite;
}

.empty-state-title {
    font-size: 1.25rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f1f5f9, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.empty-state-description {
    color: #94a3b8;
    font-size: 0.95rem;
    max-width: 400px;
    margin: 0 auto 1.5rem;
    line-height: 1.5;
}

.empty-state-action {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    background: linear-gradient(135deg, #3b82f6, #22d3ee);
    border: none;
    border-radius: 10px;
    color: white;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

.empty-state-action:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
}
"""

# =============================================================================
# SKELETON LOADING CSS
# =============================================================================

SKELETON_CSS = """
/* ========================================
   SKELETON LOADING
   ======================================== */

.skeleton-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(100, 116, 139, 0.2);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
}

.skeleton-icon {
    width: 40px;
    height: 40px;
    border-radius: 8px;
    margin: 0 auto 0.75rem;
}

.skeleton-value {
    width: 60%;
    height: 2rem;
    border-radius: 6px;
    margin: 0 auto 0.5rem;
}

.skeleton-label {
    width: 80%;
    height: 1rem;
    border-radius: 4px;
    margin: 0 auto;
}

.loading-shimmer {
    background: linear-gradient(
        90deg,
        rgba(30, 41, 59, 0.9) 0%,
        rgba(59, 130, 246, 0.2) 50%,
        rgba(30, 41, 59, 0.9) 100%
    );
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
}
"""

# =============================================================================
# FORECAST CARDS CSS
# =============================================================================

FORECAST_CARDS_CSS = """
/* ========================================
   FORECAST DAY CARDS
   ======================================== */

.forecast-day-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    transition: all 0.3s ease;
    min-width: 100px;
}

.forecast-day-card:hover {
    transform: translateY(-4px);
    border-color: rgba(59, 130, 246, 0.4);
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.2);
}

.forecast-day-card.peak {
    border-color: #f59e0b;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(15, 23, 42, 0.95));
}

.forecast-day-card.peak::after {
    content: '★';
    position: absolute;
    top: -8px;
    right: -8px;
    font-size: 1.25rem;
    color: #f59e0b;
}

.forecast-day {
    font-size: 0.75rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.forecast-date {
    font-size: 0.7rem;
    color: #64748b;
    margin-bottom: 0.5rem;
}

.forecast-value {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}

.forecast-accuracy {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    margin-top: 0.5rem;
}

.accuracy-high {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
}

.accuracy-medium {
    background: rgba(34, 211, 238, 0.15);
    color: #22d3ee;
}

.accuracy-low {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
}

.forecast-ci {
    font-size: 0.65rem;
    color: #64748b;
    margin-top: 0.25rem;
}
"""

# =============================================================================
# COMPARISON PANEL CSS
# =============================================================================

COMPARISON_CSS = """
/* ========================================
   BEFORE/AFTER COMPARISON
   ======================================== */

.comparison-panel {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 1.5rem;
    align-items: stretch;
}

.comparison-before,
.comparison-after {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(100, 116, 139, 0.2);
}

.comparison-before {
    border-top: 4px solid #ef4444;
}

.comparison-after {
    border-top: 4px solid #22c55e;
}

.comparison-title {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 1rem;
}

.comparison-before .comparison-title {
    color: #ef4444;
}

.comparison-after .comparison-title {
    color: #22c55e;
}

.comparison-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    color: #3b82f6;
    padding: 0 1rem;
}

.comparison-savings {
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(15, 23, 42, 0.9));
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 12px;
    margin-top: 1.5rem;
}

.savings-value {
    font-size: 2rem;
    font-weight: 800;
    color: #22c55e;
}

.savings-label {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}
"""

# =============================================================================
# RESPONSIVE CSS
# =============================================================================

RESPONSIVE_CSS = """
/* ========================================
   RESPONSIVE DESIGN
   ======================================== */

@media (max-width: 1200px) {
    .northstar-kpi-grid {
        flex-wrap: wrap;
    }

    .northstar-kpi {
        min-width: calc(33% - 0.75rem);
    }
}

@media (max-width: 768px) {
    .northstar-strip {
        padding: 0.75rem 1rem;
    }

    .northstar-content {
        flex-direction: column;
        gap: 1rem;
    }

    .northstar-kpi {
        min-width: calc(50% - 0.5rem);
    }

    .comparison-panel {
        grid-template-columns: 1fr;
    }

    .comparison-arrow {
        transform: rotate(90deg);
        padding: 1rem 0;
    }

    .glass-card {
        padding: 1rem;
    }
}

@media (max-width: 480px) {
    .northstar-kpi {
        min-width: 100%;
    }

    .northstar-value {
        font-size: 1.25rem;
    }

    .kpi-value {
        font-size: 1.5rem;
    }
}
"""


# =============================================================================
# MAIN STYLE GENERATOR
# =============================================================================

def get_unified_dashboard_styles() -> str:
    """
    Generate complete CSS for the unified dashboard.

    Returns:
        Complete CSS string wrapped in <style> tags
    """
    return f"""
<style>
{ANIMATION_KEYFRAMES}
{NORTHSTAR_HEADER_CSS}
{GLASSMORPHIC_CARD_CSS}
{STATUS_INDICATORS_CSS}
{ACTION_CARDS_CSS}
{EMPTY_STATE_CSS}
{SKELETON_CSS}
{FORECAST_CARDS_CSS}
{COMPARISON_CSS}
{RESPONSIVE_CSS}

/* Stagger Animation Helper */
.stagger-list > * {{
    animation: fadeIn 0.4s ease-out forwards;
    opacity: 0;
}}

.stagger-list > *:nth-child(1) {{ animation-delay: 0.1s; }}
.stagger-list > *:nth-child(2) {{ animation-delay: 0.15s; }}
.stagger-list > *:nth-child(3) {{ animation-delay: 0.2s; }}
.stagger-list > *:nth-child(4) {{ animation-delay: 0.25s; }}
.stagger-list > *:nth-child(5) {{ animation-delay: 0.3s; }}
.stagger-list > *:nth-child(6) {{ animation-delay: 0.35s; }}
.stagger-list > *:nth-child(7) {{ animation-delay: 0.4s; }}

/* Tab Content Animation */
.tab-content {{
    animation: fadeIn 0.3s ease-out forwards;
}}

/* Section Headers */
.section-header {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}}

.section-header-icon {{
    font-size: 1.25rem;
}}

/* Glass Divider */
.glass-divider {{
    height: 1px;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.1),
        transparent
    );
    margin: 1.5rem 0;
}}
</style>
"""
