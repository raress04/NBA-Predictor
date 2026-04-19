
# Stat-specific weights for blending Simulation Confidence and Player History DB Confidence.
# Assists are noisy, so we trust historical hit rate equal to the sim.
# Points and Totals are more reliable in Sims, so we weight Sim higher.
SIM_DB_WEIGHTS = {
    'assists': (0.50, 0.50),
    'rebounds': (0.60, 0.40),
    'points': (0.80, 0.20),
    'totals': (0.80, 0.20),
    'combo': (0.70, 0.30)
}

# ── Banned Combinations ──────────────────────────────────────────
# ISSUE 4 FIX (Stage 0): Reduced from 5 to 2 bans.
# Only keep categories with VERIFIED losing win rates over n>=20 picks.
# REBOUNDS UNDER, POINTS UNDER, TOTAL UNDER removed — premature bans on
# tiny sample sizes that froze the learning loop entirely.
BANNED_COMBINATIONS = {
    # REBOUNDS ban LIFTED 2026-04-07 — 55.1% win rate across 9,291 picks (Jan-Mar 2026)
    # Re-ban if live win rate drops below 52% over next 30 days
    ("SPREAD",   "COVER"),  # 38.5% verified win rate (n=1,204) — confirmed dead weight
}

# ── Per-Category Priors ──────────────────────────────────────────
# Priors derived from real observed win rates in March live data.
# Used for Phase 2.A Beta-posterior confidence calculations.
CATEGORY_PRIORS = {
    "TOTAL_OVER":      0.53,  # updated — 83 picks, 53.0%
    "ASSISTS_OVER":    0.56,  # updated — 9,157 picks, 55.7%
    "ASSISTS_UNDER":   0.56,  # updated — same pool
    "POINTS_OVER":     0.55,  # updated — 9,346 picks, 54.5%
    "POINTS_UNDER":    0.55,  # updated — same pool
    "REBOUNDS_OVER":   0.55,  # updated — 9,291 picks, 55.1% (ban lifted)
    "REBOUNDS_UNDER":  0.55,  # updated — same pool (ban lifted)
    "SPREAD_COVER":    0.38,  # unchanged — banned, kept for logging only
    "DEFAULT":         0.54,  # unchanged
}

# ── Tiered Bias Correction Config (Phase 4.A) ─────────────────────
# Projection value thresholds that separate bench players from starters
PROJECTION_TIERS = {
    "POINTS":   10.0,   # < 10.0 = bench tier, >= 10.0 = starter tier
    "REBOUNDS":  4.0,   # < 4.0  = bench tier, >= 4.0  = starter tier
    "ASSISTS":   3.0,   # < 3.0  = bench tier, >= 3.0  = starter tier
}

# TIERED_BIAS_CORRECTIONS — recomputed 2026-04-07
# Source: retrospective_db.db — retroactive_live (n=8,937+ rows, Jan 1–Mar 24, 2026)
# Previous values (rollback reference):
#   POINTS bench=+1.10, POINTS starter=-0.67
#   REBOUNDS bench=+0.77, REBOUNDS starter=+0.38
#   ASSISTS bench=+0.27, ASSISTS starter=-0.17
#   TOTAL: not previously set
# TIERED_BIAS_CORRECTIONS — recomputed 2026-04-07 from bias_corrections_computed.json
# Source: retrospective_db.db — 29,081 picks Jan–Mar 2026
# TOTAL_high zeroed per spec: negative correction contradicts expected pattern,
#   likely reflects missing injury data in retroactive high-scoring games.
#   Revisit after full injury matrix re-run.
TIERED_BIAS_CORRECTIONS = {
    ('POINTS',   'bench'):   +0.63,    # n=3,811  — computed (JSON)
    ('POINTS',   'starter'): +0.28,    # n=5,889  — computed (JSON)
    ('REBOUNDS', 'bench'):   +0.22,    # n=4,654  — computed (JSON)
    ('REBOUNDS', 'starter'): +0.18,    # n=5,024  — computed (JSON)
    ('ASSISTS',  'bench'):   +0.13,    # n=5,864  — computed (JSON)
    ('ASSISTS',  'starter'): +0.39,    # n=3,635  — computed (JSON)
    ('TOTAL',    'high'):    +0.00,    # n=32  — ZEROED (negative raw: -5.51, spec rule applied)
    ('TOTAL',    'medium'):  +3.39,    # n=41  — computed (JSON, valid)
    ('TOTAL',    'low'):     +10.00,   # n=10  — spec fallback (insufficient data)
}

# ── Confidence Parameters ────────────────────────────────────────
# Updated 2026-04-07 per decision: prior_strength=150 for stronger pull
# toward category priors across 5k simulations.
# Previous: 50 → now 150 (plan item #4)
CONFIDENCE_PRIOR_STRENGTH = 400  # Multi-objective calibration (ECE, Brier, Var, AUC) on Jan-Mar 2026 retro.
# Baseline: prior=100 -> ECE=0.03463, Brier=0.24857, Var=0.001502, AUC=0.48447
# Selected: prior=400 -> ECE=0.02411, Brier=0.24812, Var=0.001086, AUC=0.48304
# Rule: smallest prior_strength where ECE/Brier improve AND Var/AUC stay informative.

# Max valuable picks surfaced per day (caps pick volume, improves selectivity)
# Previous: no explicit cap → now 15 (plan item #4)
MAX_VALUABLE_PICKS_PER_DAY = 15

# ── TOTAL Parlay Gate ─────────────────────────────────────────────
# Do NOT use TOTAL as a parlay leg until sample size and MAE thresholds are met.
# Plan item #5: gate lifted when n > 200 AND MAE < 8.0
TOTAL_MIN_PICKS_FOR_PARLAY = 200   # current: ~83 picks — gate CLOSED
TOTAL_MAX_MAE_FOR_PARLAY   = 8.0   # current: ~15.2 MAE — gate CLOSED

# Minimum edge percentage required to consider a pick "valuable"
MIN_EDGE = 2.0

# Max prop confidence cap — raised from 78.0 based on n=29,081 retroactive analysis.
MAX_PROP_CONFIDENCE = 82.0

# ── Trade Context + EWMA (Issue 5) ───────────────────────────────
# Minimum post-trade games before switching to new-team-only stats.
# Below this threshold, full career log is used but high_uncertainty=True.
TRADE_CONTEXT_MIN_GAMES   = 5
# After this many games on new team, new-team stats are used exclusively
# with no uncertainty flag.
TRADE_CONTEXT_BLEND_GAMES = 25
# EWMA span: exponential recency weight applied to last N games.
# span=10 means the 10 most recent games carry ~63% of total weight.
EWMA_SPAN = 10



def is_allowed(category: str, direction: str) -> bool:
    """Returns True if the category/direction combination is not banned."""
    return (category.upper(), direction.upper()) not in BANNED_COMBINATIONS

def get_prior(category: str, direction: str) -> float:
    """Returns the prior_mean for a specific category/direction."""
    key = f"{category.upper()}_{direction.upper()}"
    return CATEGORY_PRIORS.get(key, CATEGORY_PRIORS["DEFAULT"])
