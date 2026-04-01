
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
    ("REBOUNDS", "OVER"),   # 40.0% verified win rate (n=20+)
    ("SPREAD",   "COVER"),  # 38.5% verified win rate (n=20+)
}

# ── Per-Category Priors ──────────────────────────────────────────
# Priors derived from real observed win rates in March live data.
# Used for Phase 2.A Beta-posterior confidence calculations.
CATEGORY_PRIORS = {
    "TOTAL_OVER":     0.75,  # 75.0% empirical win rate
    "ASSISTS_UNDER":  0.65,  # 66.7% empirical win rate
    "POINTS_OVER":    0.54,  # insufficient live data — use neutral prior
    "POINTS_UNDER":   0.50,  # 50.0% — coin flip, strong neutral pull
    "REBOUNDS_OVER":  0.40,  # banned, but retain prior for logging
    "REBOUNDS_UNDER": 0.40,  # banned, but retain prior for logging
    "SPREAD_COVER":   0.38,  # 38.5% — banned, but retain prior for logging
    "DEFAULT":        0.54,  # fallback for any unlisted category
}

# ── Tiered Bias Correction Config (Phase 4.A) ─────────────────────
# Projection value thresholds that separate bench players from starters
PROJECTION_TIERS = {
    "POINTS":   10.0,   # < 10.0 = bench tier, >= 10.0 = starter tier
    "REBOUNDS":  4.0,   # < 4.0  = bench tier, >= 4.0  = starter tier
    "ASSISTS":   3.0,   # < 3.0  = bench tier, >= 3.0  = starter tier
}

# Bias corrections derived from projection_outcomes (actual - projected mean)
# Format: (category, tier) -> correction to ADD to raw projection
TIERED_BIAS_CORRECTIONS = {
    ("POINTS",   "bench"):   +1.10,
    ("POINTS",   "starter"): -0.67,
    ("REBOUNDS", "bench"):   +0.77,
    ("REBOUNDS", "starter"): +0.38,
    ("ASSISTS",  "bench"):   +0.27,
    ("ASSISTS",  "starter"): -0.17,
}

def is_allowed(category: str, direction: str) -> bool:
    """Returns True if the category/direction combination is not banned."""
    return (category.upper(), direction.upper()) not in BANNED_COMBINATIONS

def get_prior(category: str, direction: str) -> float:
    """Returns the prior_mean for a specific category/direction."""
    key = f"{category.upper()}_{direction.upper()}"
    return CATEGORY_PRIORS.get(key, CATEGORY_PRIORS["DEFAULT"])
