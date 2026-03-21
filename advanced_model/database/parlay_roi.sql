-- ============================================================
-- Phase 5.D — Parlay ROI Monitoring Queries
-- Run these against bet_tracker.db to inspect parlay health
-- ============================================================

-- 1. Daily parlay ROI
SELECT
    game_date,
    COUNT(*)                        AS n_parlays,
    SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) AS wins,
    ROUND(AVG(roi) * 100, 1)        AS avg_roi_pct,
    ROUND(SUM(payout - stake), 4)   AS net_units
FROM parlays
WHERE result IS NOT NULL
GROUP BY game_date
ORDER BY game_date DESC;

-- 2. Rolling 30-day summary
SELECT
    COUNT(*)                        AS n_parlays,
    SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) AS wins,
    ROUND(AVG(roi) * 100, 1)        AS avg_roi_pct,
    ROUND(SUM(payout - stake), 4)   AS net_units_30d
FROM parlays
WHERE game_date >= date('now', '-30 days')
  AND result IS NOT NULL;

-- 3. Leg win rate by category in winning vs losing parlays
SELECT
    p.result                        AS parlay_result,
    pl.stat_category,
    COUNT(*)                        AS n_legs,
    SUM(pr.is_win)                  AS leg_wins,
    ROUND(100.0 * SUM(pr.is_win) / COUNT(*), 1) AS leg_win_pct
FROM parlays p
JOIN parlay_legs pl ON pl.parlay_id = p.id
LEFT JOIN pick_results pr ON pr.pick_id = pl.pick_id
WHERE p.result IS NOT NULL
  AND pr.is_win IS NOT NULL
GROUP BY p.result, pl.stat_category
ORDER BY p.result, n_legs DESC;

-- 4. Shadow pick validation: ban_reason vs. actual outcomes
SELECT
    ban_reason,
    COUNT(*)                                        AS n,
    SUM(actual_result)                              AS hypothetical_wins,
    ROUND(100.0 * SUM(actual_result) / COUNT(*), 1) AS hypothetical_win_pct
FROM shadow_picks
WHERE actual_result IS NOT NULL
GROUP BY ban_reason
ORDER BY n DESC;

-- 5. Pending parlays (not yet resolved)
SELECT id, game_date, combined_odds, stake
FROM parlays
WHERE result IS NULL
ORDER BY game_date DESC;
