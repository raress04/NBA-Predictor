#!/bin/bash
export PYTHONIOENCODING=utf-8

# Navigate to the advanced_model directory in case the script is run from elsewhere
cd "$(dirname "$0")/.."
export PYTHONPATH=$PYTHONPATH:$(pwd)/..:$(pwd)

# ── Dates: one rule for the whole run (utils/slate_date.py) ───
# SLATE_DATE = US-Eastern date of this run = the date live_ingestion stamps on its picks.
# Nightly logs are named and filed by it too, so a run after local midnight stays with its slate.
SLATE_DATE=$(python3 -m utils.slate_date)
DATE="${SLATE_DATE}"
MONTH=$(date -d "${SLATE_DATE}" +%B | tr '[:upper:]' '[:lower:]')

# ── Log directory structure ──────────────────────────────────
LOG_BASE="${PIPELINE_LOG_ROOT:-logs}/${MONTH}"
mkdir -p "${LOG_BASE}/daily_metrics"
mkdir -p "${LOG_BASE}/monitoring"
mkdir -p "${LOG_BASE}/retro_live_alignment"
mkdir -p "${LOG_BASE}/odds_coverage_audit"
mkdir -p "${LOG_BASE}/date_audit"
mkdir -p "${LOG_BASE}/injuries"

# ── W2: unattended failures are loud and non-fatal (utils/run_alerts.py) ──
# Every problem becomes one "[ALERT ERROR|WARN] source: ..." line, also kept in
# logs/scheduled/alerts_<slate>.log. The night ends with NIGHT STATUS: OK | DEGRADED | FAILED,
# and a Windows pop-up when it is not OK.
python3 -m utils.run_alerts --start

alert() {   # alert LEVEL SOURCE MESSAGE...
    python3 -m utils.run_alerts "$@" || echo "[ALERT $1] $2: ${*:3}"
}

finish() {  # finish [--failed]; prints the night status line, then exits
    python3 -m utils.run_alerts --summary "$@" || echo "NIGHT STATUS: unknown (run_alerts failed)"
    if [ "$1" = "--failed" ]; then exit 1; fi
    exit 0
}

run_step() {  # run_step NAME CMD...; a failed step is a WARN, never stops the night
    local name="$1"; shift
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        alert WARN "$name" "exited with code $rc -> later steps still run; see the lines above"
    fi
    return 0
}


echo "========================================="
echo "🔄 Preparing Yesterday's Results & Analysis..."
echo "========================================="
python3 -m etl.prepare_analyze
PREP_RC=$?
if [ $PREP_RC -ne 0 ]; then
    # W2: grading yesterday failed (box scores missing, results file or PDF not written).
    # Tonight's predictions still run; the next night grades this date (catch-up step).
    alert ERROR prepare_analyze "grading yesterday failed (exit ${PREP_RC}) -> tonight's predictions run anyway; grading retried next night. If box scores were missing, tonight's form lacks yesterday's games"
else
    echo "[Stage] Yesterday preparation complete."
fi
echo ""

# ── T3 (2026-11-06): store the official injury report every day ──
# Previous slate date's 5 PM ET report -> nba_data.db historical_injuries (INSERT OR IGNORE).
# Non-fatal: a failure here never stops the pipeline.
echo "========================================="
echo "🩹 Storing yesterday's official injury report..."
echo "========================================="
python3 -m etl.scrape_historical_injuries --previous-slate > "${LOG_BASE}/injuries/injuries_${DATE}.txt" 2>&1 \
    && tail -3 "${LOG_BASE}/injuries/injuries_${DATE}.txt" \
    || alert WARN injury_report "official injury report not stored (see ${LOG_BASE}/injuries/injuries_${DATE}.txt) -> continuing"
echo ""

echo "========================================="
echo "🏀 Running Live Scraper..."
echo "========================================="
python3 -m etl.live_ingestion
LIVE_RC=$?

# No predictions tonight (ESPN down, no games matched, crash): the night is FAILED.
if [ $LIVE_RC -ne 0 ]; then
    alert ERROR live_ingestion "exited with code ${LIVE_RC} -> no predictions tonight; later steps skipped"
    finish --failed
fi

echo ""
echo "========================================="
echo "🔍 Step 3.4 — Bias Alignment Verification"
echo "========================================="
sqlite3 database/bet_tracker.db "
SELECT
    po.category,
    ROUND(AVG(po.bookmaker_line - po.projected_value), 2) AS avg_gap,
    ROUND(AVG(ABS(po.bookmaker_line - po.projected_value)), 2) AS mae,
    COUNT(*) AS n
FROM projection_outcomes po
WHERE po.game_date = '${SLATE_DATE}'
  AND po.bookmaker_line IS NOT NULL
  AND COALESCE(po.superseded, 0) = 0
GROUP BY po.category;
" | awk -F'|' 'BEGIN {
    printf "%-12s %8s %8s %6s  %s\n", "CATEGORY", "AVG_GAP", "MAE", "N", "STATUS"
    printf "%s\n", "--------------------------------------------"
}
{
    cat=$1; gap=$2; mae=$3; n=$4
    status="✅ OK"
    if (cat == "TOTAL"   && (gap+0 > 5.0 || gap+0 < -5.0)) status="⚠️  OVER TARGET"
    if (cat != "TOTAL"   && (gap+0 > 2.0 || gap+0 < -2.0)) status="⚠️  OVER TARGET"
    printf "%-12s %8s %8s %6s  %s\n", cat, gap, mae, n, status
}'
echo ""

echo ""
echo "========================================="
echo "📈 Phase 7 — Data Seeding & Resolution..."
echo "========================================="
run_step seed_all_live python3 -m etl.seed_all_live
run_step fetch_range_box_scores python3 -m etl.fetch_range_box_scores
echo ""

echo "========================================="
echo "📊 Phase 7 — Daily Metrics & Health Report..."
echo "========================================="
run_step daily_metrics python3 -m ml.daily_metrics   --out "${LOG_BASE}/daily_metrics/daily_metrics_${DATE}.txt"
run_step monitoring python3 -m ml.monitoring       --out "${LOG_BASE}/monitoring/monitoring_${DATE}.txt"
run_step compare_retro_live python3 -m ml.compare_retro_live --out "${LOG_BASE}/retro_live_alignment/retro_live_alignment_${DATE}.txt"
run_step audit_odds python3 -m scripts.audit_odds          --out "${LOG_BASE}/odds_coverage_audit/odds_coverage_audit_${DATE}.txt"

# Date-consistency check (read-only; never fails the pipeline). See Plans/CRITICAL_BUGS_FOUND_06.11.md §15.
DATE_AUDIT="${LOG_BASE}/date_audit/date_audit_${DATE}.txt"
if python3 scripts/audit_dates.py . > "${DATE_AUDIT}" 2>&1; then
    grep -E "^Summary by status|flagged" "${DATE_AUDIT}" | grep -v " 0 flagged" || true
    echo "[Date audit] Full report: ${DATE_AUDIT}"
else
    alert WARN audit_dates "date audit failed to run -> see ${DATE_AUDIT}"
fi
echo "✅ All logs generated in ${LOG_BASE}"
finish

