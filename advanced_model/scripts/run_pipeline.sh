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
LOG_BASE="logs/${MONTH}"
mkdir -p "${LOG_BASE}/daily_metrics"
mkdir -p "${LOG_BASE}/monitoring"
mkdir -p "${LOG_BASE}/retro_live_alignment"
mkdir -p "${LOG_BASE}/odds_coverage_audit"
mkdir -p "${LOG_BASE}/date_audit"


echo "========================================="
echo "🔄 Preparing Yesterday's Results & Analysis..."
echo "========================================="
python3 -m etl.prepare_analyze
if [ $? -ne 0 ]; then
    echo "❌ Yesterday preparation failed. Aborting pipeline before model invocation."
    exit 1
fi
echo "[Stage] Yesterday preparation complete."
echo ""

echo "========================================="
echo "🏀 Running Live Scraper..."
echo "========================================="
python3 -m etl.live_ingestion

# Check if the python script executed successfully
if [ $? -ne 0 ]; then
    echo "❌ Scraper failed to run. Aborting PDF generation."
    exit 1
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
python3 -m etl.seed_all_live
python3 -m etl.fetch_range_box_scores
echo ""

echo "========================================="
echo "📊 Phase 7 — Daily Metrics & Health Report..."
echo "========================================="
python3 -m ml.daily_metrics   --out "${LOG_BASE}/daily_metrics/daily_metrics_${DATE}.txt"
python3 -m ml.monitoring       --out "${LOG_BASE}/monitoring/monitoring_${DATE}.txt"
python3 -m ml.compare_retro_live --out "${LOG_BASE}/retro_live_alignment/retro_live_alignment_${DATE}.txt"
python3 -m scripts.audit_odds          --out "${LOG_BASE}/odds_coverage_audit/odds_coverage_audit_${DATE}.txt"

# Date-consistency check (read-only; never fails the pipeline). See Plans/CRITICAL_BUGS_FOUND_06.11.md §15.
DATE_AUDIT="${LOG_BASE}/date_audit/date_audit_${DATE}.txt"
if python3 scripts/audit_dates.py . > "${DATE_AUDIT}" 2>&1; then
    grep -E "^Summary by status|flagged" "${DATE_AUDIT}" | grep -v " 0 flagged" || true
    echo "[Date audit] Full report: ${DATE_AUDIT}"
else
    echo "⚠️  Date audit failed to run — see ${DATE_AUDIT}"
fi
echo "✅ All logs generated in ${LOG_BASE}"

