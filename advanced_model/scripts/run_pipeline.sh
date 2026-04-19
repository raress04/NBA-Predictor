#!/bin/bash

# Navigate to the advanced_model directory in case the script is run from elsewhere
cd "$(dirname "$0")/.."
export PYTHONPATH=$PYTHONPATH:$(pwd)/..:$(pwd)

echo "========================================="
echo "📊 Checking bias corrections staleness..."
echo "========================================="
# Monthly: recompute bias corrections if stale (>30 days or insufficient samples)
python3 -m analyze.compute_retroactive_bias --auto-update
echo "[Stage 3] Bias correction check complete."
echo ""

echo "========================================="
echo "🏀 Running Live Scraper..."
echo "========================================="
python3 etl/live_scraper.py

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
WHERE po.game_date = date('now')
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
echo "📄 Generating LaTeX Report (predicts_report.tex)..."
echo "========================================="
python3 etl/predicts_to_pdf.py

if [ -f "predicts_report.tex" ]; then
    pdflatex -interaction=nonstopmode predicts_report.tex
    
    # Cleanup all the auxiliary LaTeX pollution files gracefully
    rm -f *.aux *.log *.out *.fls *.fdb_latexmk texput.log
    
    echo ""
    echo "✅ Finished! Your PDF is compiled (predicts_report.pdf)."
else
    echo "❌ predicts_report.tex not found in $(pwd)"
fi

echo ""
echo "========================================="
echo "🗄️ Archiving Daily Predictions..."
echo "========================================="
# Get current date and month (lowercase)
DATE=$(date +%Y-%m-%d)
MONTH=$(date +%B | tr '[:upper:]' '[:lower:]')

# Ensure the month directory exists
ARCHIVE_DIR="predictions/${MONTH}"
mkdir -p "$ARCHIVE_DIR"

if [ -f "predicts.txt" ]; then
    cp predicts.txt "${ARCHIVE_DIR}/predicts_${DATE}.txt"
    echo "✅ Saved exact predictions to: ${ARCHIVE_DIR}/predicts_${DATE}.txt"
else
    echo "⚠️  No predicts.txt found to archive."
fi

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
python3 -m analysis.daily_metrics
python3 -m analysis.monitoring
echo ""
