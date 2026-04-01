#!/bin/bash

# Navigate to the advanced_model directory in case the script is run from elsewhere
cd "$(dirname "$0")/.."

echo "========================================="
echo "📊 Checking bias corrections staleness..."
echo "========================================="
# Monthly: recompute bias corrections if stale (>30 days or insufficient samples)
python3 -m analysis.compute_retroactive_bias --auto-update
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
