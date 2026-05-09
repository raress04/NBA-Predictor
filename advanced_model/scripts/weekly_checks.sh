#!/bin/bash
# Weekly promotion gate — run manually or via cron once per week.
# Checks whether the model meets all criteria for Kelly re-enablement.

cd "$(dirname "$0")/.."

echo "========================================="
echo "🏆 Weekly Promotion Gate Check"
echo "========================================="
python3 -m ml.check_promotion
