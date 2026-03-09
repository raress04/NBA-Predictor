# 🏀 AiBetTool — NBA Parlay Prediction Engine

A quantitative NBA betting tool that runs Monte Carlo simulations to generate optimized parlays, complete with daily PDF projection reports and long-term performance tracking.

---

## 🚀 One-Time Setup

Before you start using the tool, run these commands to prepare your environment.

```bash
# 1. Install dependencies
pip install nba_api requests beautifulsoup4 pandas numpy python-dotenv

# 2. Build the historical database (first time only — takes ~30 seconds)
cd advanced_model
python3 -m etl.fast_backfiller

# 3. Add your Odds API key (Get a free key at https://the-odds-api.com)
echo ODDS_API_KEY=your_key_here > .env
```

---

## 📅 Daily Workflow: Projections & Picks (Morning)

Every morning, run these commands to fetch today's matchups, simulate the games, find edges against live sportsbook lines, and generate your parlay picks.

```bash
# Ensure you are in the advanced_model directory
cd advanced_model

# 1. Run the simulation engine and fetch odds
python3 -m etl.live_scraper
```
**What this does:**
- Detects tonight's NBA games.
- Pulls live injury reports and builds active rosters/minutes.
- Runs 500 Markov chain simulations per game.
- Fetches live odds and compares them to the model to find valuable edges.
- Generates `predicts.txt` (raw text) and `predicts_report.tex` (formatted report).

```bash
# 2. Generate the beautiful PDF report
pdflatex predicts_report.tex
```
**What this does:**
- Converts the mathematical LaTeX report into a highly readable PDF (`predicts_report.pdf`).
- This PDF contains your game-by-game breakdowns, highlighted valuable picks, and top parlay recommendations. Includes your historical win rate automatically pulled from the DB.

---

## 📈 Daily Workflow: Review & Tracking (Next Morning)

The next day, evaluate how your predictions performed and update the AI's internal tracking database so it can calibrate its confidence scores.

```bash
# 1. Fetch yesterday's actual game results and evaluate picks
python3 -m etl.fetch_yesterday
```
**What this does:**
- Downloads the official NBA box scores from last night.
- Compares those actual results against yesterday's `predicts.txt`.
- Generates `yesterday_results.txt`, along with the analysis reports: `analyze.txt` and `analyze.tex`.

```bash
# 2. Generate the Analysis PDF report
pdflatex analyze.tex
```
**What this does:**
- Compiles the performance review into `analyze.pdf`, giving you a harsh, honest breakdown of what picks won/lost, your expected value difference, and where the model deviated.

```bash
# 3. Seed the results into the permanent Bet Tracker Database
python3 -m etl.bet_tracker seed --date 2026-03-04
# Note: Replace '2026-03-04' with the date you are reviewing, or use '--date all'
```
**What this does:**
- Reads the newly generated `analyze.tex` and inserts the exact picks, odds, and win/loss outcomes into the permanent SQLite `bet_tracker.db`.
- **Crucial:** The `parlay_builder` uses this database to *automatically calibrate* future confidence scores. If the model is losing on Assists, it will automatically penalize Assist confidence tonight.

---

## ⏪ Backtesting: Adding Past Days to the Database

If you miss a day, or want to simulate a game from the past to build up your AI's tracking database (`bet_tracker.db`), you must run the entire pipeline backward for a specific date (e.g., `YYYY-MM-DD`). Note: Historical days will not have live odds, but the model will still simulate and grade the projections.

```bash
# 1. Run the simulation for the past date (Outputs predicts_YYYY-MM-DD.txt)
python3 -m etl.live_scraper --date 2026-03-01

# 2. Fetch the actual results for that past date (Outputs results_YYYY-MM-DD.txt)
python3 -m etl.fetch_yesterday --date 2026-03-01

# 3. (Optional) You can run the analysis script on the past files if you want the PDF report
# python3 -m etl.analyze_predictions --predicts predicts_2026-03-01.txt --results results_2026-03-01.txt

# 4. Seed the results into the permanent database
python3 -m etl.bet_tracker seed --date 2026-03-01
```

---

## 📊 Analytics & Maintenance Commands

Use these commands for manual system checks or data insight.

```bash
# View all-time model performance summary
python3 -m etl.bet_tracker summary
```
**What this does:**
- Prints a terminal dashboard showing your total win rate, win rate broken down by specific category (Points, Rebounds, Spread, Total), and accuracy grouped by Confidence Band (e.g. 80-89%, 90%+).

```bash
# Rebuild the core statistical database from scratch (if corrupted)
python3 -m etl.fast_backfiller
```
**What this does:**
- Wipes the existing `nba_data.db` and rapidly redownloads 7 seasons of NBA play-by-play and box score data. Only use if simulations are missing players or acting strangely.
