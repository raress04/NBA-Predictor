# 🏀 AiBetTool — Quantitative NBA Prop & Betting Engine

An end-to-end quantitative analytics and decision-support engine for NBA player proposition markets (Points, Rebounds, Assists) and game lines. Built to model true distribution tails, exploit sportsbook pricing inefficiencies, and calibrate confidence using Bayesian methods and gradient boosting.

---

## ⚡ Core Architecture

```
                    [ Live Ingestion & Scrapers ]
                                 │
                     (Odds, Rosters, Injuries)
                                 ▼
                 [ Markov Chain / MCMC Simulation ]
                     5,000 iterations per game
                                 │
                     (Raw Prop Distributions)
                                 ▼
                     [ ML Calibration Layer ]
               XGBoost Regressors + Beta-Posterior Tuning
                                 │
                     (Calibrated Probabilities)
                                 ▼
                 [ Production Decision Engine ]
               Kelly EV Gates • Min Edge • Blowout Filter
                   │                                │
            (Passed Gates)                   (Filtered Out)
                   ▼                                ▼
       [ Production Bets / Parlays ]        [ Shadow Logging ]
       Logged to bet_tracker.db          Continuous evaluation
                   │                                │
                   ▼                                ▼
              [ Automated Daily LaTeX / PDF Reports ]
```

1. **ETL & Live Ingestion**:
   - Automated ingestion of historical box scores, live rotowire/injury designations, and real-time sportsbook consensus lines.
   - Dual-database SQLite storage (`nba_data.db` for stats, `bet_tracker.db` for ledger and audit logs).

2. **Simulation & Projection**:
   - Positional Markov Chain Monte Carlo simulations estimating possessions, rotation minutes, shot distributions, and rebound opportunities.
   - Computes empirical prop distributions rather than static point projections.

3. **Machine Learning & Calibration**:
   - **XGBoost Layer**: Gradient boosted trees refine projection distributions against market lines.
   - **Bayesian Beta-Posterior Calibration**: Mitigates overconfidence by tuning stated probability against empirical historical hit rate via grid search on Expected Calibration Error (ECE).
   - **Risk Gates**: Filters on fractional Kelly Criterion, minimum expected value (EV), confidence interval width, and blowout susceptibility.

4. **Shadow Pipeline**:
   - Stores rejected candidate picks (`shadow_picks`) and ML candidate props (`ml_shadow_picks`) to eliminate survivorship bias and validate production gates over long sample horizons.

5. **Automated Settlement & Reporting**:
   - Daily automated pipelines compile publication-ready LaTeX PDF dossiers for morning projections, yesterday's settled performance analysis, and ML shadow diagnostics.

---

## 📁 Repository Structure

```plaintext
AiBetTool/
├── advanced_model/
│   ├── analyze/          # Historical LaTeX and PDF analysis audits
│   ├── config/           # Calibration parameters, team maps, model configs
│   ├── database/         # SQLite storage (nba_data.db, bet_tracker.db)
│   ├── etl/              # Live scrapers, backfillers, settlement scripts
│   ├── ml/               # XGBoost models, training scripts, feature engineering
│   ├── predictions/      # Daily prediction outputs and compiled PDFs
│   ├── results/          # Yesterday box score settlement artifacts
│   └── scripts/          # run_pipeline.sh and automation wrappers
├── frontend/             # Interactive UI dashboard
└── Plans/                # Technical audit documents and architecture specs
```

---

## 🚀 Getting Started

### 1. Requirements & Setup

```bash
# Clone repository
git clone https://github.com/your-repo/AiBetTool.git
cd AiBetTool

# Virtual environment setup
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r advanced_model/requirements.txt  # or install pandas numpy xgboost requests beautifulsoup4

# Configuration
echo "ODDS_API_KEY=your_the_odds_api_key" > advanced_model/.env
```

### 2. Daily Pipeline

Run the automated morning run:

```bash
cd advanced_model
./scripts/run_pipeline.sh
```

The script automatically:
1. Reconciles and grades yesterday's props against official box scores (`prepare_analyze.py`).
2. Scrapes today's injuries, rosters, and live market lines (`live_ingestion.py`).
3. Runs MCMC simulations and applies ML calibration.
4. Compiles the daily projection report into `predictions/.../predicts_report_YYYY-MM-DD.pdf`.

---

## 📊 Evaluation & Diagnostics

- **Model Win Rate & Ledger**: `python3 -m etl.bet_tracker summary`
- **ML Shadow Diagnostics**: `python3 -m etl.analyze_ml_shadow --date YYYY-MM-DD`
- **Manual Backtesting**: `python3 -m etl.live_scraper --date YYYY-MM-DD`
