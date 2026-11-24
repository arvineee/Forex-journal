
📈 Forex Journal AI
Forex Journal AI is a next-generation trading journal designed not just to record trades, but to actively analyze trader psychology, detect behavioral patterns, and provide actionable insights using Machine Learning. It features a conversational AI assistant, advanced statistical analysis, and automated bias detection.
🚀 Key Features
🧠 The AI Learning Engine (ForexAI)
The core of the application is the ForexAI engine, which uses scikit-learn and statistical modeling to analyze your trading history.
 * Psychological Bias Detection: Automatically detects "Loss Aversion," "Recency Bias," and "Confirmation Bias" based on holding times and trade clustering.
 * Behavioral Pattern Recognition: Identifies "Revenge Trading" (increasing size/frequency after losses) and analyzes winning/losing streaks.
 * Emotional Impact Analysis: Correlates self-reported emotions (e.g., "Anxious", "Confident") with P&L to determine your optimal emotional state.
 * Predictive Modeling: Uses a Random Forest Classifier to estimate the probability of success for new trade setups based on historical data.
🤖 Conversational AI Assistant
Integrated chatbot capable of context-aware trading analysis.
 * Natural Language Queries: Ask "How is my performance this week?" or "Analyze trade #42".
 * Context Memory: The bot remembers previous questions and entities (symbols, strategies) for a fluid conversation.
 * Intent Classification: Distinguishes between performance queries, emotional analysis, risk assessments, and educational requests.
📊 Advanced Analytics
 * Broker-Accurate P&L: Custom calculation logic for Forex, Gold (XAU), Silver (XAG), and JPY pairs ensuring accurate pip values.
 * Metric Calculation: Sharpe Ratio, Profit Factor, Maximum Drawdown, and Expectancy calculations.
 * Equity Curves: Visual generation of account growth over time.
🛠 System Features
 * Multi-Strategy Tracking: Associate trades with specific strategies to track their individual win rates.
 * Image Annotation: Upload and annotate trade entry/exit charts.
 * Termux Support: Optimized for running on Android devices via Termux.
📂 Project Structure
forex_journal/
├── app/
│   ├── ai/
│   │   ├── chatbot.py         # NLP and intent classification logic
│   │   ├── learning_engine.py # Scikit-learn models and pattern detection
│   │   └── models.py          # AI-specific database models (AIInsight)
│   ├── auth/                  # User authentication routes
│   ├── utils/
│   │   ├── analysis.py        # Statistical calculations (Sharpe, Drawdown)
│   │   └── helpers.py         # P&L calculations and image optimization
│   ├── models.py              # SQLAlchemy database models
│   └── routes.py              # Dashboard and AI API endpoints
├── migrations/                # Database migration history
├── static/
│   └── uploads/               # Trade images storage
├── config.py                  # Application configuration
├── run.py                     # Entry point
├── setup_simple.sh            # One-click setup script
└── requirements.txt           # Python dependencies

⚡ Quick Start
1. Prerequisites
 * Python 3.8+
 * pip
2. Installation
You can use the automated setup script (Linux/Termux) or install manually.
Option A: Automated Setup
chmod +x setup_simple.sh
./setup_simple.sh

This script installs Python, upgrades pip, installs dependencies, creates the .env file, and initializes the SQLite database.
Option B: Manual Installation
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Create Environment Variables
echo "SECRET_KEY=dev-key" > .env
echo "DATABASE_URL=sqlite:///forex_journal.db" >> .env

# 3. Initialize Database
python init_db_simple.py

3. Running the Application
python run.py

The server will start at http://localhost:5000 (or 0.0.0.0:5000 for mobile access).
🧪 Testing & Dummy Data
The application includes a sophisticated data generator to test the AI capabilities without risking real capital.
Generate Synthetic Data:
python dummy_data_generator.py

This script creates:
 * ~3000 realistic trades spanning 2 years.
 * Specific "Winning" and "Losing" patterns for the AI to detect.
 * Emotional streaks (e.g., "Anxious" trades leading to losses).
 * Account balance history.
 * A test user: test_trader / test123.
Run Tests:
python test_app.py

Validates database connections, imports, and analytical functions.
🧠 AI & Math Implementation Details
Sharpe Ratio Calculation
The application calculates the Sharpe Ratio to measure risk-adjusted return:
Where R_p is the portfolio return, R_f is the risk-free rate (assumed 0), and \sigma_p is the standard deviation of the portfolio's excess return.
Revenge Trading Detection
The ForexAI engine flags a trade as "Revenge Trading" if it meets specific criteria relative to the previous loss:
 * Entry time < 30 minutes from previous exit.
 * Position size > 1.5 \times previous trade.
 * Same symbol.
 * Emotional tags include "frustrated", "angry", or "impulsive".
Broker Import Logic
The system supports CSV imports and normalizes data. It specifically handles different asset classes:
 * Gold (XAU/USD): Pip size calculated as 0.01.
 * Silver (XAG/USD): Pip size calculated as 0.001.
 * JPY Pairs: Pip size calculated as 0.01.
🤝 Contributing
 * Fork the repository.
 * Create your feature branch (git checkout -b feature/AmazingFeature).
 * Commit your changes.
 * Push to the branch.
 * Open a Pull Request.
📝 License
Distributed under the MIT License.



## Reliability and deployment notes

1. Copy `.env.example` to `.env`, then set a unique `SECRET_KEY` and your real database/API/mail credentials. Never commit `.env`.
2. Set `COOKIE_SECURE=true` when serving over HTTPS. For local HTTP development only, set it to `false`.
3. Install dependencies with `pip install -r requirements.txt`.
4. Back up the database before applying any schema migration. Existing migrations and user data are intentionally preserved.
5. Run smoke tests with `python test_app.py` after dependencies are installed.
6. Configure `AI_MODEL_DIR` to a persistent writable directory shared by application workers if you want trained per-user models to survive restarts. Keep this directory private and backed up.
7. In production, run `python run_ai_scheduler.py` as one dedicated worker. Do not start a separate scheduler in every Gunicorn/web worker.

The personal AI uses a user's own closed-trade history. Its machine-learning estimates are retrospective correlations and should not be treated as guaranteed predictions or trading signals.


### Existing database preservation

The application now honours `SQLALCHEMY_DATABASE_URI` when explicitly configured. If it is not set, it checks the legacy SQLite location used by the previous version and uses it when that file exists; otherwise it uses a project-local SQLite file. Before deployment, confirm the path in your hosting environment and take a database backup. Do not point the updated application at a new empty database by accident.

### Security action required

The uploaded archive contained a live-looking environment configuration. It has been excluded from this cleaned package. Rotate the old Flask secret key and any database, email, or AI credentials that were present in that file. Put replacement values in your server's private environment configuration, not in source control.


## Multi-Timeframe Chart Intelligence

The AI Chart Lab is available from the navigation as **Chart Lab**. Upload four
screenshots for the same instrument: 4H, 1H, 30M and 15M. The vision analysis
reviews higher-timeframe bias, market structure, liquidity, BOS/CHoCH, order
blocks, fair value gaps, entry location, conflicts and invalidation. It stores
the structured observations for later comparison.

To teach the personal chart model from outcomes, link each saved chart review
to its corresponding completed journal trade. The system then compares the
feature profile of winning and losing examples. It withholds a personal numeric
win-probability estimate until there are at least 40 labelled chart reviews,
including at least 10 wins and 10 losses. The estimate is calibrated and
cross-validated, but is still a historical association, not a guarantee.

Before deploying this update:
1. Back up the database and uploaded files.
2. Ensure `GEMINI_API_KEY` is configured and `GEMINI_MODEL` supports image input.
3. Install dependencies from `requirements.txt`.
4. Run `flask --app manage db upgrade` to create `ai_chart_analyses`.
5. Restart the Flask application.

Chart screenshots are private to the account that uploaded them. Do not use
synthetic trade records as real training labels. A chart should only be linked
to the trade that actually resulted from that setup.

## Chart Intelligence and Performance Centre

The AI Chart Lab accepts four separate screenshots for the same instrument: **4H, 1H, 30M and 15M**. It asks the vision model for top-down market structure, liquidity, BOS/CHoCH, order-block/FVG quality, premium/discount location, confluences, conflicts and invalidation. Session-specific claims must be visible in the chart or supplied context; the model is instructed not to invent them.

The **Performance Intelligence Centre** is available at `/ai/performance`. It summarizes closed-trade results by symbol, strategy, direction and recorded entry hour, and compares chart-feature scores between labelled winners and losers. Entry-hour results use the timestamp as stored by the journal and are not automatically classified as London or New York sessions.

### Personal chart outcome model

Chart analyses become training examples only after they are linked to a completed journal trade with a positive or negative P&L. Breakeven trades are excluded from binary win/loss labels. The personal model requires at least 40 unique labelled trades and at least 10 wins and 10 losses. It uses cross-validation and reports balanced accuracy and Brier score. Until the threshold is met, the application withholds numerical win probabilities. A probability is a historical model estimate, not a guarantee or a substitute for risk management.

Before deployment, back up the database and upload directory, preserve the existing `.env`, install requirements, and run `flask --app manage db upgrade`.

### Adaptive SMC Playbook (Phase 3)

The Performance Intelligence Centre now compares chart conditions against your own realised trade outcomes. It reports weak/mixed/strong condition groups only after at least eight labelled trades are available in a group, compares them with your personal baseline, and shows a nine-factor confluence-score breakdown. These are historical associations, not causal rules, guaranteed edges, trade signals, or win probabilities. Asian/London session features are excluded from these findings unless session timing can be verified separately. The authenticated JSON endpoint is `/ai/smc-playbook`.


### SMC demo account (synthetic data)

Use a throwaway or development database, not your live journal.

```bash
flask --app manage db upgrade                  # creates ai_chart_analyses if needed
python smc_demo/generate_smc_demo.py           # optional: regenerate (default 30 setups, seed 2026)
python seed_smc_demo.py                        # creates user smc_demo (password printed; --password to choose)
python seed_smc_demo.py --remove               # deletes the demo user, trades, analyses and files
```

The bundled demo now holds **50 setups** (200 entry charts + 50 result charts). To grow it further without touching the existing setups, run `python smc_demo/generate_smc_demo.py --count N --append` (same seed), then re-run `python seed_smc_demo.py` (it purges and re-seeds only the demo user).

**Hand-entry test trades.** `python smc_demo/generate_manual_trades.py` rebuilds `smc_demo/manual_trades/`: three extra setups (A+, weak, mixed) that are *not* seeded. Follow `ENTRY_SHEET.md` to enter them yourself and upload each trade's four charts to Chart Lab; read `ANSWER_KEY.md` only afterwards to grade the AI.

Then log in as `smc_demo` and open **Chart Lab**, choose a setup and use *View analysis*, then open **Performance** for the SMC playbook.

What is real and what is not: the charts, trades and analyses are generated. The saved analyses are written directly in Chart Lab's JSON schema, so the live Gemini vision step is not exercised. The simulator ties win odds to setup quality, so any pattern the playbook or model finds is built into the data and says nothing about real markets. The personal chart model unlocks at 40 labelled trades (10 wins, 10 losses); even then, small synthetic samples give cross-validated metrics close to chance. Use `--count 150` on the generator for a larger set and check the reported metrics rather than a single probability.
