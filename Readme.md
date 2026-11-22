
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


## User timezone support

Each account now has a saved IANA timezone preference. The initial value for existing and new accounts is `Africa/Nairobi`; users can change it under **Account menu → Timezone Settings**, or detect the timezone reported by their device. The preference is account-based and therefore follows the user across devices.

Database timestamps remain stored as naive UTC values for compatibility with the existing SQLite/SQLAlchemy schema. New trade and account-balance form timestamps are interpreted in the user's selected timezone and converted to UTC before storage. Existing records are not rewritten; legacy naive timestamps are displayed as UTC when converted to the user's timezone. If older records were originally entered in a non-UTC server timezone, review those records before relying on historical local-time analytics.

### Applying the database migration

Back up the database first, then deploy the updated files and run:

```bash
flask db upgrade
```

This adds `users.timezone` with a safe `Africa/Nairobi` default and does not delete or rewrite existing rows. If your deployment does not use Flask-Migrate, add a `VARCHAR(64)` `timezone` column to `users` with default `Africa/Nairobi` after taking a database backup.
