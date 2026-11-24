# Forex Journal – Improvement Notes

## Reliability and security
- Fixed persistent Remember Me cookie configuration and made its duration configurable.
- Preserved the `next` login destination while rejecting external redirect destinations.
- Changed logout to a CSRF-validated POST action.
- Removed the machine-specific database override and added legacy SQLite-path detection.
- Removed the secret-bearing `.env` file from the distributable; added `.env.example`.
- Made production startup fail clearly when `SECRET_KEY` is missing.
- Made blueprint registration fail visibly instead of silently launching a partial application.

## AI learning
- Retrained the per-user classifier only on closed trades.
- Added class-diversity checks and stratified holdout validation.
- Added balanced class weights, baseline accuracy and model metadata.
- Made model storage configurable and persistent instead of relying on `/tmp`.
- Added safe JSON conversion for NumPy values and more defensive AI API handling.
- Removed expensive full AI analysis from ordinary trade-list page loads.
- Added a dedicated single-instance AI scheduler entry point.

## Analytics and uploads
- Fixed timeframe analytics so every trade is counted and monthly grouping is populated.
- Normalized metric output to JSON-safe Python numeric types.
- Hardened numeric trade validation against malformed, non-finite and non-positive values.
- Validated uploaded image contents, normalized image output to JPEG and avoided broken database rows when an upload is invalid.

## Interface
- Added consistent dark surfaces, spacing, borders, form styling, responsive layout and refined navigation styling.

## Verification
- All Python source files compile.
- All Jinja HTML templates parse.
- Timeframe aggregation and trade metric calculations were exercised with synthetic trades.
- Numeric trade validation was exercised with valid, malformed and non-finite values.
- Full Flask integration tests are included in `test_app.py`; they must be run after installing dependencies in the target environment.

## Multi-timeframe chart intelligence
- Added a four-image 4H/1H/30M/15M top-down Chart Lab using Gemini vision.
- Added structured market-structure, liquidity, BOS/CHoCH, order-block, FVG,
  premium/discount and Asian-range/London-sweep observations.
- Persisted each chart review and linked it to its real journal trade outcome.
- Added win-versus-loss chart-factor comparisons and personal calibrated model
  estimates only after 40 unique labelled trades with at least 10 wins and 10 losses.
- Added cross-validated balanced-accuracy and Brier-score reporting, per-user
  model caching, and duplicate-trade protection against evaluation leakage.
- Added a database migration and Chart Lab navigation entry.

## Trading Performance Intelligence Centre
- Added `/ai/performance` with personal P&L, win rate, profit factor, expectancy and breakeven-aware outcome counts.
- Added performance breakdowns by symbol, strategy, direction and recorded entry hour.
- Added a comparison of chart-feature averages for linked winning and losing trades.
- Added model-readiness and cross-validation diagnostics.
- Breakeven trades are excluded from the binary win/loss training labels rather than being silently classified as losses.

## Phase 3 — Adaptive SMC Playbook

- Added a per-user SMC playbook built from the latest chart analysis linked to each completed, non-breakeven trade.
- Groups observed SMC conditions into weak, mixed and strong bands and compares each sufficiently populated group with the user's own baseline win rate and average P&L.
- Hides condition groups with fewer than eight trades and labels results as associations rather than causal rules.
- Adds a transparent nine-factor SMC confluence score breakdown (low / medium / high); it is explicitly not a win probability or entry signal.
- Excludes Asian/London/session fields from playbook edge findings because screenshot timestamps may not establish session boundaries.
- Adds authenticated JSON endpoint `/ai/smc-playbook` and displays the results in the Performance Intelligence Centre.
