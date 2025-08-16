

## Multi-symbol Alpaca trading bot with **SMA (20/50)** + **Bollinger Bands (20, 1σ)** signals.  
Two modes:


- **Live** trading against Alpaca (paper by default).
- **Backtest** with an equity curve PNG + CSV and **Sharpe ratio**.

## Features
- Trades multiple symbols (e.g., `SPY`, `AAPL`, `MSFT`).
- Signals: SMA crossover first, then Bollinger breakout as fallback.
- Simple, transparent risk sizing:
  - Deploys `CashAtRisk` fraction of available cash per run.
  - Splits evenly across all active signals.
  - Caps per-symbol allocation via `PerSymbolCap`.
- Live orders via Alpaca REST.
- Backtesting (daily):
  - Loads local CSVs if present, otherwise pulls Alpaca Market Data (IEX).
  - Tracks cash, positions, equity, returns.
  - Outputs `backtest_equity.csv`, `backtest_equity.png`, and Sharpe ratio.
- Configurable via env vars and/or `config.yml`.

## Requirements
- Python 3.10+ (tested with `/usr/local/bin/python3` on macOS).
- Packages:
  ```bash
  pip install requests pandas pyyaml matplotlib
````

## Files

* `simplified_tradingbot.py` — the bot, signals, backtester, and entrypoints.
* `config.yml` — optional config (API keys, symbols, risk, backtest dates).
* Optional CSVs for backtests: `{SYMBOL}_ohlcv.csv` (e.g., `AAPL_ohlcv.csv`).

> The CSV loader supports:
>
> * Standard Yahoo-style headers.
> * Two-row header format:
>
>   * Row 0: `Price,Close,High,Low,Open,Volume`
>   * Row 1: first column has `Date`, others `Ticker,...`
>   * Data from row 2.

## Credentials & Endpoints

Set environment variables (recommended):

```bash
export APCA_API_KEY_ID="YOUR_KEY_ID"
export APCA_API_SECRET_KEY="YOUR_SECRET"
# paper trading host (default)
export APCA_API_BASE_URL="https://paper-api.alpaca.markets/v2"
# market data host (separate from trading)
export APCA_DATA_BASE_URL="https://data.alpaca.markets/v2"
# free tier usually needs IEX feed
export APCA_DATA_FEED="iex"
```

Or fill `config.yml`:

```yaml
alpaca:
  API_KEY: "YOUR_KEY_ID"
  API_SECRET: "YOUR_SECRET"
  BASE_URL: "https://paper-api.alpaca.markets/v2"
  DATA_BASE_URL: "https://data.alpaca.markets/v2"
```

> The code prefers env vars. If missing, it falls back to `config.yml`.

## Configuration (`config.yml`)

Minimal example:

```yaml
alpaca:
  API_KEY: ""     
  API_SECRET: ""
  BASE_URL: "https://paper-api.alpaca.markets/v2"
  DATA_BASE_URL: "https://data.alpaca.markets/v2"

Symbols:
  - SPY
  - AAPL
  - MSFT

# Live/backtest shared
CashAtRisk: 0.10          # fraction of available cash per run
PerSymbolCap: 0.50        # cap per symbol as fraction of available cash
LookbackBars: 200

# Backtest options
StartDate: "2024-01-01"
EndDate: null             # to “today”
InitialCash: 100000
RiskFreeAnnual: 0.00
CommissionPerShare: 0.00
BacktestCSV: "backtest_equity.csv"
Plot: true

# Execution preferences
AllowShorts: false
CloseOnOpposite: true
```

## Running

### Live trading (paper by default)

Places orders, prints signals, **no chart**:

```bash
/usr/local/bin/python3 /Users/qikunye/Documents/GitHub/tradingbot/simplified_tradingbot.py
```

Symbols and parameters can come from `config.yml` or be edited in the bottom “Live path” block.

### Backtest (chart + CSV + Sharpe)

Runs the backtester, shows chart, writes files alongside the script:

```bash
RUN_BACKTEST=1 /usr/local/bin/python3 /Users/qikunye/Documents/GitHub/tradingbot/simplified_tradingbot.py
```

Outputs:

* `backtest_equity.csv` (date, equity, ret)
* `backtest_equity.png` (equity curve with Sharpe in title)
* Console line with **Final Equity** and **Sharpe**


## Signals

* **SMA crossover** (20 fast / 50 slow):

  * `buy`: fast crosses above slow.
  * `sell`: fast crosses below slow.
* **Bollinger breakout** (20, 1σ):

  * `buy`: close > upper band.
  * `sell`: close < lower band.
* Priority: **SMA first, then BB** fallback.
* No naked shorts by default (`AllowShorts: false`).

### Make it more/less active

* Shorten SMA windows: e.g., 10/20.
* Change Bollinger `n_std` to 0.8–1.2.
* Switch priority (BB first, then SMA) where signals are combined.

## Sizing

* Available cash × `CashAtRisk` = total deployable.
* Split evenly across active signals.
* Per symbol capped by `PerSymbolCap` × available cash.
* Share quantity = floor(allocation\_cash / last\_price).

## Backtesting details

* Frequency: daily.
* Data source order:

  1. Local CSV `{SYMBOL}_ohlcv.csv` next to the script.
  2. Alpaca Market Data v2 (IEX feed by default).

### Portfolio

* Tracks cash and integer share positions.
* Optional short entries if `AllowShorts: true`.
* Closes opposite side first if `CloseOnOpposite: true`.

### Metrics

* Equity curve saved to PNG and CSV.
* Returns = daily pct change of equity.
* Sharpe ratio (annualized):

  ```
  Sharpe = ((mean(returns) - risk_free) / stdev(returns)) * sqrt(252)
  ```
* `RiskFreeAnnual` is converted to daily via:

  ```
  (1 + r_f)^(1/252) - 1
  ```



