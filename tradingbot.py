import os, json, time, requests
import pandas as pd
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Config loading
# ------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parent

def _load_cfg(path=None):
    if path is None:
        path = REPO_DIR / "config.yml"
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}

# ------------------------------------------------------------
# Trader
# ------------------------------------------------------------

class SimpleSmaTrader:
    def __init__(
        self,
        symbols=("SPY",),
        cash_at_risk=0.10,
        per_symbol_cap=0.50,
        bar_days=120,
        cfg_path=None,
    ):
        """
        symbols: list/tuple or comma string (e.g. "SPY,AAPL,MSFT")
        cash_at_risk: fraction of AVAILABLE cash you can deploy per run (0.10 = 10%)
        per_symbol_cap: max fraction of available cash that can go into one symbol
        bar_days: how many daily bars to pull for SMA/BB calc
        cfg_path: optional override to a config.yml path
        """
        # ---- load config (optional) ----
        cfg = _load_cfg(cfg_path)
        alp = cfg.get("alpaca", {}) if isinstance(cfg, dict) else {}

        # -------- symbols ----------
        if isinstance(symbols, str):
            symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if not symbols:
            symbols = cfg.get("Symbols") or ["SPY", "AAPL", "MSFT"]
        self.symbols = sorted({str(s).upper() for s in symbols})

        # -------- risk knobs -------
        self.cash_at_risk = float(cash_at_risk if cash_at_risk is not None else cfg.get("CashAtRisk", 0.10))
        self.per_symbol_cap = float(per_symbol_cap)
        self.bar_days = int(bar_days)

        # -------- alpaca creds -----
        # Accept common env var styles, then fall back to config.yml
        env_key    = os.getenv("APCA_API_KEY_ID")     or os.getenv("APCA_API_KEY")
        env_secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET")
        env_base   = os.getenv("APCA_API_BASE_URL")

        self.api_key    = env_key    or alp.get("API_KEY")
        self.api_secret = env_secret or alp.get("API_SECRET")
        self.base_url   = (env_base or alp.get("BASE_URL") or "https://paper-api.alpaca.markets/v2").rstrip("/")

        # Market data host is different from trading host
        self.data_base_url = os.getenv("APCA_DATA_BASE_URL", "https://data.alpaca.markets/v2").rstrip("/")

        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing Alpaca creds. Set APCA_API_KEY_ID / APCA_API_SECRET_KEY or fill alpaca.API_KEY / alpaca.API_SECRET in config.yml.")

        # ---- shared HTTP session with auth headers ----
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        # Optional execution knobs
        self.allow_shorts = False
        self.close_on_opposite = True

    # --------------------- Alpaca helpers ---------------------

    def _get_account(self):
        r = self.session.get(f"{self.base_url}/account", timeout=30)
        r.raise_for_status()
        return r.json()

    def _get_cash(self):
        acct = self._get_account()
        return float(acct.get("cash", 0.0))

    def _get_position_qty(self, symbol):
        r = self.session.get(f"{self.base_url}/positions/{symbol}", timeout=30)
        if r.status_code == 404:
            return 0.0
        r.raise_for_status()
        j = r.json()
        # Alpaca returns qty as a string; positive for long. Short detection uses j.get("side").
        return float(j.get("qty", 0)) if isinstance(j, dict) else 0.0

    def _submit_market_order(self, symbol, side, qty):
        payload = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        r = self.session.post(f"{self.base_url}/orders", data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        return r.json()

    def _get_daily_bars(self, symbol, limit=120):
        """
        Pull recent daily bars from Alpaca MARKET DATA v2 (IEX feed by default).
        """
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=limit * 2)  # pad for weekends/holidays

        url = f"{self.data_base_url}/stocks/{symbol}/bars"
        params = {
            "timeframe": "1Day",
            "start": f"{start}T00:00:00Z",
            "end":   f"{end}T23:59:59Z",
            "limit": limit,
            "adjustment": "raw",
            "feed": os.getenv("APCA_DATA_FEED", "iex"),
        }

        r = self.session.get(url, params=params, timeout=30)

        if r.status_code == 403:
            raise RuntimeError(
                "403 from market data. Your account likely lacks access to the requested feed.\n"
                "• If you're on the free/paper tier, use feed='iex' (already set).\n"
                "• If you need SIP, upgrade your data subscription in the Alpaca dashboard.\n"
                "• You can also set APCA_DATA_FEED=iex in your environment."
            )
        if r.status_code == 404:
            params.pop("feed", None)
            r = self.session.get(url, params=params, timeout=30)

        r.raise_for_status()
        data = r.json().get("bars", [])
        if not data:
            return None

        df = pd.DataFrame(data).rename(columns={"t":"datetime","o":"open","h":"high","l":"low","c":"close","v":"volume"})
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df.dropna().sort_values("datetime").set_index("datetime")
        keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
        return df[keep] if len(keep) >= 4 else None

    # --------------------- Signals & sizing ---------------------

    def _sma_signal(self, df, fast=20, slow=50, verbose=True):
        """SMA crossover: fast over slow = BUY, fast under slow = SELL."""
        if df is None or df.empty or len(df) < max(fast, slow) + 1:
            return None, None

        closes = df["close"].astype(float)
        sma_f = closes.rolling(fast).mean()
        sma_s = closes.rolling(slow).mean()

        bull = sma_f.iloc[-2] < sma_s.iloc[-2] and sma_f.iloc[-1] > sma_s.iloc[-1]
        bear = sma_f.iloc[-2] > sma_s.iloc[-2] and sma_f.iloc[-1] < sma_s.iloc[-1]
        last = float(closes.iloc[-1])

        if verbose:
            print(
                f"    SMA fast[{fast}] last={sma_f.iloc[-1]:.4f} prev={sma_f.iloc[-2]:.4f} | "
                f"slow[{slow}] last={sma_s.iloc[-1]:.4f} prev={sma_s.iloc[-2]:.4f} | close={last:.4f}"
            )

        if bull:
            print("    signal=SMA cross ↑")
            return "buy", last
        if bear:
            print("    signal=SMA cross ↓")
            return "sell", last
        return None, last

    def _bb_signal(self, df, period=20, n_std=1.0, verbose=True):
        """Bollinger breakout (1σ): close > upper = BUY, close < lower = SELL."""
        if df is None or df.empty or len(df) < period + 1:
            return None, None

        closes = df["close"].astype(float)
        ma = closes.rolling(period).mean()
        sd = closes.rolling(period).std()
        upper = ma + n_std * sd
        lower = ma - n_std * sd

        last_close = float(closes.iloc[-1])
        up = float(upper.iloc[-1])
        lo = float(lower.iloc[-1])

        if verbose:
            print(
                f"    BB(period={period}, n={n_std}) | close={last_close:.4f}, mid={ma.iloc[-1]:.4f}, "
                f"upper={up:.4f}, lower={lo:.4f}"
            )

        if last_close > up:
            print("    signal=BB breakout ↑")
            return "buy", last_close
        if last_close < lo:
            print("    signal=BB breakout ↓")
            return "sell", last_close
        return None, last_close

    def _size_allocation(self, available_cash, n_active):
        """Split available risk cash across active signals, with a per-symbol cap."""
        if n_active <= 0:
            return 0.0
        total_risk_cash = max(0.0, available_cash * self.cash_at_risk)
        per = total_risk_cash / n_active
        cap = available_cash * self.per_symbol_cap
        return max(0.0, min(per, cap))

    def _qty_for_allocation(self, allocation_cash, last_price):
        if not (allocation_cash > 0 and last_price > 0):
            return 0
        return int(allocation_cash // last_price)

    # --------------------- Live entrypoint ---------------------

    def run_once(self):
        """
        Fetch bars for each symbol, compute signals (SMA cross, then BB breakout),
        split risk across active signals, and place market orders.
        """
        signals = []   # list of (symbol, side, last_price)

        for sym in self.symbols:
            try:
                df = self._get_daily_bars(sym, limit=self.bar_days)
                if df is None or df.empty:
                    print(f"[{sym}] Skip: no bars.")
                    continue

                # Signals (SMA first, then BB)
                side, last_px = self._sma_signal(df, fast=20, slow=50, verbose=True)
                src = "SMA"
                if side is None:
                    side, last_px = self._bb_signal(df, period=20, n_std=1.0, verbose=True)
                    src = "BB" if side is not None else src

                if side is not None:
                    print(f"[{sym}] → {side.upper()} via {src}")
                    signals.append((sym, side, last_px))
                else:
                    print(f"[{sym}] Neutral (no SMA cross / no BB 1σ breakout).")

            except Exception as e:
                print(f"[{sym}] Error fetching/signal: {e}")

        if not signals:
            print("No active signals.")
            return

        # Risk split
        try:
            available_cash = self._get_cash()
        except Exception as e:
            print(f"Account/cash error: {e}")
            return

        allocation_per = self._size_allocation(available_cash, len(signals))
        if allocation_per <= 0:
            print("No deployable cash based on risk settings.")
            return

        # Execute orders
        for sym, side, last_px in signals:
            try:
                qty_held = self._get_position_qty(sym)
                if side == "sell" and qty_held <= 0 and not self.allow_shorts:
                    print(f"[{sym}] Sell signal but no long position; skipping to avoid naked short.")
                    continue
                if side == "buy" and qty_held > 0:
                    print(f"[{sym}] Already long ({qty_held}); skipping duplicate buy.")
                    continue

                qty = self._qty_for_allocation(allocation_per, last_px)
                if qty <= 0:
                    print(f"[{sym}] Allocation too small for 1 share at {last_px:.2f}.")
                    continue

                order = self._submit_market_order(sym, side, qty)
                oid = order.get("id", "?")
                print(f"[{sym}] {side.upper()} {qty} @ ~{last_px:.2f}  (order_id={oid})")
                time.sleep(0.25)

            except requests.HTTPError as e:
                try:
                    detail = e.response.json()
                except Exception:
                    detail = e.response.text
                print(f"[{sym}] Order error: {detail}")
            except Exception as e:
                print(f"[{sym}] Unexpected error: {e}")

# ------------------------------------------------------------
# Backtesting helpers (module-level)
# ------------------------------------------------------------

def _load_csv_ohlcv(symbol: str, base_dir: Path) -> "pd.DataFrame | None":
    """
    Robust CSV loader for files like AAPL_ohlcv.csv that have a 2-row header:
    Row0: Price,Close,High,Low,Open,Volume
    Row1: Ticker,AAPL,... and 'Date' in the first column.
    Also supports standard Yahoo-style headers.
    """
    p = base_dir / f"{symbol}_ohlcv.csv"
    if not p.exists():
        return None

    # sniff first two lines
    sniff = []
    with p.open("r", newline="") as f:
        try:
            sniff.append(next(f))
            sniff.append(next(f))
        except StopIteration:
            pass

    def looks_like_two_row_header(lines):
        if len(lines) < 2:
            return False
        first = lines[0].strip().lower()
        second = lines[1].split(",")[0].strip().lower()
        return first.startswith("price,close,high,low,open,volume") and second in ("ticker", "date")

    if looks_like_two_row_header(sniff):
        df = pd.read_csv(p, skiprows=2)
        rename_map = {
            "Date": "datetime",
            "Unnamed: 1": "close",
            "Unnamed: 2": "high",
            "Unnamed: 3": "low",
            "Unnamed: 4": "open",
            "Unnamed: 5": "volume",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    else:
        df = pd.read_csv(p)
        # datetime normalization
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "datetime"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        elif "Price" in df.columns:
            df = df.rename(columns={"Price": "datetime"})
        elif df.columns and df.columns[0].lower().startswith("unnamed"):
            df = df.rename(columns={df.columns[0]: "datetime"})
        # OHLCV names
        canonical_map = {
            "Open": "open", "High": "high", "Low": "low", "Close": "close",
            "Adj Close": "close", "Volume": "volume",
            "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
        }
        df = df.rename(columns={k: v for k, v in canonical_map.items() if k in df.columns})
        # fallback unnamed
        if "close" not in df.columns and "Unnamed: 1" in df.columns:
            df = df.rename(columns={"Unnamed: 1": "close"})
        if "high" not in df.columns and "Unnamed: 2" in df.columns:
            df = df.rename(columns={"Unnamed: 2": "high"})
        if "low" not in df.columns and "Unnamed: 3" in df.columns:
            df = df.rename(columns={"Unnamed: 3": "low"})
        if "open" not in df.columns and "Unnamed: 4" in df.columns:
            df = df.rename(columns={"Unnamed: 4": "open"})
        if "volume" not in df.columns and "Unnamed: 5" in df.columns:
            df = df.rename(columns={"Unnamed: 5": "volume"})

    if "datetime" not in df.columns:
        return None

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "open", "high", "low", "close"])
          .sort_values("datetime")
          .set_index("datetime")
    )
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols]

def _get_daily_bars_range(trader: SimpleSmaTrader, symbol: str, start_ts, end_ts) -> "pd.DataFrame | None":
    """
    Use Alpaca Market Data v2 for a date range (timezone-aware UTC datetimes).
    """
    url = f"{trader.data_base_url}/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Day",
        "start": start_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end":   end_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 10000,
        "adjustment": "raw",
        "feed": os.getenv("APCA_DATA_FEED", "iex"),
    }
    r = trader.session.get(url, params=params, timeout=30)
    if r.status_code == 403:
        raise RuntimeError("403 from data API. Set APCA_DATA_FEED=iex or upgrade data plan.")
    if r.status_code == 404:
        params.pop("feed", None)
        r = trader.session.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("bars", [])
    if not data:
        return None
    df = pd.DataFrame(data).rename(columns={"t":"datetime","o":"open","h":"high","l":"low","c":"close","v":"volume"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna().sort_values("datetime").set_index("datetime")
    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    return df[keep] if len(keep) >= 4 else None

# ------------------------------------------------------------
# Backtest runner (equity curve + Sharpe + CSV/PNG)
# ------------------------------------------------------------

def backtest(
    trader: SimpleSmaTrader,
    start="2024-01-01",
    end=None,
    initial_cash=100_000.0,
    rf_annual=0.00,             # annual risk-free (e.g., 0.02 = 2%)
    commission_per_share=0.00,  # optional
    save_csv="backtest_equity.csv",
    plot=True
):
    from datetime import datetime as _dt

    start_ts = pd.Timestamp(start).tz_localize("UTC") if isinstance(start, str) else pd.Timestamp(start, tz="UTC")
    end_ts = (pd.Timestamp(_dt.now(timezone.utc)) if end is None
              else (pd.Timestamp(end).tz_localize("UTC") if isinstance(end, str) else pd.Timestamp(end, tz="UTC")))

    # 1) Load data per symbol (CSV first, then Alpaca)
    all_data = {}
    for sym in trader.symbols:
        df = _load_csv_ohlcv(sym, REPO_DIR)
        if df is None:
            try:
                df = _get_daily_bars_range(trader, sym, start_ts, end_ts)
            except Exception as e:
                print(f"[{sym}] Data error: {e}")
                df = None
        if df is None:
            print(f"[{sym}] No data available; skipping.")
            continue
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        if df.empty:
            print(f"[{sym}] No bars in range; skipping.")
            continue
        all_data[sym] = df

    if not all_data:
        print("No data for any symbol.")
        return None, float("nan")

    # 2) Unified calendar
    calendar = sorted(set().union(*[df.index.date for df in all_data.values()]))
    calendar = pd.to_datetime(calendar).tz_localize("UTC")

    # 3) Portfolio state
    cash = float(initial_cash)
    pos = {sym: 0 for sym in all_data.keys()}  # integer shares
    equity_series = []

    def portfolio_value(ts):
        val = cash
        for sym, qty in pos.items():
            if qty != 0:
                sub = all_data[sym].loc[all_data[sym].index <= ts]
                if not sub.empty:
                    val += qty * float(sub["close"].iloc[-1])
        return val

    # 4) Daily loop
    for day in calendar:
        day = pd.Timestamp(day).tz_convert("UTC")

        todays_signals = []
        for sym, df in all_data.items():
            up_to = df.loc[df.index <= day]
            if len(up_to) < 60:
                continue
            side, last_px = trader._sma_signal(up_to, fast=20, slow=50, verbose=False)
            if side is None:
                side, last_px = trader._bb_signal(up_to, period=20, n_std=1.0, verbose=False)
            if side is not None:
                todays_signals.append((sym, side, float(last_px)))

        if not todays_signals:
            equity_series.append({"date": day, "equity": portfolio_value(day)})
            continue

        allocation_per = trader._size_allocation(cash, len(todays_signals))

        for sym, side, price in todays_signals:
            if price <= 0:
                continue

            qty_held = pos.get(sym, 0)

            # Close opposite if enabled
            if trader.close_on_opposite:
                if qty_held > 0 and side == "sell":
                    trade_qty = int(qty_held)
                    cash += trade_qty * price - commission_per_share * trade_qty
                    pos[sym] = 0
                elif qty_held < 0 and side == "buy":
                    trade_qty = int(abs(qty_held))
                    cash -= trade_qty * price + commission_per_share * trade_qty
                    pos[sym] = 0

            # Entries
            if side == "buy":
                if pos[sym] <= 0:
                    qty = trader._qty_for_allocation(allocation_per, price)
                    if qty > 0:
                        cost = qty * price + commission_per_share * qty
                        if cost <= cash:
                            cash -= cost
                            pos[sym] += qty

            elif side == "sell":
                if pos[sym] > 0:
                    # already flattened above if close_on_opposite=True
                    pass
                elif pos[sym] == 0 and trader.allow_shorts:
                    qty = trader._qty_for_allocation(allocation_per, price)
                    if qty > 0:
                        proceeds = qty * price - commission_per_share * qty
                        cash += proceeds
                        pos[sym] -= qty

        equity_series.append({"date": day, "equity": portfolio_value(day)})

    # 5) Metrics & chart
    eq = pd.DataFrame(equity_series).set_index("date").sort_index()
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)

    rf_daily = (1.0 + rf_annual) ** (1/252) - 1.0
    excess = eq["ret"] - rf_daily
    mu = excess.mean()
    sd = excess.std(ddof=1)
    sharpe = (mu / sd) * math.sqrt(252) if sd > 0 else float("nan")

    print(
        f"Backtest complete: start={start_ts.date()} end={end_ts.date()} "
        f"| Final Equity=${eq['equity'].iloc[-1]:,.2f} | Sharpe={sharpe:.3f}"
    )

    if save_csv:
        eq.to_csv(save_csv, index=True)
        print(f"Saved equity & returns to {save_csv}")

    if plot:
        ax = eq["equity"].plot(title=f"Equity Curve (Sharpe {sharpe:.2f})")
        plt.xlabel("Date"); plt.ylabel("Equity ($)")
        plt.tight_layout()
        out_png = REPO_DIR / "backtest_equity.png"
        fig = ax.get_figure()
        fig.savefig(out_png, dpi=144)
        print(f"Saved chart to {out_png}")
        plt.show()

    return eq, sharpe

# ------------------------------------------------------------
# Config-driven helpers
# ------------------------------------------------------------

def make_trader_from_config(cfg_path=None) -> SimpleSmaTrader:
    cfg = _load_cfg(cfg_path)
    syms = cfg.get("Symbols") or ["SPY", "AAPL", "MSFT"]
    risk = float(cfg.get("CashAtRisk", 0.10))
    bar_days = int(cfg.get("LookbackBars", 200))
    per_cap = float(cfg.get("PerSymbolCap", 0.50))

    t = SimpleSmaTrader(
        symbols=[str(s).upper() for s in syms],
        cash_at_risk=risk,
        per_symbol_cap=per_cap,
        bar_days=bar_days,
        cfg_path=cfg_path,
    )
    t.allow_shorts = bool(cfg.get("AllowShorts", False))
    t.close_on_opposite = bool(cfg.get("CloseOnOpposite", True))
    return t

def run_backtest_from_config(cfg_path=None):
    cfg = _load_cfg(cfg_path)
    start = cfg.get("StartDate", "2024-01-01")
    end = cfg.get("EndDate", None)
    initial = float(cfg.get("InitialCash", 100_000))
    rf = float(cfg.get("RiskFreeAnnual", 0.00))
    comm = float(cfg.get("CommissionPerShare", 0.00))
    out_csv = cfg.get("BacktestCSV", "backtest_equity.csv")
    do_plot = bool(cfg.get("Plot", True))

    trader = make_trader_from_config(cfg_path)
    eq, sharpe = backtest(
        trader,
        start=start,
        end=end,
        initial_cash=initial,
        rf_annual=rf,
        commission_per_share=comm,
        save_csv=out_csv,
        plot=do_plot,
    )
    print(f"Sharpe: {sharpe:.3f}")
    return eq, sharpe

# ------------------------------------------------------------
# Entrypoints
# ------------------------------------------------------------

# Backtest path (produces CSV + chart)
if os.getenv("RUN_BACKTEST") == "1":
    run_backtest_from_config()

# Live path (places orders; no chart)
else:
    trader = SimpleSmaTrader(
        symbols=["SPY", "AAPL", "MSFT"],
        cash_at_risk=0.10,
        per_symbol_cap=0.50,
        bar_days=200,
    )
    trader.allow_shorts = False
    trader.close_on_opposite = True
    trader.run_once()
