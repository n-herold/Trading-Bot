# Trading Bot V3

Automated paper trading bot in Python for 7 markets simultaneously.
No broker required — data via yfinance, trade logging via SQLite.

---

## Overview

Trading Bot V3 is the result of iterative development across multiple versions:

- **V1/V2** — Basic EMA crossover, BOS/CHoCH structure, partial close + trailing stop
- **V3** — Simplified and improved: pure SL/TP, per-symbol strategy routing, consistent backtest-to-live logic

The key insight from V2 → V3: partial close + trailing stop compressed effective RR to ~1:1, making the strategy unprofitable at 33% win rate. Removing them and fixing the RR at 3:1 pushed Profit Factor from 0.98 → 1.26.

---

## Markets & Strategies

| Symbol | Market | Strategy |
|--------|--------|----------|
| EURUSD=X | EUR/USD | EMA Pullback |
| GBPUSD=X | GBP/USD | Mean Reversion |
| USDJPY=X | USD/JPY | Breakout |
| AUDUSD=X | AUD/USD | Mean Reversion |
| GC=F | Gold | Breakout |
| BTC-USD | Bitcoin | EMA Pullback |
| NQ=F | Nasdaq | EMA Pullback |

### EMA Pullback
Trend aligned (EMA20 > EMA50 > EMA200 for Long). Entry when previous candle touches EMA20 and current candle closes back above. ADX ≥ 20, RSI 35–75.

### Mean Reversion
Ranging market (ADX ≤ 35). Long when previous candle closes below lower Bollinger Band and current returns above. RSI < 45 / > 55. TP at BB midline.

### Breakout
Long when close exceeds 20-candle rolling high of previous candle. Short below rolling low. ADX ≥ 20. ATR-based SL.

---

## Parameters

```python
RISK_PER_TRADE = 0.01       # 1% risk per trade

STRATEGY_TP = {
    "ema_pullback":   3.0,   # 3R
    "mean_reversion": 1.5,   # TP at BB midline (~1.5R effective)
    "breakout":       3.0,   # 3R
}

SL_ATR_MULT    = 2.0        # Stop Loss = 2× ATR (breakout)
                             # ema_pullback/mean_reversion: prev candle low/high ± 0.5× ATR

MAX_LOSSES_ROW = 3          # Circuit breaker: 3 consecutive losses
MAX_DAILY_LOSS = 0.02       # Circuit breaker: 2% daily loss

SYMBOL_BB = {
    "AUDUSD=X": (14, 2.0),  # faster BB → better mean reversion signals
    # all others: BB(20, 2.0)
}
```

---

## Backtest Results (2 years, 1H data)

| Symbol | Strategy | Return | Trades | Win Rate | Profit Factor |
|--------|----------|--------|--------|----------|---------------|
| EUR/USD | EMA Pullback | +27.9% | 221 | 28.5% | 1.15 |
| GBP/USD | Mean Reversion | +16.5% | 186 | 44.1% | 1.14 |
| USD/JPY | Breakout | +94.3% | 231 | 32.9% | 1.40 |
| AUD/USD | Mean Reversion | +12.9% | 196 | 41.8% | 1.11 |
| Gold | Breakout | +45.3% | 198 | 30.3% | 1.28 |
| Bitcoin | EMA Pullback | +62.8% | 239 | 30.5% | 1.31 |
| Nasdaq | EMA Pullback | +41.6% | 160 | 31.2% | 1.35 |
| **Combined** | | **Ø +43.0%** | **1431** | **34.0%** | **1.26** |

~2–3 trades/day across all 7 symbols.

> ⚠️ Backtest results do not guarantee future performance.

---

## Quick Start

### Requirements

```bash
pip install yfinance pandas numpy requests
```

### Run

```python
# Open trading_bot_v3.py in Spyder → F5, then:

# Backtest all 7 symbols
run_multi_backtest()

# Backtest single symbol
run_backtest("GC=F")

# Paper trading (hourly scans, runs until Ctrl+C)
run_paper_trading_v2()

# Short test (3 scans, 10s interval)
run_paper_trading_v2(max_scans=3, scan_interval=10)

# Compare strategies on one symbol
compare_strategies("EURUSD=X")
```

---

## Architecture

```
trading_bot_v3.py
├── Module 1  — Data loading (yfinance, 1H/4H/Daily/Weekly)
├── Module 2  — BOS/CHoCH trend structure detection
├── Module 3  — Signal generator (9 checks)
├── Module 4  — News filter (Finnhub API, blackout ±60min)
├── Module 5  — Risk manager (position sizing, circuit breaker)
├── Module 6  — Backtesting engine
├── Module 7  — Paper trading (run_paper_trading_v2)
├── Module 8  — Walk-forward optimization (Optuna)
├── Module 9  — Telegram alerts
└── Module 10 — V3 features (VIX filter, DXY, candlestick patterns)
```

### Paper Trading Logic (`run_paper_trading_v2`)
The paper trading loop mirrors the backtest exactly — same entry logic, same indicators, same SL/TP. Each hourly scan checks the last completed 1H candle via `get_backtest_signal()`. Tracks `last_candle_ts` per symbol to avoid double signals. Capital persists across sessions via SQLite.

---

## Data & Persistence

Trades are logged to SQLite:
```
logs/trades_v2.db
```

On restart, the bot loads the last capital from the DB automatically — no data loss between sessions.

---

## Key Design Decisions

**Pure SL/TP — no partial close, no trailing stop**
V2 had 50% partial close at 1.5R + trailing stop (EMA20). This compressed effective RR to ~1:1. With ~33% win rate, profitable trading requires RR ≥ 2:1. After removing: PF 0.98 → 1.26.

**Per-symbol strategy routing**
EUR/USD with ema_pullback only was -16%. Mean reversion fits ranging pairs (GBP/USD, AUD/USD) far better. Breakout for USD/JPY and Gold: +94% and +45%.

**15min tested, 1H kept**
15min backtest: combined PF 1.04 vs 1H PF 1.26. Breakout on 15min collapses due to noise (USD/JPY: +94% → -24%). 1H remains the entry timeframe.

**AUD/USD: BB(14) instead of BB(20)**
Shorter Bollinger Bands react faster → better mean reversion entries. +12.9% vs +11.8% with BB(20).

**Backtest-to-live consistency**
Earlier versions used different signal logic for paper trading vs backtesting (BOS/CHoCH in paper, not in backtest). V3 extracts entry logic into `get_backtest_signal()` used by both — what you backtest is what runs live.

---

## Go-Live Plan

- [x] Backtest validation (2 years, all 7 symbols)
- [ ] Paper trading 4–6 weeks (started ~10.06.2026)
- [ ] Compare live results to backtest
- [ ] Live trading with 200–500€
- [ ] VPS (Hetzner CX11 ~5€/month) for 24/7 operation

---

## Tech Stack

- Python 3.12 (Anaconda / Spyder)
- yfinance — market data
- pandas / numpy — calculations
- sqlite3 — trade logging
- requests — API calls
- Telegram Bot — trade alerts
- Finnhub — high-impact news filter
- Optuna — walk-forward optimization (planned)
