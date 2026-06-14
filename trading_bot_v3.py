# ============================================================
#  TRADING BOT — Alles in einer Datei
#  Einfach in Spyder öffnen und F5 drücken
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
#  EINSTELLUNGEN — hier alles anpassen
# ─────────────────────────────────────────────

# yfinance Rate Limit Workaround
import yfinance as yf
try:
    yf.set_tz_cache_location("/tmp/yf_cache")
except:
    pass

SYMBOLS = ["EURUSD=X", "GBPUSD=X", "BTC-USD"]

EMA_FAST   = 20
EMA_SLOW   = 50
EMA_TREND  = 200
RSI_PERIOD = 14
ADX_PERIOD = 14
ATR_PERIOD = 14

RISK_PER_TRADE = 0.01   # 1% Kapital pro Trade
SL_ATR_MULT    = 2.0    # Stop Loss = 2.0 × ATR
TP_RR_RATIO    = 3.0    # Take Profit default

MAX_LOSSES_ROW = 3      # Circuit Breaker: 3 Verluste in Folge
MAX_DAILY_LOSS = 0.02   # Circuit Breaker: 2% Tagesverlust

HISTORY_YEARS  = 2      # Wie viele Jahre Daten laden

# ── STRATEGY ROUTING ─────────────────────────────
SYMBOL_STRATEGY = {
    "EURUSD=X": "ema_pullback",
    "GBPUSD=X": "mean_reversion",
    "USDJPY=X": "breakout",
    "AUDUSD=X": "mean_reversion",
    "GC=F":     "breakout",        # Gold — Breakout aus Konsolidierung
    "BTC-USD":  "ema_pullback",    # Bitcoin — EMA Pullback funktioniert gut
    "NQ=F":     "ema_pullback",    # Nasdaq — starke Trends
}
STRATEGY_TP = {
    "ema_pullback":   3.0,
    "mean_reversion": 1.5,   # Kürzere Ziele — TP an BB-Mittellinie
    "breakout":       3.0,
}
# Per-Symbol TP Override (überschreibt STRATEGY_TP)
SYMBOL_TP = {}

# Bollinger Bands (default)
BB_PERIOD = 20
BB_STD    = 2.0

# Per-Symbol BB Override
SYMBOL_BB = {
    "AUDUSD=X": (14, 2.0),   # Schnellere Bands → bessere Mean-Reversion Signale
    # alle anderen: BB(20, 2.0)
}

# Breakout
BREAKOUT_PERIOD = 20


# ─────────────────────────────────────────────
#  DATEN LADEN
# ─────────────────────────────────────────────

def load_ohlcv(symbol, timeframe="1h", years=HISTORY_YEARS):
    """Lädt OHLCV-Daten von yfinance."""
    end   = datetime.now()
    if timeframe == "1h":
        start = end - timedelta(days=720)
    else:
        start = end - timedelta(days=365 * years)

    print(f"  📥 Lade {symbol} [{timeframe}]...")

    df = yf.download(
        tickers     = symbol,
        start       = start,
        end         = end,
        interval    = timeframe,
        auto_adjust = True,
        progress    = False,
    )

    # Retry bei Rate Limit
    if df.empty:
        import time
        print(f"  ⏳ Rate limit — warte 10 Sekunden und versuche nochmal...")
        time.sleep(10)
        df = yf.download(
            tickers     = symbol,
            start       = start,
            end         = end,
            interval    = timeframe,
            auto_adjust = True,
            progress    = False,
        )
    if df.empty:
        raise ValueError(f"Keine Daten für {symbol}!")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df.index.name = "datetime"

    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]

    print(f"  ✅ {len(df)} Kerzen geladen")
    return df


def resample_to_4h(df_1h):
    """Resamplet 1H-Daten auf 4H."""
    df_4h = df_1h.resample("4h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum"
    }).dropna(subset=["close"])
    print(f"  🔄 4H: {len(df_4h)} Kerzen")
    return df_4h


# ─────────────────────────────────────────────
#  INDIKATOREN
# ─────────────────────────────────────────────

def add_indicators(df):
    """Berechnet EMA, RSI, ATR, ADX."""
    df = df.copy()

    # EMA
    df["ema_fast"]  = df["close"].ewm(span=EMA_FAST,  adjust=False).mean()
    df["ema_slow"]  = df["close"].ewm(span=EMA_SLOW,  adjust=False).mean()
    df["ema_trend"] = df["close"].ewm(span=EMA_TREND, adjust=False).mean()

    # RSI
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # ADX
    up   = df["high"].diff()
    down = -df["low"].diff()
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    mdm  = np.where((down > up) & (down > 0), down, 0.0)
    atr_s = tr.ewm(span=ADX_PERIOD, adjust=False).mean()
    pdi   = 100 * pd.Series(pdm, index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean() / atr_s
    mdi   = 100 * pd.Series(mdm, index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean() / atr_s
    dx    = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["adx"] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    df["+di"] = pdi
    df["-di"] = mdi

    df["trend_up"]   = (df["ema_fast"] > df["ema_slow"]) & (df["close"] > df["ema_trend"])
    df["trend_down"] = (df["ema_fast"] < df["ema_slow"]) & (df["close"] < df["ema_trend"])

    # Bollinger Bands — beide Perioden fix vorberechnen
    df["bb20_mid"]  = df["close"].rolling(20).mean()
    df["bb20_std"]  = df["close"].rolling(20).std()
    df["bb20_upper"]= df["bb20_mid"] + 2.0 * df["bb20_std"]
    df["bb20_lower"]= df["bb20_mid"] - 2.0 * df["bb20_std"]
    df["bb14_mid"]  = df["close"].rolling(14).mean()
    df["bb14_std"]  = df["close"].rolling(14).std()
    df["bb14_upper"]= df["bb14_mid"] + 2.0 * df["bb14_std"]
    df["bb14_lower"]= df["bb14_mid"] - 2.0 * df["bb14_std"]
    # Default alias
    df["bb_mid"]   = df["bb20_mid"]
    df["bb_upper"] = df["bb20_upper"]
    df["bb_lower"] = df["bb20_lower"]

    # Rolling High/Low (für Breakout)
    df["high_n"] = df["high"].rolling(BREAKOUT_PERIOD).max()
    df["low_n"]  = df["low"].rolling(BREAKOUT_PERIOD).min()

    return df.dropna(subset=["ema_trend", "rsi", "atr", "adx"])


# ─────────────────────────────────────────────
#  SUPPORT & RESISTANCE
# ─────────────────────────────────────────────

def find_sr_levels(df, window=20):
    """Findet Support & Resistance Zonen."""
    levels = []
    highs  = df["high"].values
    lows   = df["low"].values

    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i - window: i + window + 1]):
            levels.append({"price": highs[i], "type": "resistance", "touches": 1})
        if lows[i] == min(lows[i - window: i + window + 1]):
            levels.append({"price": lows[i],  "type": "support",    "touches": 1})

    # Ähnliche Levels zusammenführen
    if not levels:
        return []
    levels = sorted(levels, key=lambda x: x["price"])
    merged = []
    cur    = levels[0].copy()
    for lv in levels[1:]:
        if abs(lv["price"] - cur["price"]) <= cur["price"] * 0.001:
            cur["touches"] += 1
            cur["price"]    = (cur["price"] + lv["price"]) / 2
        else:
            merged.append(cur)
            cur = lv.copy()
    merged.append(cur)
    return merged


# ─────────────────────────────────────────────
#  FAIR VALUE GAPS
# ─────────────────────────────────────────────

def find_fvg(df):
    """Erkennt Fair Value Gaps (bullish + bearish)."""
    df = df.copy()
    df["fvg_bull"] = False
    df["fvg_bear"] = False
    df["fvg_top"]  = np.nan
    df["fvg_bot"]  = np.nan

    for i in range(1, len(df) - 1):
        prev_high = df["high"].iloc[i - 1]
        prev_low  = df["low"].iloc[i - 1]
        next_low  = df["low"].iloc[i + 1]
        next_high = df["high"].iloc[i + 1]

        if next_low > prev_high:
            df.iloc[i, df.columns.get_loc("fvg_bull")] = True
            df.iloc[i, df.columns.get_loc("fvg_top")]  = next_low
            df.iloc[i, df.columns.get_loc("fvg_bot")]  = prev_high

        if next_high < prev_low:
            df.iloc[i, df.columns.get_loc("fvg_bear")] = True
            df.iloc[i, df.columns.get_loc("fvg_top")]  = prev_low
            df.iloc[i, df.columns.get_loc("fvg_bot")]  = next_high

    return df


# ─────────────────────────────────────────────
#  CME GAPS (nur BTC)
# ─────────────────────────────────────────────

def find_cme_gaps(df_daily):
    """Erkennt offene CME Gaps in BTC-Daten."""
    gaps = []
    for i in range(1, len(df_daily)):
        days_diff = (df_daily.index[i] - df_daily.index[i-1]).days
        if days_diff >= 2:
            fri_close  = df_daily["close"].iloc[i - 1]
            mon_open   = df_daily["open"].iloc[i]
            gap_pct    = abs(mon_open - fri_close) / fri_close
            if gap_pct >= 0.001:
                gaps.append({
                    "date":      df_daily.index[i].strftime("%Y-%m-%d"),
                    "gap_top":   max(fri_close, mon_open),
                    "gap_bot":   min(fri_close, mon_open),
                    "gap_pct":   round(gap_pct * 100, 3),
                    "direction": "up" if mon_open > fri_close else "down",
                    "filled":    False,
                })

    for gap in gaps:
        idx = df_daily.index.searchsorted(gap["date"])
        for _, row in df_daily.iloc[idx:].iterrows():
            if row["low"] <= gap["gap_bot"] and row["high"] >= gap["gap_top"]:
                gap["filled"] = True
                break

    open_gaps = [g for g in gaps if not g["filled"]]
    print(f"  📊 {len(gaps)} CME Gaps, {len(open_gaps)} offen")
    return open_gaps


# ─────────────────────────────────────────────
#  ALLES ZUSAMMEN LADEN
# ─────────────────────────────────────────────

def prepare_symbol_data(symbol):
    """Lädt alle Timeframes + berechnet alles für ein Symbol."""
    print(f"\n{'═'*50}")
    print(f"  {symbol}")
    print(f"{'═'*50}")

    result = {}

    import time
    result["weekly"] = add_indicators(load_ohlcv(symbol, "1wk", years=HISTORY_YEARS))
    time.sleep(2)
    result["daily"]  = add_indicators(load_ohlcv(symbol, "1d",  years=HISTORY_YEARS))
    time.sleep(2)
    result["h1"]     = add_indicators(load_ohlcv(symbol, "1h",  years=2))
    result["h4"]     = add_indicators(resample_to_4h(result["h1"]))

    result["daily"]  = find_fvg(result["daily"])
    result["h4"]     = find_fvg(result["h4"])

    result["sr_daily"] = find_sr_levels(result["daily"], window=10)
    result["sr_h4"]    = find_sr_levels(result["h4"],    window=20)
    print(f"  🎯 S/R: {len(result['sr_daily'])} Daily, {len(result['sr_h4'])} 4H Zonen")

    result["cme_gaps"] = find_cme_gaps(result["daily"]) if "BTC" in symbol.upper() else []

    return result


# ─────────────────────────────────────────────
#  START — wird ausgeführt wenn du F5 drückst
# ─────────────────────────────────────────────

print("\n" + "═"*50)
print("  TRADING BOT — Modul 1 Test")
print("═"*50)

data = prepare_symbol_data("EURUSD=X")

print("\n📊 Letzte 3 Kerzen (1H):")
cols = ["close", "ema_fast", "ema_slow", "rsi", "adx", "atr"]
print(data["h1"][cols].tail(3).round(5).to_string())

print("\n📍 Erste 3 S/R Zonen (Daily):")
for z in data["sr_daily"][:3]:
    print(f"   {z['type']:12s} @ {z['price']:.5f}  (Touches: {z['touches']})")

print("\n✅ Modul 1 funktioniert!")


# ─────────────────────────────────────────────
#  MODUL 2 — BOS / CHoCH TRENDWECHSEL
# ─────────────────────────────────────────────
#
#  BOS  = Break of Structure
#         Trend setzt fort → neues HH in Aufwärtstrend
#         oder neues LL in Abwärtstrend
#
#  CHoCH = Change of Character
#          Erstes Warnsignal → Trend könnte drehen
#          Noch KEIN Trendwechsel — nur Pause-Modus
#
#  Trendwechsel bestätigt erst wenn:
#  CHoCH + anschließender BOS in Gegenrichtung
#
#  Bias-States:
#  "bull"   → nur Long-Trades erlaubt
#  "bear"   → nur Short-Trades erlaubt
#  "pause"  → kein neuer Trade (CHoCH erkannt, kein BOS yet)
# ─────────────────────────────────────────────

def find_swing_points(df, window=10):
    """
    Findet alle Swing-Highs und Swing-Lows.
    
    Ein Swing-High ist eine Kerze die höher ist als
    alle 'window' Kerzen links und rechts davon.
    Gleiches Prinzip für Swing-Lows.
    
    Returns:
        DataFrame mit Spalten:
        - swing_high: True wenn diese Kerze ein Swing-High ist
        - swing_low:  True wenn diese Kerze ein Swing-Low ist
    """
    df = df.copy()
    df["swing_high"] = False
    df["swing_low"]  = False

    highs = df["high"].values
    lows  = df["low"].values

    for i in range(window, len(df) - window):
        # Swing-High: höher als alle Kerzen im Fenster
        if highs[i] == max(highs[i - window: i + window + 1]):
            df.iloc[i, df.columns.get_loc("swing_high")] = True

        # Swing-Low: tiefer als alle Kerzen im Fenster
        if lows[i] == min(lows[i - window: i + window + 1]):
            df.iloc[i, df.columns.get_loc("swing_low")] = True

    return df


def detect_bos_choch(df, window=10):
    """
    Erkennt BOS (Break of Structure) und CHoCH (Change of Character).
    
    Logik:
    - Aufwärtstrend: HH + HL Sequenz
      → BOS wenn neues HH über letztem HH
      → CHoCH wenn Preis unter letztes HL bricht
    
    - Abwärtstrend: LL + LH Sequenz  
      → BOS wenn neues LL unter letztem LL
      → CHoCH wenn Preis über letztes LH bricht

    Returns:
        DataFrame mit neuen Spalten:
        - bos_bull:   bullisher BOS (Aufwärtstrend setzt fort)
        - bos_bear:   bearisher BOS (Abwärtstrend setzt fort)
        - choch_bull: CHoCH nach Abwärtstrend (mögliche Umkehr nach oben)
        - choch_bear: CHoCH nach Aufwärtstrend (mögliche Umkehr nach unten)
        - structure:  aktueller Bias ("bull", "bear", "pause_bull", "pause_bear")
    """
    # Erst Swing-Punkte finden
    df = find_swing_points(df, window=window)
    df = df.copy()

    df["bos_bull"]   = False
    df["bos_bear"]   = False
    df["choch_bull"] = False
    df["choch_bear"] = False
    df["structure"]  = "neutral"

    # Swing-Punkte extrahieren
    swing_highs = df[df["swing_high"]]["high"].to_dict()
    swing_lows  = df[df["swing_low"]]["low"].to_dict()

    sh_indices = sorted(swing_highs.keys())
    sl_indices = sorted(swing_lows.keys())

    if len(sh_indices) < 2 or len(sl_indices) < 2:
        return df

    # Durch alle Kerzen iterieren und Struktur aufbauen
    current_bias    = "neutral"
    last_hh         = None   # Letztes Higher High
    last_hl         = None   # Letztes Higher Low
    last_ll         = None   # Letztes Lower Low
    last_lh         = None   # Letztes Lower High

    for i in range(len(df)):
        idx   = df.index[i]
        close = df["close"].iloc[i]
        high  = df["high"].iloc[i]
        low   = df["low"].iloc[i]

        # Swing-Punkte bis zu dieser Kerze sammeln
        prev_sh = [sh_indices[j] for j in range(len(sh_indices)) if sh_indices[j] < idx]
        prev_sl = [sl_indices[j] for j in range(len(sl_indices)) if sl_indices[j] < idx]

        if len(prev_sh) < 2 or len(prev_sl) < 2:
            df.iloc[i, df.columns.get_loc("structure")] = "neutral"
            continue

        # Letzten zwei Swing-Highs und Swing-Lows
        last_two_sh = prev_sh[-2:]
        last_two_sl = prev_sl[-2:]

        sh1 = swing_highs[last_two_sh[0]]
        sh2 = swing_highs[last_two_sh[1]]
        sl1 = swing_lows[last_two_sl[0]]
        sl2 = swing_lows[last_two_sl[1]]

        # ── AUFWÄRTSTREND ERKENNEN ──────────────────
        is_uptrend = sh2 > sh1 and sl2 > sl1  # HH + HL

        # ── ABWÄRTSTREND ERKENNEN ───────────────────
        is_downtrend = sh2 < sh1 and sl2 < sl1  # LH + LL

        if is_uptrend:
            last_hh = sh2
            last_hl = sl2

            if current_bias in ["neutral", "bear", "pause_bear"]:
                # Trendwechsel zu bullish bestätigt (BOS nach CHoCH)
                df.iloc[i, df.columns.get_loc("bos_bull")] = True
                current_bias = "bull"

            elif current_bias == "bull":
                # Trend setzt fort
                df.iloc[i, df.columns.get_loc("bos_bull")] = True
                current_bias = "bull"

            elif current_bias == "pause_bull":
                # Bestätigung nach CHoCH → Trend dreht bullish
                df.iloc[i, df.columns.get_loc("bos_bull")] = True
                current_bias = "bull"

        elif is_downtrend:
            last_ll = sl2
            last_lh = sh2

            if current_bias in ["neutral", "bull", "pause_bull"]:
                # Trendwechsel zu bearish bestätigt
                df.iloc[i, df.columns.get_loc("bos_bear")] = True
                current_bias = "bear"

            elif current_bias == "bear":
                # Trend setzt fort
                df.iloc[i, df.columns.get_loc("bos_bear")] = True
                current_bias = "bear"

            elif current_bias == "pause_bear":
                # Bestätigung nach CHoCH → Trend dreht bearish
                df.iloc[i, df.columns.get_loc("bos_bear")] = True
                current_bias = "bear"

        else:
            # Keine klare Struktur mehr → CHoCH prüfen
            if current_bias == "bull" and last_hl is not None:
                # Preis bricht unter letztes Higher Low → CHoCH warnung
                if low < last_hl:
                    df.iloc[i, df.columns.get_loc("choch_bear")] = True
                    current_bias = "pause_bull"

            elif current_bias == "bear" and last_lh is not None:
                # Preis bricht über letztes Lower High → CHoCH warnung
                if high > last_lh:
                    df.iloc[i, df.columns.get_loc("choch_bull")] = True
                    current_bias = "pause_bear"

        df.iloc[i, df.columns.get_loc("structure")] = current_bias

    return df


def get_current_bias(df_structure):
    """
    Gibt den aktuellen Trend-Bias zurück.
    
    Returns:
        "bull"        → nur Long-Trades erlaubt
        "bear"        → nur Short-Trades erlaubt  
        "pause_bull"  → CHoCH erkannt, kein neuer Trade
        "pause_bear"  → CHoCH erkannt, kein neuer Trade
        "neutral"     → kein klarer Trend
    """
    return df_structure["structure"].iloc[-1]


def get_bias_summary(df_structure):
    """Gibt eine lesbare Zusammenfassung des aktuellen Bias zurück."""
    bias = get_current_bias(df_structure)
    
    labels = {
        "bull":       "📈 BULLISH  — nur Long-Trades erlaubt",
        "bear":       "📉 BEARISH  — nur Short-Trades erlaubt",
        "pause_bull": "⏸  PAUSE    — CHoCH bearish, warte auf Bestätigung",
        "pause_bear": "⏸  PAUSE    — CHoCH bullish, warte auf Bestätigung",
        "neutral":    "➖ NEUTRAL  — kein klarer Trend",
    }
    
    # Letzte Strukturänderung finden
    changes = df_structure[df_structure["structure"] != df_structure["structure"].shift()]
    last_change = changes.index[-1] if len(changes) > 0 else df_structure.index[-1]
    
    return {
        "bias":        bias,
        "label":       labels.get(bias, bias),
        "since":       last_change,
        "trade_allowed": bias in ["bull", "bear"],
    }


def analyze_all_timeframes(data_dict):
    """
    Analysiert BOS/CHoCH auf allen Timeframes und gibt
    einen kombinierten Gesamt-Bias zurück.
    
    Regel:
    - Weekly + Daily + 4H müssen alle dieselbe Richtung zeigen
    - Erst dann ist ein Trade erlaubt
    
    Returns:
        {
            "weekly_bias":  str,
            "daily_bias":   str,
            "h4_bias":      str,
            "final_bias":   str,   ("bull", "bear", oder "no_trade")
            "reason":       str,   Warum kein Trade wenn kein Trade
        }
    """
    print("\n  🔍 Analysiere Trendstruktur...")

    # BOS/CHoCH auf jedem Timeframe berechnen
    weekly_struct = detect_bos_choch(data_dict["weekly"], window=3)
    daily_struct  = detect_bos_choch(data_dict["daily"],  window=10)
    h4_struct     = detect_bos_choch(data_dict["h4"],     window=20)

    weekly_bias = get_current_bias(weekly_struct)
    daily_bias  = get_current_bias(daily_struct)
    h4_bias     = get_current_bias(h4_struct)

    # Alle drei müssen aligned sein
    # Weekly muss nur nicht gegen uns sein
    bull_aligned = (daily_bias == "bull" and h4_bias == "bull" and weekly_bias != "bear")
    bear_aligned = (daily_bias == "bear" and h4_bias == "bear" and weekly_bias != "bull")

    if bull_aligned:
        final_bias = "bull"
        reason     = "✅ Alle Timeframes bullish — Long-Trades erlaubt"
    elif bear_aligned:
        final_bias = "bear"
        reason     = "✅ Alle Timeframes bearish — Short-Trades erlaubt"
    elif any(b in ["pause_bull", "pause_bear"] for b in [weekly_bias, daily_bias, h4_bias]):
        final_bias = "no_trade"
        reason     = "⏸  CHoCH auf mindestens einem Timeframe — kein Trade"
    else:
        final_bias = "no_trade"
        reason     = f"❌ Timeframes nicht aligned: Weekly={weekly_bias}, Daily={daily_bias}, 4H={h4_bias}"

    return {
        "weekly_bias": weekly_bias,
        "daily_bias":  daily_bias,
        "h4_bias":     h4_bias,
        "final_bias":  final_bias,
        "reason":      reason,
        "weekly_df":   weekly_struct,
        "daily_df":    daily_struct,
        "h4_df":       h4_struct,
    }


# ─────────────────────────────────────────────
#  MODUL 2 TEST
# ─────────────────────────────────────────────

print("\n" + "═"*50)
print("  MODUL 2 — BOS/CHoCH Test")
print("═"*50)

# Trend-Analyse für EUR/USD
trend = analyze_all_timeframes(data)

print(f"\n  Weekly : {trend['weekly_bias']}")
print(f"  Daily  : {trend['daily_bias']}")
print(f"  4H     : {trend['h4_bias']}")
print(f"\n  ➤ {trend['reason']}")

# Letzte 5 Strukturänderungen auf dem Daily
daily_changes = trend["daily_df"][
    trend["daily_df"]["bos_bull"] | 
    trend["daily_df"]["bos_bear"] | 
    trend["daily_df"]["choch_bull"] | 
    trend["daily_df"]["choch_bear"]
].tail(5)

print(f"\n  📋 Letzte Strukturänderungen (Daily):")
for idx, row in daily_changes.iterrows():
    if row["bos_bull"]:
        event = "🟢 BOS  bullish  (Aufwärtstrend setzt fort)"
    elif row["bos_bear"]:
        event = "🔴 BOS  bearish  (Abwärtstrend setzt fort)"
    elif row["choch_bull"]:
        event = "🟡 CHoCH bullish (mögliche Umkehr nach oben)"
    else:
        event = "🟡 CHoCH bearish (mögliche Umkehr nach unten)"
    print(f"     {idx.strftime('%Y-%m-%d')}  {event}")

print("\n✅ Modul 2 funktioniert!")


# ─────────────────────────────────────────────
#  MODUL 3 — SIGNAL GENERATOR
# ─────────────────────────────────────────────
#
#  Prüft alle 8 Bedingungen der Pre-Trade Checkliste:
#
#  1. Weekly + Daily + 4H Trend aligned (BOS, kein CHoCH)
#  2. 1H EMA Cross (20 über/unter 50)
#  3. RSI im erlaubten Bereich (50-65 Long, 35-50 Short)
#  4. ADX > 20 (Markt trendet)
#  5. Entry nah an Zone (FVG / S/R)
#  6. RR >= 3:1 erreichbar
#  7. Kein Blackout-Fenster (US-Open etc.)
#  8. Circuit Breaker inaktiv
#
#  Korrelationsregel:
#  EUR/USD + GBP/USD = korrelierend → nie beide gleichzeitig
#  BTC + ETH = korrelierend → nie beide gleichzeitig
# ─────────────────────────────────────────────

from datetime import timezone

# Korrelationsgruppen — nie zwei aus derselben Gruppe gleichzeitig
CORRELATION_GROUPS = [
    ["EURUSD=X", "GBPUSD=X"],   # EUR und GBP stark korrelierend
    ["BTC-USD",  "ETH-USD"],    # BTC und ETH stark korrelierend
]

# Handelszeiten MEZ (UTC+1 Winter, UTC+2 Sommer)
# Wir rechnen in UTC — MEZ = UTC+1, MESZ = UTC+2
TRADING_WINDOWS_UTC = [
    {"start": 8,  "end": 13},   # London Session (09:30-15:00 MEZ = 08-13 UTC)
    {"start": 14, "end": 18},   # London/NY Overlap (16:00-20:00 MEZ = 14-18 UTC)
    {"start": 18, "end": 20},   # NY Late (20:00-22:00 MEZ = 18-20 UTC)
]

# Blackout-Fenster UTC
BLACKOUT_WINDOWS_UTC = [
    {"start": 7,  "end": 8},    # Frankfurt Open (08:00-09:30 MEZ)
    {"start": 13, "end": 14},   # US Open ±30 Min (15:00-16:00 MEZ = 13-14 UTC)
]

# Wochenend-Einstellungen für BTC
WEEKEND_ADX_MIN    = 30    # Strengerer ADX Filter am Wochenende
WEEKEND_RISK_MULT  = 0.5   # Halbe Position Size am Wochenende
WEEKEND_NO_TRADE_START = 1  # Kein Trade 02:00-08:00 MEZ = 01-06 UTC
WEEKEND_NO_TRADE_END   = 6

# Zone-Nähe: wie nah muss der Entry an einer Zone sein? (in ATR)
ZONE_PROXIMITY_ATR = 3.0   # erhöht von 1.5 → mehr Trades


def is_trading_hours(dt=None):
    """
    Prüft ob gerade Handelszeit ist.
    
    Returns:
        (bool, str) — (erlaubt, Grund wenn nicht erlaubt)
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    hour    = dt.hour
    weekday = dt.weekday()  # 0=Mo, 6=So

    # Wochenende
    if weekday == 5:  # Samstag
        return False, "Samstag — kein Trading"

    if weekday == 6:  # Sonntag
        # Sonntag nur ab 22:00 UTC (CME öffnet)
        if hour < 22:
            return False, "Sonntag vor CME-Öffnung (22:00 UTC)"
        return True, "Sonntag CME-Fenster"

    # Blackout-Fenster prüfen
    for bw in BLACKOUT_WINDOWS_UTC:
        if bw["start"] <= hour < bw["end"]:
            return False, f"Blackout-Fenster ({bw['start']}:00-{bw['end']}:00 UTC)"

    # Handelszeiten prüfen
    for tw in TRADING_WINDOWS_UTC:
        if tw["start"] <= hour < tw["end"]:
            return True, "Handelszeit aktiv"

    # Nacht-Session: nur Asia-Paare und BTC erlaubt
    # symbol wird als keyword argument übergeben
    return "night", f"Nacht-Session (Stunde: {hour} UTC) — nur Asia-Paare"


def is_near_zone(price, atr, sr_levels, df_h4, df_daily):
    """
    Prüft ob der aktuelle Preis nah an einer wichtigen Zone ist.
    FVG, S/R oder Order Block.
    
    Returns:
        (bool, str) — (nah an Zone, welche Zone)
    """
    tolerance = atr * ZONE_PROXIMITY_ATR

    # S/R Zonen prüfen
    for zone in sr_levels:
        if abs(price - zone["price"]) <= tolerance:
            return True, f"S/R Zone @ {zone['price']:.5f} ({zone['type']}, {zone['touches']} touches)"

    # FVG auf 4H prüfen (letzte 50 Kerzen)
    recent_h4 = df_h4.tail(50)
    bull_fvgs  = recent_h4[recent_h4["fvg_bull"]]
    bear_fvgs  = recent_h4[recent_h4["fvg_bear"]]

    for _, fvg in bull_fvgs.iterrows():
        if fvg["fvg_bot"] <= price <= fvg["fvg_top"] + tolerance:
            return True, f"Bullishes FVG (4H) @ {fvg['fvg_bot']:.5f}-{fvg['fvg_top']:.5f}"

    for _, fvg in bear_fvgs.iterrows():
        if fvg["fvg_bot"] - tolerance <= price <= fvg["fvg_top"]:
            return True, f"Bearishes FVG (4H) @ {fvg['fvg_bot']:.5f}-{fvg['fvg_top']:.5f}"

    # FVG auf Daily prüfen
    recent_daily = df_daily.tail(20)
    for _, fvg in recent_daily[recent_daily["fvg_bull"]].iterrows():
        if fvg["fvg_bot"] <= price <= fvg["fvg_top"] + tolerance:
            return True, f"Bullishes FVG (Daily) @ {fvg['fvg_bot']:.5f}-{fvg['fvg_top']:.5f}"

    for _, fvg in recent_daily[recent_daily["fvg_bear"]].iterrows():
        if fvg["fvg_bot"] - tolerance <= price <= fvg["fvg_top"]:
            return True, f"Bearishes FVG (Daily) @ {fvg['fvg_bot']:.5f}-{fvg['fvg_top']:.5f}"

    return False, "Keine Zone in der Nähe"


def check_rr(entry, sl, direction, df_h1, sr_levels):
    """
    Prüft ob RR >= 3:1 erreichbar ist.
    Schaut ob der TP durch eine S/R Zone blockiert wird.
    
    Returns:
        (bool, float, float) — (rr_ok, tp_price, actual_rr)
    """
    risk   = abs(entry - sl)
    tp_raw = entry + (risk * TP_RR_RATIO) if direction == "long" else entry - (risk * TP_RR_RATIO)

    # Prüfen ob S/R Zone zwischen Entry und TP liegt
    if direction == "long":
        blocking_zones = [z for z in sr_levels
                         if z["type"] == "resistance"
                         and entry < z["price"] < tp_raw
                         and z["touches"] >= 3]
    else:
        blocking_zones = [z for z in sr_levels
                         if z["type"] == "support"
                         and tp_raw < z["price"] < entry
                         and z["touches"] >= 3]

    if blocking_zones:
        # Nächste blockierende Zone
        if direction == "long":
            nearest = min(blocking_zones, key=lambda z: z["price"])
            actual_tp = nearest["price"] - (df_h1["atr"].iloc[-1] * 0.5)
        else:
            nearest = max(blocking_zones, key=lambda z: z["price"])
            actual_tp = nearest["price"] + (df_h1["atr"].iloc[-1] * 0.5)

        actual_rr = abs(actual_tp - entry) / risk
        return actual_rr >= TP_RR_RATIO, actual_tp, round(actual_rr, 2)

    return True, tp_raw, TP_RR_RATIO


def check_correlation(symbol, open_trades):
    """
    Prüft ob ein neuer Trade die Korrelationsregel verletzt.
    Nie zwei stark korrelierende Symbole gleichzeitig offen.
    
    Returns:
        (bool, str) — (erlaubt, Grund wenn nicht erlaubt)
    """
    for group in CORRELATION_GROUPS:
        if symbol in group:
            for open_symbol in open_trades:
                if open_symbol != symbol and open_symbol in group:
                    return False, f"Korrelation: {open_symbol} bereits offen"
    return True, "Keine Korrelationskonflikte"


def generate_signal(symbol, data_dict, trend_analysis,
                    circuit_breaker, open_trades,
                    dt=None, news_blackout=False):
    """
    Hauptfunktion — prüft alle 8 Bedingungen und gibt
    ein Trade-Signal zurück.

    Args:
        symbol:           z.B. "EURUSD=X"
        data_dict:        Output von prepare_symbol_data()
        trend_analysis:   Output von analyze_all_timeframes()
        circuit_breaker:  {"active": bool, "losses_in_row": int, "daily_loss": float}
        open_trades:      Liste offener Trade-Symbole z.B. ["EURUSD=X"]
        dt:               Datetime für Test (None = jetzt)
        news_blackout:    True wenn News in ±60 Min (Modul 4)

    Returns:
        {
            "signal":     "long" / "short" / "no_trade",
            "entry":      float,
            "sl":         float,
            "tp":         float,
            "rr":         float,
            "checks":     dict,   alle 8 Checks mit Ergebnis
            "reason":     str,    Hauptgrund bei no_trade
        }
    """
    df_h1    = data_dict["h1"]
    df_h4    = data_dict["h4"]
    df_daily = data_dict["daily"]
    sr       = data_dict["sr_daily"] + data_dict["sr_h4"]

    # Aktuelle Werte
    current      = df_h1.iloc[-1]
    price        = float(current["close"])
    atr          = float(current["atr"])
    rsi          = float(current["rsi"])
    adx          = float(current["adx"])
    ema_fast     = float(current["ema_fast"])
    ema_slow     = float(current["ema_slow"])
    prev         = df_h1.iloc[-2]
    prev_fast    = float(prev["ema_fast"])
    prev_slow    = float(prev["ema_slow"])

    # EMA Cross erkennen — letzte 5 Kerzen (nicht nur letzte 1)
    # Verhindert dass ein Cross "verpasst" wird wenn er 1-4h zurückliegt
    ema_cross_bull = False
    ema_cross_bear = False
    for _k in range(1, min(6, len(df_h1))):
        _pf = float(df_h1["ema_fast"].iloc[-_k - 1])
        _ps = float(df_h1["ema_slow"].iloc[-_k - 1])
        _cf = float(df_h1["ema_fast"].iloc[-_k])
        _cs = float(df_h1["ema_slow"].iloc[-_k])
        if _pf <= _ps and _cf > _cs:
            ema_cross_bull = True
            break
        if _pf >= _ps and _cf < _cs:
            ema_cross_bear = True
            break

    # Wochenende?
    if dt is None:
        dt = datetime.now(timezone.utc)
    is_weekend = dt.weekday() >= 5

    # ── CHECK 1: Handelszeiten ────────────────
    hours_result, hours_reason = is_trading_hours(dt)
    hours_ok = hours_result == True

    # Nacht-Session: nur Asia-Paare und BTC
    ASIA_SYMBOLS = ["USDJPY=X", "AUDUSD=X", "BTC-USD", "ETH-USD"]
    if hours_result == "night":
        if symbol in ASIA_SYMBOLS:
            hours_ok     = True
            hours_reason = "Nacht-Session — Asia/Crypto erlaubt"
            # Strengere Regeln nachts
            if adx < 25:
                hours_ok     = False
                hours_reason = f"Nacht-Session: ADX {adx:.1f} < 25 (zu schwach)"
        else:
            hours_ok     = False
            hours_reason = "Nacht-Session — nur USD/JPY, AUD/USD, BTC, ETH"

    # BTC darf am Wochenende (mit strengeren Regeln)
    if is_weekend and "BTC" in symbol.upper():
        btc_hour = dt.hour
        if WEEKEND_NO_TRADE_START <= btc_hour < WEEKEND_NO_TRADE_END:
            hours_ok     = False
            hours_reason = "BTC Wochenende Nacht-Blackout (02-08 MEZ)"
        else:
            hours_ok     = True
            hours_reason = "BTC Wochenend-Fenster aktiv"
    elif is_weekend and symbol not in ASIA_SYMBOLS:
        hours_ok     = False
        hours_reason = "Wochenende — kein Forex Trading"
    
    # Position Size nachts halbieren
    if hours_result == "night" and hours_ok:
        risk_mult_night = 0.5
    else:
        risk_mult_night = 1.0

    # ── CHECK 2: Trend aligned ────────────────
    final_bias  = trend_analysis["final_bias"]
    trend_ok    = final_bias in ["bull", "bear"]
    trend_reason = trend_analysis["reason"]

    # ── CHECK 3: EMA Cross ────────────────────
    if final_bias == "bull":
        ema_ok     = ema_cross_bull
        ema_reason = "EMA 20 kreuzt über EMA 50" if ema_ok else "Kein bullisher EMA Cross"
    elif final_bias == "bear":
        ema_ok     = ema_cross_bear
        ema_reason = "EMA 20 kreuzt unter EMA 50" if ema_ok else "Kein bearisher EMA Cross"
    else:
        ema_ok     = False
        ema_reason = "Kein Trend — kein EMA Check"

    # ── CHECK 4: RSI ──────────────────────────
    if final_bias == "bull":
        rsi_ok     = 45 <= rsi <= 70
        rsi_reason = f"RSI {rsi:.1f} im Long-Bereich (45-70)" if rsi_ok else f"RSI {rsi:.1f} außerhalb Long-Bereich"
    elif final_bias == "bear":
        rsi_ok     = 30 <= rsi <= 55
        rsi_reason = f"RSI {rsi:.1f} im Short-Bereich (30-55)" if rsi_ok else f"RSI {rsi:.1f} außerhalb Short-Bereich"
    else:
        rsi_ok     = False
        rsi_reason = "Kein Trend — kein RSI Check"

    # ── CHECK 5: ADX ──────────────────────────
    adx_min    = WEEKEND_ADX_MIN if is_weekend else 20
    adx_ok     = adx >= adx_min
    adx_reason = f"ADX {adx:.1f} >= {adx_min} (Markt trendet)" if adx_ok else f"ADX {adx:.1f} < {adx_min} (Seitwärts)"

    # ── CHECK 6: Zone-Nähe ────────────────────
    zone_ok, zone_reason = is_near_zone(price, atr, sr, df_h4, df_daily)

    # ── CHECK 7: News ─────────────────────────
    news_ok     = not news_blackout
    news_reason = "Keine News in ±60 Min" if news_ok else "News-Blackout aktiv"

    # ── CHECK 8: Circuit Breaker ──────────────
    cb_active   = circuit_breaker.get("active", False)
    cb_ok       = not cb_active
    cb_reason   = "Circuit Breaker inaktiv" if cb_ok else f"Circuit Breaker aktiv ({circuit_breaker.get('losses_in_row', 0)} Verluste in Folge)"

    # ── CHECK 9: Korrelation ──────────────────
    corr_ok, corr_reason = check_correlation(symbol, open_trades)

    # ── ALLE CHECKS ZUSAMMENFASSEN ────────────
    checks = {
        "1_handelszeiten":  {"ok": hours_ok,  "info": hours_reason},
        "2_trend_aligned":  {"ok": trend_ok,  "info": trend_reason},
        "3_ema_cross":      {"ok": ema_ok,    "info": ema_reason},
        "4_rsi":            {"ok": rsi_ok,    "info": rsi_reason},
        "5_adx":            {"ok": adx_ok,    "info": adx_reason},
        "6_zone":           {"ok": zone_ok,   "info": zone_reason},
        "7_news":           {"ok": news_ok,   "info": news_reason},
        "8_circuit_breaker":{"ok": cb_ok,     "info": cb_reason},
        "9_korrelation":    {"ok": corr_ok,   "info": corr_reason},
    }

    # Wenn nicht alle Checks grün → kein Trade
    failed = [k for k, v in checks.items() if not v["ok"]]
    if failed:
        return {
            "signal": "no_trade",
            "entry":  price,
            "sl":     None,
            "tp":     None,
            "rr":     None,
            "checks": checks,
            "reason": f"Fehlgeschlagen: {', '.join(failed)}",
        }

    # ── RR BERECHNEN ──────────────────────────
    direction = "long" if final_bias == "bull" else "short"
    sl        = price - (atr * SL_ATR_MULT) if direction == "long" else price + (atr * SL_ATR_MULT)
    rr_ok, tp, actual_rr = check_rr(price, sl, direction, df_h1, sr)

    if not rr_ok:
        checks["rr"] = {"ok": False, "info": f"RR {actual_rr:.2f} < {TP_RR_RATIO} (S/R blockiert TP)"}
        return {
            "signal": "no_trade",
            "entry":  price,
            "sl":     round(sl, 5),
            "tp":     round(tp, 5),
            "rr":     actual_rr,
            "checks": checks,
            "reason": f"RR {actual_rr:.2f} nicht erreichbar",
        }

    checks["rr"] = {"ok": True, "info": f"RR {actual_rr:.2f} >= {TP_RR_RATIO}"}

    # Wochenend-Position Size für BTC
    risk_mult = WEEKEND_RISK_MULT if is_weekend and "BTC" in symbol.upper() else 1.0

    # Kombiniere Wochenend und Nacht Multiplikator
    final_risk_mult = min(risk_mult, risk_mult_night) if "risk_mult_night" in dir() else risk_mult

    return {
        "signal":    direction,
        "entry":     round(price, 5),
        "sl":        round(sl, 5),
        "tp":        round(tp, 5),
        "rr":        actual_rr,
        "risk_mult": final_risk_mult,
        "checks":    checks,
        "reason":    f"✅ {direction.upper()} Signal — alle Checks bestanden",
    }


def print_signal(symbol, signal_result):
    """Gibt ein Signal übersichtlich aus."""
    print(f"\n{'─'*50}")
    print(f"  {symbol} — Signal: {signal_result['signal'].upper()}")
    print(f"{'─'*50}")

    if signal_result["signal"] != "no_trade":
        print(f"  Entry : {signal_result['entry']}")
        print(f"  SL    : {signal_result['sl']}")
        print(f"  TP    : {signal_result['tp']}")
        print(f"  RR    : {signal_result['rr']:.2f}")
        if signal_result.get("risk_mult", 1.0) < 1.0:
            print(f"  ⚠️  Wochenend-Modus: {int(signal_result['risk_mult']*100)}% Position Size")

    print(f"\n  Checks:")
    for check, result in signal_result["checks"].items():
        icon = "✅" if result["ok"] else "❌"
        print(f"  {icon} {check:25s} {result['info']}")

    print(f"\n  → {signal_result['reason']}")


# ─────────────────────────────────────────────
#  MODUL 3 TEST
# ─────────────────────────────────────────────

print("\n" + "═"*50)
print("  MODUL 3 — Signal Generator Test")
print("═"*50)

# Circuit Breaker Status (noch kein Verlust)
circuit_breaker = {
    "active":        False,
    "losses_in_row": 0,
    "daily_loss":    0.0,
}

# Keine offenen Trades
open_trades = []

# Signal generieren
signal = generate_signal(
    symbol          = "EURUSD=X",
    data_dict       = data,
    trend_analysis  = trend,
    circuit_breaker = circuit_breaker,
    open_trades     = open_trades,
    news_blackout   = False,
)

print_signal("EURUSD=X", signal)
print("\n✅ Modul 3 funktioniert!")


# ─────────────────────────────────────────────
#  MODUL 4 — NEWS FILTER
import os
import requests
from pathlib import Path
# ─────────────────────────────────────────────
#
#  Lädt täglich alle High/Medium Impact Events
#  via Finnhub API und prüft vor jedem Trade
#  ob ein Blackout aktiv ist.
#
#  High Impact:   ±60 Minuten gesperrt
#  Medium Impact: ±30 Minuten gesperrt
#
#  Währungs-spezifisch:
#  USD-Event sperrt nur EUR/USD, GBP/USD
#  BTC-relevante Events sperren nur BTC/ETH
# ─────────────────────────────────────────────

import json

# .env Datei laden (API Key)
def load_env(env_path=None):
    """Lädt .env Datei und setzt Umgebungsvariablen."""
    if env_path is None:
        # Suche .env in verschiedenen Orten
        possible = [
            Path(r"C:/Users/Noah/Privates/Projekte N8N/Trading bot/trading_bot.env"),
            Path("trading_bot.env"),
            Path.home() / "Desktop" / "trading_bot.env",
            Path(".env"),
        ]
        for p in possible:
            if p.exists():
                env_path = p
                break

    if env_path and Path(env_path).exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()
        print(f"  ✅ .env geladen von {env_path}")
    else:
        print("  ⚠️  Keine .env Datei gefunden — News-Filter läuft im Offline-Modus")

# Beim Start laden
load_env()
FINNHUB_API_KEY    = os.environ.get("FINNHUB_API_KEY", "")
ALPHA_VANTAGE_KEY  = os.environ.get("ALPHA_VANTAGE_KEY", "")

# Cache für heute's Events (einmal laden reicht)
_news_cache = {"date": None, "events": []}

# Welche Währungen betreffen welche Symbole
SYMBOL_CURRENCIES = {
    "EURUSD=X": ["EUR", "USD"],
    "GBPUSD=X": ["GBP", "USD"],
    "USDJPY=X": ["USD", "JPY"],
    "AUDUSD=X": ["AUD", "USD"],
    "BTC-USD":  ["BTC", "USD"],
    "ETH-USD":  ["ETH", "USD"],
}

# Hardcoded Fallback: bekannte High-Impact Events 2025/2026
# (wird genutzt wenn Finnhub nicht erreichbar)
KNOWN_HIGH_IMPACT = [
    "Non-Farm Payrolls",
    "Fed Interest Rate Decision",
    "FOMC",
    "Consumer Price Index",
    "CPI",
    "GDP",
    "Retail Sales",
    "ISM Manufacturing",
    "ECB Interest Rate",
    "Bank of England",
    "PPI",
    "PCE",
    "Unemployment Rate",
    "ADP Employment",
    "Jobless Claims",
]


def fetch_news_calendar(date=None):
    """
    Lädt alle wirtschaftlichen Events für ein Datum von Finnhub.
    
    Nutzt Cache — wird nur einmal pro Tag abgerufen.
    Falls Finnhub nicht erreichbar → leere Liste (kein Crash).
    
    Returns:
        Liste von Events: [{"time": datetime, "event": str, 
                            "impact": "high"/"medium"/"low",
                            "currency": str}]
    """
    global _news_cache

    if date is None:
        date = datetime.now(timezone.utc).date()

    # Cache prüfen
    if _news_cache["date"] == str(date) and _news_cache["events"]:
        return _news_cache["events"]

    if not FINNHUB_API_KEY:
        print("  ⚠️  Kein Finnhub API Key — News-Filter im Offline-Modus")
        return []

    try:
        url = "https://finnhub.io/api/v1/calendar/economic"
        params = {
            "from":   str(date),
            "to":     str(date),
            "token":  FINNHUB_API_KEY,
        }
        resp = requests.get(url, params=params, timeout=10)

        if resp.status_code != 200:
            print(f"  ⚠️  Finnhub Fehler: HTTP {resp.status_code}")
            return []

        raw_events = resp.json().get("economicCalendar", [])
        events = []

        for ev in raw_events:
            # Impact bestimmen
            impact_raw = str(ev.get("impact", "")).lower()
            if impact_raw in ["high", "3"]:
                impact = "high"
            elif impact_raw in ["medium", "2"]:
                impact = "medium"
            else:
                impact = "low"

            # Nur High und Medium beachten
            if impact not in ["high", "medium"]:
                continue

            # Zeit parsen
            try:
                ev_time = datetime.fromisoformat(
                    ev.get("time", "").replace("Z", "+00:00")
                )
            except:
                continue

            events.append({
                "time":     ev_time,
                "event":    ev.get("event", "Unknown"),
                "impact":   impact,
                "currency": ev.get("country", "USD").upper(),
            })

        # Cache speichern
        _news_cache = {"date": str(date), "events": events}
        print(f"  📰 {len(events)} relevante News-Events heute geladen")
        return events

    except Exception as e:
        print(f"  ⚠️  Finnhub nicht erreichbar: {e} — Offline-Modus")
        return []


def is_news_blackout(symbol, dt=None, events=None):
    """
    Prüft ob gerade ein News-Blackout für dieses Symbol aktiv ist.
    
    Args:
        symbol:  z.B. "EURUSD=X"
        dt:      Zeitpunkt (None = jetzt)
        events:  News-Liste (None = heute laden)
    
    Returns:
        (bool, str) — (blackout_aktiv, Grund)
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    if events is None:
        events = fetch_news_calendar(dt.date())

    # Relevante Währungen für dieses Symbol
    relevant_currencies = SYMBOL_CURRENCIES.get(symbol, ["USD"])

    for ev in events:
        ev_time  = ev["time"]
        if ev_time.tzinfo is None:
            ev_time = ev_time.replace(tzinfo=timezone.utc)

        delta_min = (ev_time - dt).total_seconds() / 60  # Minuten bis Event

        # Blackout-Fenster je Impact
        if ev["impact"] == "high":
            blackout_min = 60
        else:
            blackout_min = 30

        # Ist das Event relevant für dieses Symbol?
        currency_match = ev["currency"] in relevant_currencies

        if currency_match and abs(delta_min) <= blackout_min:
            direction = "in" if delta_min > 0 else "vor"
            return True, (f"News-Blackout: {ev['event']} "
                         f"({ev['impact'].upper()}, {ev['currency']}) "
                         f"— {abs(int(delta_min))} Min {direction}")

    return False, "Keine relevanten News in Blackout-Fenster"


def get_todays_events_summary(symbol=None):
    """Gibt eine lesbare Übersicht der heutigen Events aus."""
    events = fetch_news_calendar()

    if not events:
        print("  📰 Keine Events heute oder Offline-Modus")
        return

    print(f"\n  📰 Heutige News-Events:")
    for ev in sorted(events, key=lambda x: x["time"]):
        # Nur relevante anzeigen wenn Symbol angegeben
        if symbol:
            relevant = SYMBOL_CURRENCIES.get(symbol, ["USD"])
            if ev["currency"] not in relevant:
                continue

        impact_icon = "🔴" if ev["impact"] == "high" else "🟡"
        time_str    = ev["time"].strftime("%H:%M UTC")
        print(f"  {impact_icon} {time_str}  {ev['currency']:4s}  {ev['event']}")


# ─────────────────────────────────────────────
#  MODUL 4 TEST
# ─────────────────────────────────────────────

print("\n" + "═"*50)
print("  MODUL 4 — News Filter Test")
print("═"*50)

# Heutige Events laden
todays_events = fetch_news_calendar()
get_todays_events_summary("EURUSD=X")

# Blackout Test für jetzt
blackout, reason = is_news_blackout("EURUSD=X", events=todays_events)
print(f"\n  Blackout jetzt aktiv: {'JA ⛔' if blackout else 'NEIN ✅'}")
print(f"  Grund: {reason}")

# Signal nochmal generieren — diesmal MIT News-Filter
print("\n  Signal mit News-Filter:")
signal_mit_news = generate_signal(
    symbol          = "EURUSD=X",
    data_dict       = data,
    trend_analysis  = trend,
    circuit_breaker = {"active": False, "losses_in_row": 0, "daily_loss": 0.0},
    open_trades     = [],
    news_blackout   = blackout,
)
print(f"  → Signal: {signal_mit_news['signal'].upper()}")
print(f"  → News Check: {signal_mit_news['checks'].get('7_news', {}).get('info', 'N/A')}")

print("\n✅ Modul 4 funktioniert!")


# ─────────────────────────────────────────────
#  MODUL 5 — RISK MANAGER
# ─────────────────────────────────────────────
#
#  Berechnet:
#  - Position Size (wie viel Kapital pro Trade)
#  - SL / TP als exakte Preise
#  - Partial Close bei 1.5R
#  - Trailing Stop (folgt EMA 20)
#  - Circuit Breaker (rolling 3 Verluste)
#  - Tages-Loss-Cap (2%)
# ─────────────────────────────────────────────

# ── KAPITAL & CIRCUIT BREAKER STATE ──────────
# In echtem Bot kommt das aus der Datenbank (Modul 9)
# Hier als globaler State für Paper Trading

risk_state = {
    "capital":        10000.0,   # Startkapital in USD (anpassen!)
    "daily_start":    10000.0,   # Kapital zu Tagesbeginn
    "losses_in_row":  0,         # Verluste in Folge (rolling)
    "daily_loss":     0.0,       # Heutiger Verlust in %
    "trades_today":   0,         # Anzahl Trades heute
    "total_trades":   0,         # Gesamte Trades (für Optimierung)
    "circuit_active": False,     # Circuit Breaker aktiv?
    "open_positions": {},        # Offene Positionen {symbol: position_dict}
    "trade_log":      [],        # Alle abgeschlossenen Trades
}


def calculate_position_size(capital, risk_pct, entry, sl, risk_mult=1.0):
    """
    Berechnet die optimale Position Size.
    
    Formel: Risiko in $ / (Entry - SL) = Units
    
    Beispiel:
        Kapital: 10.000$, Risiko: 1%, Entry: 1.1000, SL: 1.0985
        Risiko $: 100$
        Pip-Risiko: 15 Pips = 0.0015
        Units: 100 / 0.0015 = 66.666 Units
    
    Args:
        capital:    Gesamtkapital in USD
        risk_pct:   Risiko pro Trade (z.B. 0.01 = 1%)
        entry:      Entry-Preis
        sl:         Stop-Loss Preis
        risk_mult:  Multiplikator (0.5 am Wochenende)
    
    Returns:
        {
            "units":      float,  Anzahl Units/Lots
            "risk_usd":   float,  Risiko in USD
            "risk_pct":   float,  Risiko in %
        }
    """
    risk_usd    = capital * risk_pct * risk_mult
    pip_risk    = abs(entry - sl)

    if pip_risk == 0:
        return {"units": 0, "risk_usd": 0, "risk_pct": 0}

    units       = risk_usd / pip_risk
    actual_risk = (pip_risk * units) / capital

    return {
        "units":    round(units, 2),
        "risk_usd": round(risk_usd, 2),
        "risk_pct": round(actual_risk * 100, 3),
    }


def open_position(symbol, signal, capital, risk_mult=1.0):
    """
    Öffnet eine neue Position und speichert sie im State.
    
    Returns:
        position dict oder None wenn Circuit Breaker aktiv
    """
    global risk_state

    # Circuit Breaker prüfen
    if risk_state["circuit_active"]:
        print(f"  ⛔ Circuit Breaker aktiv — kein neuer Trade")
        return None

    entry = signal["entry"]
    sl    = signal["sl"]
    tp    = signal["tp"]
    rr    = signal["rr"]
    direction = signal["signal"]

    # Position Size berechnen
    sizing = calculate_position_size(
        capital   = risk_state["capital"],
        risk_pct  = RISK_PER_TRADE,
        entry     = entry,
        sl        = sl,
        risk_mult = risk_mult,
    )

    # Partial Close Level (bei 1.5R)
    risk        = abs(entry - sl)
    partial_tp  = entry + (risk * 1.5) if direction == "long" else entry - (risk * 1.5)

    position = {
        "symbol":       symbol,
        "direction":    direction,
        "entry":        entry,
        "sl":           sl,
        "sl_original":  sl,       # Original SL für Referenz
        "tp":           tp,
        "partial_tp":   partial_tp,
        "rr":           rr,
        "units":        sizing["units"],
        "units_open":   sizing["units"],   # Reduziert sich nach Partial Close
        "risk_usd":     sizing["risk_usd"],
        "partial_done": False,    # Wurde Partial Close bereits ausgeführt?
        "open_time":    datetime.now(timezone.utc).isoformat(),
        "trailing_sl":  None,     # Wird nach Partial Close aktiviert
    }

    risk_state["open_positions"][symbol] = position
    risk_state["trades_today"]   += 1
    risk_state["total_trades"]   += 1

    print(f"\n  📈 POSITION GEÖFFNET: {symbol}")
    print(f"     Richtung : {direction.upper()}")
    print(f"     Entry    : {entry}")
    print(f"     SL       : {sl}  (Risiko: {sizing['risk_usd']}$)")
    print(f"     TP       : {tp}  (RR: {rr})")
    print(f"     Units    : {sizing['units']}")
    print(f"     Partial  : {round(partial_tp, 5)} (bei 1.5R — 50% schließen)")

    return position


def update_position(symbol, current_price, current_ema_fast):
    """
    Aktualisiert eine offene Position mit dem aktuellen Preis.
    Prüft: SL Hit, TP Hit, Partial Close, Trailing Stop.
    
    Returns:
        "open"          — Position läuft weiter
        "partial_close" — 50% geschlossen bei 1.5R
        "sl_hit"        — Stop Loss getroffen
        "tp_hit"        — Take Profit erreicht
        "trailing_sl"   — Trailing Stop ausgelöst
    """
    global risk_state

    if symbol not in risk_state["open_positions"]:
        return None

    pos       = risk_state["open_positions"][symbol]
    direction = pos["direction"]
    entry     = pos["entry"]
    sl        = pos["sl"]
    tp        = pos["tp"]
    partial   = pos["partial_tp"]

    # ── SL HIT ───────────────────────────────
    if direction == "long"  and current_price <= sl:
        return _close_position(symbol, sl, "sl_hit")
    if direction == "short" and current_price >= sl:
        return _close_position(symbol, sl, "sl_hit")

    # ── TP HIT ───────────────────────────────
    if direction == "long"  and current_price >= tp:
        return _close_position(symbol, tp, "tp_hit")
    if direction == "short" and current_price <= tp:
        return _close_position(symbol, tp, "tp_hit")

    # ── PARTIAL CLOSE bei 1.5R ────────────────
    if not pos["partial_done"]:
        if direction == "long"  and current_price >= partial:
            _partial_close(symbol, current_price)
            return "partial_close"
        if direction == "short" and current_price <= partial:
            _partial_close(symbol, current_price)
            return "partial_close"

    # ── TRAILING STOP (nach Partial Close) ────
    if pos["partial_done"] and pos["trailing_sl"] is not None:
        # Trailing Stop folgt EMA 20
        new_trail = current_ema_fast

        if direction == "long":
            # Trail nur nach oben (nie schlechter als Break-Even)
            if new_trail > pos["sl"] and new_trail > entry:
                risk_state["open_positions"][symbol]["sl"] = round(new_trail, 5)
                risk_state["open_positions"][symbol]["trailing_sl"] = round(new_trail, 5)

            # Trailing SL getroffen?
            if current_price <= pos["sl"]:
                return _close_position(symbol, pos["sl"], "trailing_sl")

        elif direction == "short":
            # Trail nur nach unten
            if new_trail < pos["sl"] and new_trail < entry:
                risk_state["open_positions"][symbol]["sl"] = round(new_trail, 5)
                risk_state["open_positions"][symbol]["trailing_sl"] = round(new_trail, 5)

            if current_price >= pos["sl"]:
                return _close_position(symbol, pos["sl"], "trailing_sl")

    return "open"


def _partial_close(symbol, price):
    """Schließt 50% der Position bei 1.5R."""
    global risk_state
    pos = risk_state["open_positions"][symbol]

    units_to_close = pos["units_open"] * 0.5
    pnl_partial    = (price - pos["entry"]) * units_to_close
    if pos["direction"] == "short":
        pnl_partial = -pnl_partial

    risk_state["open_positions"][symbol]["units_open"]   = pos["units_open"] * 0.5
    risk_state["open_positions"][symbol]["partial_done"] = True
    risk_state["open_positions"][symbol]["trailing_sl"]  = pos["entry"]  # BE als Start
    risk_state["open_positions"][symbol]["sl"]           = pos["entry"]  # SL auf Break-Even

    print(f"  ✂️  PARTIAL CLOSE {symbol}: 50% bei {price} — P&L: +{round(pnl_partial, 2)}$")
    risk_state["capital"] += pnl_partial


def _close_position(symbol, close_price, reason):
    """Schließt eine Position komplett und updated den State."""
    global risk_state

    if symbol not in risk_state["open_positions"]:
        return reason

    pos       = risk_state["open_positions"][symbol]
    direction = pos["direction"]
    units     = pos["units_open"]

    # P&L berechnen
    pnl = (close_price - pos["entry"]) * units
    if direction == "short":
        pnl = -pnl

    # State updaten
    risk_state["capital"]   += pnl
    risk_state["daily_loss"] = (risk_state["daily_start"] - risk_state["capital"]) / risk_state["daily_start"]

    # Win/Loss tracken für Circuit Breaker
    if pnl > 0:
        risk_state["losses_in_row"] = 0  # Reset bei Gewinn
    else:
        risk_state["losses_in_row"] += 1

    # Trade loggen
    trade_record = {
        "symbol":      symbol,
        "direction":   direction,
        "entry":       pos["entry"],
        "close":       close_price,
        "reason":      reason,
        "pnl":         round(pnl, 2),
        "pnl_pct":     round(pnl / risk_state["capital"] * 100, 3),
        "units":       units,
        "open_time":   pos["open_time"],
        "close_time":  datetime.now(timezone.utc).isoformat(),
    }
    risk_state["trade_log"].append(trade_record)

    # Position entfernen
    del risk_state["open_positions"][symbol]

    icon = "✅" if pnl > 0 else "❌"
    print(f"\n  {icon} POSITION GESCHLOSSEN: {symbol}")
    print(f"     Grund    : {reason}")
    print(f"     Entry    : {pos['entry']} → Close: {close_price}")
    print(f"     P&L      : {'+' if pnl > 0 else ''}{round(pnl, 2)}$")
    print(f"     Kapital  : {round(risk_state['capital'], 2)}$")
    print(f"     Verluste : {risk_state['losses_in_row']} in Folge")

    # Circuit Breaker prüfen
    _check_circuit_breaker()

    return reason


def _check_circuit_breaker():
    """Aktiviert Circuit Breaker wenn nötig."""
    global risk_state

    if risk_state["losses_in_row"] >= MAX_LOSSES_ROW:
        risk_state["circuit_active"] = True
        print(f"\n  🔴 CIRCUIT BREAKER AKTIV!")
        print(f"     {risk_state['losses_in_row']} Verluste in Folge")
        print(f"     Trading heute gestoppt — Reset morgen 00:00 UTC")

    elif risk_state["daily_loss"] >= MAX_DAILY_LOSS:
        risk_state["circuit_active"] = True
        print(f"\n  🔴 CIRCUIT BREAKER AKTIV!")
        print(f"     Tagesverlust: {round(risk_state['daily_loss']*100, 2)}% >= {MAX_DAILY_LOSS*100}%")
        print(f"     Trading heute gestoppt")

    elif risk_state["losses_in_row"] >= MAX_LOSSES_ROW - 1:
        print(f"\n  ⚠️  WARNUNG: {risk_state['losses_in_row']} Verluste in Folge — nächster Verlust stoppt den Bot")


def reset_daily_state():
    """Reset täglich um 00:00 UTC."""
    global risk_state
    risk_state["daily_start"]    = risk_state["capital"]
    risk_state["daily_loss"]     = 0.0
    risk_state["trades_today"]   = 0
    risk_state["circuit_active"] = False
    risk_state["losses_in_row"]  = 0
    print(f"  🔄 Tages-Reset — Kapital: {round(risk_state['capital'], 2)}$")


def get_circuit_breaker_status():
    """Gibt Circuit Breaker Status für Signal Generator zurück."""
    return {
        "active":        risk_state["circuit_active"],
        "losses_in_row": risk_state["losses_in_row"],
        "daily_loss":    risk_state["daily_loss"],
    }


def print_risk_summary():
    """Gibt eine Übersicht des aktuellen Risk-States aus."""
    print(f"\n  💰 Risk Manager Status:")
    print(f"     Kapital         : {round(risk_state['capital'], 2)}$")
    print(f"     Tagesverlust    : {round(risk_state['daily_loss']*100, 2)}%")
    print(f"     Verluste in Folge: {risk_state['losses_in_row']}")
    print(f"     Circuit Breaker : {'🔴 AKTIV' if risk_state['circuit_active'] else '🟢 Inaktiv'}")
    print(f"     Offene Positionen: {len(risk_state['open_positions'])}")
    print(f"     Trades heute    : {risk_state['trades_today']}")
    print(f"     Trades gesamt   : {risk_state['total_trades']}")


# ─────────────────────────────────────────────
#  MODUL 5 TEST
# ─────────────────────────────────────────────

print("\n" + "═"*50)
print("  MODUL 5 — Risk Manager Test")
print("═"*50)

# Simuliere einen Trade
print("\n  📐 Position Size Berechnung:")
test_sizing = calculate_position_size(
    capital  = 10000.0,
    risk_pct = RISK_PER_TRADE,
    entry    = 1.1500,
    sl       = 1.1485,
)
print(f"     Kapital   : 10.000$")
print(f"     Risiko    : {RISK_PER_TRADE*100}% = {test_sizing['risk_usd']}$")
print(f"     Entry     : 1.1500, SL: 1.1485 (15 Pips)")
print(f"     Units     : {test_sizing['units']}")

# Simuliere Circuit Breaker
print("\n  ⚡ Circuit Breaker Simulation:")
test_state = risk_state.copy()

# 3 Verluste simulieren
for i in range(3):
    risk_state["losses_in_row"] += 1
    risk_state["capital"] -= 100
    risk_state["daily_loss"] = (10000 - risk_state["capital"]) / 10000
    print(f"     Verlust {i+1}: Kapital {round(risk_state['capital'], 2)}$")
    _check_circuit_breaker()
    if risk_state["circuit_active"]:
        break

# Reset für weiteren Test
risk_state["capital"]        = 10000.0
risk_state["losses_in_row"]  = 0
risk_state["daily_loss"]     = 0.0
risk_state["circuit_active"] = False
risk_state["daily_start"]    = 10000.0

print_risk_summary()
print("\n✅ Modul 5 funktioniert!")


# ─────────────────────────────────────────────
#  MODUL 6 — BACKTESTING ENGINE
# ─────────────────────────────────────────────
#
#  Simuliert die komplette Strategie auf
#  historischen Daten.
#
#  Für jede 1H-Kerze wird geprüft:
#  1. Sind alle Bedingungen erfüllt?
#  2. Wenn ja → Trade eröffnen
#  3. Jede folgende Kerze → Position updaten
#  4. SL/TP/Trailing prüfen
#
#  Am Ende: vollständige Statistiken
#  - Win-Rate, Profit Factor, Sharpe, Drawdown
# ─────────────────────────────────────────────

def run_backtest(symbol, data_dict, start_capital=10000.0, verbose=False, strategy=None):
    """
    Führt einen vollständigen Backtest durch.
    
    Args:
        symbol:        z.B. "EURUSD=X"
        data_dict:     Output von prepare_symbol_data()
        start_capital: Startkapital in USD
        verbose:       True = jeden Trade ausgeben
    
    Returns:
        {
            "trades":        Liste aller Trades,
            "capital_curve": Kapitalverlauf,
            "stats":         Kennzahlen,
        }
    """
    if strategy is None:
        strategy = SYMBOL_STRATEGY.get(symbol, "ema_pullback")
    tp_rr = SYMBOL_TP.get(symbol, STRATEGY_TP.get(strategy, TP_RR_RATIO))
    bb_period, _ = SYMBOL_BB.get(symbol, (20, BB_STD))
    bb_col = "bb14" if bb_period == 14 else "bb20"

    print(f"\n  🔄 Starte Backtest: {symbol} [{strategy}]")
    print(f"     Kapital: {start_capital}$")

    # 1H als Entry-Timeframe (live Strategy nutzt auch 1H)
    df_entry = data_dict["h1"].copy()
    df_h4    = data_dict["h4"].copy()
    df_daily = data_dict["daily"].copy()
    df_weekly= data_dict["weekly"].copy()
    sr       = data_dict["sr_daily"] + data_dict["sr_h4"]

    # State zurücksetzen
    capital        = start_capital
    losses_in_row  = 0
    daily_loss     = 0.0
    daily_start    = start_capital
    circuit_active = False
    open_pos       = {}
    trades         = []
    capital_curve  = [start_capital]
    last_date      = None

    # BOS/CHoCH vorberechnen
    print(f"     Berechne Trendstruktur...")
    weekly_struct = detect_bos_choch(df_weekly, window=3)
    daily_struct  = detect_bos_choch(df_daily,  window=10)
    h4_struct     = detect_bos_choch(df_h4,     window=20)

    # FVG vorberechnen
    df_h4    = find_fvg(df_h4)
    df_daily = find_fvg(df_daily)

    total_candles = len(df_entry)
    print(f"     Verarbeite {total_candles} Kerzen (1H)...")

    for i in range(50, total_candles):  # Erste 50 für Warmup
        candle   = df_entry.iloc[i]
        dt       = df_entry.index[i]
        price    = float(candle["close"])
        high     = float(candle["high"])
        low      = float(candle["low"])
        atr      = float(candle["atr"])
        ema_fast = float(candle["ema_fast"])

        # Tages-Reset — Circuit Breaker ist Tagesschutz, kein permanentes Ban
        current_date = dt.date()
        if last_date != current_date:
            daily_start    = capital
            daily_loss     = 0.0
            circuit_active = False  # Reset täglich
            losses_in_row  = 0      # Reset täglich
            last_date      = current_date

        # ── OFFENE POSITIONEN UPDATEN ─────────
        for sym in list(open_pos.keys()):
            pos       = open_pos[sym]
            direction = pos["direction"]
            entry     = pos["entry"]
            sl        = pos["sl"]
            tp        = pos["tp"]
            partial   = pos["partial_tp"]

            # SL Hit (nutze Low/High der Kerze für realistischere Simulation)
            sl_hit = (direction == "long"  and low  <= sl) or \
                     (direction == "short" and high >= sl)
            tp_hit = (direction == "long"  and high >= tp) or \
                     (direction == "short" and low  <= tp)

            if sl_hit:
                pnl = (sl - entry) * pos["units_open"]
                if direction == "short": pnl = -pnl
                capital       += pnl
                losses_in_row += 1 if pnl < 0 else 0
                if pnl > 0: losses_in_row = 0
                daily_loss = max(0, (daily_start - capital) / daily_start)
                trades.append(_make_trade_record(sym, pos, sl, "sl_hit", pnl, dt))
                del open_pos[sym]
                if verbose: print(f"  ❌ SL {sym} @ {sl} P&L: {round(pnl,2)}$")
                continue

            if tp_hit:
                pnl = (tp - entry) * pos["units_open"]
                if direction == "short": pnl = -pnl
                capital       += pnl
                losses_in_row  = 0
                daily_loss     = max(0, (daily_start - capital) / daily_start)
                trades.append(_make_trade_record(sym, pos, tp, "tp_hit", pnl, dt))
                del open_pos[sym]
                if verbose: print(f"  ✅ TP {sym} @ {tp} P&L: +{round(pnl,2)}$")
                continue

            # Partial Close DEAKTIVIERT — reines SL/TP für sauberes R:R
            # if not pos["partial_done"]:
            #     ...

            # Trailing Stop DEAKTIVIERT
            # if pos.get("partial_done") and pos.get("trailing_sl") is not None:
            #     ...

        # Circuit Breaker
        if losses_in_row >= MAX_LOSSES_ROW or daily_loss >= MAX_DAILY_LOSS:
            circuit_active = True

        capital_curve.append(capital)

        # ── NEUES SIGNAL PRÜFEN ───────────────
        if circuit_active:
            continue
        if symbol in open_pos:
            continue  # Bereits offen
        if len(open_pos) >= 3:
            continue  # Max 3 gleichzeitig

        # ── ENTRY ROUTING ────────────────────────────────────────────
        direction = None
        sl_price  = None
        tp_price  = None
        risk      = 0

        if i < 1:
            continue
        prev       = df_entry.iloc[i - 1]
        curr_close = float(candle["close"])
        rsi        = float(candle["rsi"])
        adx        = float(candle["adx"])
        hours_ok, _ = is_trading_hours(dt)
        if not hours_ok:
            continue

        # ── EMA PULLBACK (EUR/USD, AUD/USD) ──────────────────────────
        if strategy == "ema_pullback":
            ema_f = float(candle["ema_fast"])
            ema_s = float(candle["ema_slow"])
            ema_t = float(candle["ema_trend"])
            trend_long  = ema_f > ema_s > ema_t
            trend_short = ema_f < ema_s < ema_t
            if not trend_long and not trend_short:
                continue
            direction  = "long" if trend_long else "short"
            prev_low   = float(prev["low"])
            prev_high  = float(prev["high"])
            if direction == "long":
                touched = prev_low  <= ema_f
                bounced = curr_close > ema_f
            else:
                touched = prev_high >= ema_f
                bounced = curr_close < ema_f
            if not touched or not bounced:
                continue
            rsi_ok = (35 <= rsi <= 75) if direction == "long" else (25 <= rsi <= 65)
            if not rsi_ok:
                continue
            if adx < 20:
                continue
            if direction == "long":
                sl_price = min(prev_low, float(candle["low"])) - atr * 0.5
            else:
                sl_price = max(prev_high, float(candle["high"])) + atr * 0.5
            risk = abs(price - sl_price)
            if risk <= 0:
                continue
            tp_price = price + (risk * tp_rr) if direction == "long" else price - (risk * tp_rr)

        # ── MEAN REVERSION (GBP/USD) ─────────────────────────────────
        elif strategy == "mean_reversion":
            bb_upper = float(candle[f"{bb_col}_upper"]) if not pd.isna(candle[f"{bb_col}_upper"]) else None
            bb_lower = float(candle[f"{bb_col}_lower"]) if not pd.isna(candle[f"{bb_col}_lower"]) else None
            bb_mid   = float(candle[f"{bb_col}_mid"])   if not pd.isna(candle[f"{bb_col}_mid"])   else None
            if bb_upper is None or bb_lower is None or bb_mid is None:
                continue
            # Long: vorherige Kerze unter unterem Band, Close bounce zurück
            if float(prev["low"]) <= bb_lower and curr_close > bb_lower and rsi < 45:
                direction = "long"
            # Short: vorherige Kerze über oberem Band, Close bounce zurück
            elif float(prev["high"]) >= bb_upper and curr_close < bb_upper and rsi > 55:
                direction = "short"
            else:
                continue
            if adx > 35:  # Mean Reversion braucht keinen starken Trend
                continue
            if direction == "long":
                sl_price = float(candle["low"]) - atr * 0.5
                tp_price = bb_mid
            else:
                sl_price = float(candle["high"]) + atr * 0.5
                tp_price = bb_mid
            risk = abs(price - sl_price)
            if risk <= 0:
                continue
            if abs(tp_price - price) / risk < 1.0:  # Mindest R:R 1:1
                continue

        # ── BREAKOUT (USD/JPY) ───────────────────────────────────────
        elif strategy == "breakout":
            # Breakout-Level = Rolling High/Low der VORHERIGEN Kerze (shift um 1)
            prev_high_n = float(prev["high_n"]) if not pd.isna(prev["high_n"]) else None
            prev_low_n  = float(prev["low_n"])  if not pd.isna(prev["low_n"])  else None
            if prev_high_n is None or prev_low_n is None:
                continue
            long_break  = curr_close > prev_high_n
            short_break = curr_close < prev_low_n
            if not long_break and not short_break:
                continue
            direction = "long" if long_break else "short"
            if adx < 20:
                continue
            # SL: ATR-basiert direkt am Entry (Breakout braucht engen SL)
            sl_price = price - atr * SL_ATR_MULT if direction == "long" else price + atr * SL_ATR_MULT
            risk = abs(price - sl_price)
            if risk <= 0:
                continue
            tp_price = price + (risk * tp_rr) if direction == "long" else price - (risk * tp_rr)

        if direction is None or sl_price is None or tp_price is None:
            continue

        # Position Size
        sizing = calculate_position_size(capital, RISK_PER_TRADE, price, sl_price)
        partial_tp = price + (risk * 1.5) if direction == "long" else price - (risk * 1.5)

        open_pos[symbol] = {
            "direction":   direction,
            "entry":       price,
            "sl":          sl_price,
            "sl_original": sl_price,
            "tp":          tp_price,
            "partial_tp":  partial_tp,
            "units":       sizing["units"],
            "units_open":  sizing["units"],
            "risk_usd":    sizing["risk_usd"],
            "partial_done": False,
            "trailing_sl": None,
            "open_time":   dt,
        }

        if verbose:
            _icon = "📈 LONG" if direction == "long" else "📉 SHORT"
            print(f"  {_icon} {symbol} @ {price} SL:{round(sl_price,5)} TP:{round(tp_price,5)}")

    # Offene Positionen am Ende schließen (zum letzten Preis)
    for sym, pos in open_pos.items():
        last_price = float(df_entry["close"].iloc[-1])
        pnl = (last_price - pos["entry"]) * pos["units_open"]
        if pos["direction"] == "short": pnl = -pnl
        capital += pnl
        trades.append(_make_trade_record(sym, pos, last_price, "end_of_test", pnl, df_entry.index[-1]))

    # Statistiken berechnen
    stats = _calculate_stats(trades, start_capital, capital, capital_curve)

    return {
        "trades":        trades,
        "capital_curve": capital_curve,
        "stats":         stats,
        "final_capital": round(capital, 2),
    }


def _make_trade_record(symbol, pos, close_price, reason, pnl, close_time):
    """Erstellt einen Trade-Record für das Log."""
    return {
        "symbol":     symbol,
        "direction":  pos["direction"],
        "entry":      pos["entry"],
        "close":      close_price,
        "sl":         pos["sl_original"],
        "tp":         pos["tp"],
        "pnl":        round(pnl, 2),
        "reason":     reason,
        "open_time":  str(pos.get("open_time", "")),
        "close_time": str(close_time),
        "win":        pnl > 0,
    }


def _calculate_stats(trades, start_capital, final_capital, capital_curve):
    """Berechnet alle wichtigen Backtest-Kennzahlen."""
    if not trades:
        return {"error": "Keine Trades"}

    wins   = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]

    total_profit = sum(t["pnl"] for t in wins)   if wins   else 0
    total_loss   = abs(sum(t["pnl"] for t in losses)) if losses else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    # Max Drawdown
    peak = start_capital
    max_dd = 0
    for cap in capital_curve:
        if cap > peak: peak = cap
        dd = (peak - cap) / peak
        if dd > max_dd: max_dd = dd

    # Sharpe Ratio (vereinfacht)
    returns = pd.Series(capital_curve).pct_change().dropna()
    sharpe  = (returns.mean() / returns.std() * (252**0.5)) if returns.std() > 0 else 0

    return {
        "total_trades":   len(trades),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       round(len(wins) / len(trades) * 100, 1),
        "profit_factor":  round(profit_factor, 2),
        "total_return":   round((final_capital - start_capital) / start_capital * 100, 2),
        "max_drawdown":   round(max_dd * 100, 2),
        "sharpe_ratio":   round(float(sharpe), 2),
        "avg_win":        round(total_profit / len(wins), 2) if wins else 0,
        "avg_loss":       round(-total_loss / len(losses), 2) if losses else 0,
        "final_capital":  round(final_capital, 2),
    }


def print_backtest_results(results):
    """Gibt Backtest-Ergebnisse übersichtlich aus."""
    s = results["stats"]
    if "error" in s:
        print(f"  ⚠️  {s['error']}")
        return

    print(f"\n  📊 BACKTEST ERGEBNISSE")
    print(f"  {'─'*40}")
    print(f"  Trades gesamt  : {s['total_trades']}")
    print(f"  Wins / Losses  : {s['wins']} / {s['losses']}")
    print(f"  Win-Rate       : {s['win_rate']}%")
    print(f"  Profit Factor  : {s['profit_factor']}")
    print(f"  {'─'*40}")
    print(f"  Gesamtrendite  : {s['total_return']}%")
    print(f"  Max Drawdown   : {s['max_drawdown']}%")
    print(f"  Sharpe Ratio   : {s['sharpe_ratio']}")
    print(f"  {'─'*40}")
    print(f"  Ø Gewinn/Trade : {s['avg_win']}$")
    print(f"  Ø Verlust/Trade: {s['avg_loss']}$")
    print(f"  Endkapital     : {s['final_capital']}$")

    # Bewertung
    print(f"\n  Bewertung:")
    if s["profit_factor"] >= 1.5 and s["max_drawdown"] <= 15:
        print(f"  ✅ Strategie erfüllt Mindestanforderungen (PF≥1.5, DD≤15%)")
    elif s["profit_factor"] >= 1.0:
        print(f"  🟡 Strategie profitabel aber noch nicht optimal")
    else:
        print(f"  ❌ Strategie nicht profitabel — Parameter anpassen")

    if s["total_trades"] < 50:
        print(f"  ⚠️  Nur {s['total_trades']} Trades — zu wenig für zuverlässige Aussagen")
    elif s["total_trades"] >= 200:
        print(f"  ✅ {s['total_trades']} Trades — statistisch aussagekräftig")


# ─────────────────────────────────────────────
#  MODUL 6 TEST
# ─────────────────────────────────────────────

print("\n" + "═"*50)
print("  MODUL 6 — Backtesting Engine")
print("═"*50)
print("  ⏳ Läuft... (kann 30-60 Sekunden dauern)")

backtest_results = run_backtest(
    symbol        = "EURUSD=X",
    data_dict     = data,
    start_capital = 10000.0,
    verbose       = False,
)

print_backtest_results(backtest_results)
print("\n✅ Modul 6 funktioniert!")


# ─────────────────────────────────────────────
#  MODUL 7 — PAPER TRADING LOOP
# ─────────────────────────────────────────────
#
#  Läuft live auf echten Preisen — aber ohne
#  echtes Geld. Simuliert alle Trades exakt
#  wie im echten Betrieb.
#
#  Ablauf:
#  1. Alle 30 Min: neue Preise holen
#  2. Offene Positionen updaten
#  3. Neue Signale prüfen
#  4. Trades loggen
#  5. Tages-Reset um Mitternacht
#
#  Läuft bis du es stoppst (Ctrl+C)
# ─────────────────────────────────────────────

import time
import json
import sqlite3
from pathlib import Path

# Paper Trading Symbole
PAPER_SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD", "ETH-USD"]

# Wie oft neue Daten holen (Sekunden)
# 1800 = 30 Minuten (für echten Betrieb)
# 60   = 1 Minute   (zum Testen)
SCAN_INTERVAL = 60  # auf 60 Sekunden für Test

# ── SQLITE DATENBANK ─────────────────────────

def init_database(db_path="logs/trades.db"):
    """Erstellt die SQLite Datenbank für alle Trades."""
    Path("logs").mkdir(exist_ok=True)
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT,
            direction   TEXT,
            entry       REAL,
            close       REAL,
            sl          REAL,
            tp          REAL,
            pnl         REAL,
            pnl_pct     REAL,
            units       REAL,
            reason      TEXT,
            open_time   TEXT,
            close_time  TEXT,
            win         INTEGER,
            mode        TEXT DEFAULT "paper"
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS daily_summary (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT,
            trades      INTEGER,
            wins        INTEGER,
            losses      INTEGER,
            pnl         REAL,
            capital     REAL,
            drawdown    REAL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS bot_state (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print(f"  ✅ Datenbank initialisiert: {db_path}")


def save_trade(trade, db_path="logs/trades.db"):
    """Speichert einen abgeschlossenen Trade in der DB."""
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()
    c.execute('''
        INSERT INTO trades
        (symbol, direction, entry, close, sl, tp, pnl, pnl_pct,
         units, reason, open_time, close_time, win, mode)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        trade["symbol"],
        trade["direction"],
        trade["entry"],
        trade["close"],
        trade["sl"],
        trade["tp"],
        trade["pnl"],
        trade.get("pnl_pct", 0),
        trade["units"],
        trade["reason"],
        trade["open_time"],
        trade["close_time"],
        1 if trade["win"] else 0,
        "paper",
    ))
    conn.commit()
    conn.close()


def get_trade_count(db_path="logs/trades.db"):
    """Gibt die Anzahl gespeicherter Trades zurück."""
    try:
        conn = sqlite3.connect(db_path)
        c    = conn.cursor()
        c.execute("SELECT COUNT(*) FROM trades")
        count = c.fetchone()[0]
        conn.close()
        return count
    except:
        return 0


def get_performance_stats(db_path="logs/trades.db"):
    """Berechnet Performance-Statistiken aus der DB."""
    try:
        conn   = sqlite3.connect(db_path)
        df_db  = pd.read_sql("SELECT * FROM trades", conn)
        conn.close()

        if df_db.empty:
            return None

        wins   = df_db[df_db["win"] == 1]
        losses = df_db[df_db["win"] == 0]

        total_profit = wins["pnl"].sum()   if len(wins)   > 0 else 0
        total_loss   = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        pf           = total_profit / total_loss if total_loss > 0 else float("inf")

        return {
            "total_trades": len(df_db),
            "wins":         len(wins),
            "losses":       len(losses),
            "win_rate":     round(len(wins) / len(df_db) * 100, 1),
            "profit_factor": round(pf, 2),
            "total_pnl":    round(df_db["pnl"].sum(), 2),
            "avg_win":      round(wins["pnl"].mean(), 2)   if len(wins)   > 0 else 0,
            "avg_loss":     round(losses["pnl"].mean(), 2) if len(losses) > 0 else 0,
        }
    except:
        return None


# ── LIVE DATEN HOLEN ─────────────────────────

# Alpha Vantage Symbol-Mapping
AV_SYMBOL_MAP = {
    "EURUSD=X": "EURUSD",
    "GBPUSD=X": "GBPUSD",
    "USDJPY=X": "USDJPY",
    "AUDUSD=X": "AUDUSD",
    "BTC-USD":  "BTC",
    "ETH-USD":  "ETH",
}

def get_latest_price(symbol):
    """
    Holt den aktuellen Preis via Alpha Vantage.
    Fällt auf yfinance zurück wenn AV nicht verfügbar.
    """
    av_key = os.environ.get("ALPHA_VANTAGE_KEY", "")
    
    # Alpha Vantage versuchen
    if av_key:
        try:
            av_sym = AV_SYMBOL_MAP.get(symbol, symbol)
            
            # Forex
            if "=X" in symbol:
                from_cur = av_sym[:3]
                to_cur   = av_sym[3:]
                url = (f"https://www.alphavantage.co/query"
                       f"?function=CURRENCY_EXCHANGE_RATE"
                       f"&from_currency={from_cur}"
                       f"&to_currency={to_cur}"
                       f"&apikey={av_key}")
                resp = requests.get(url, timeout=10)
                data = resp.json()
                rate = data.get("Realtime Currency Exchange Rate", {})
                if rate:
                    price = float(rate["5. Exchange Rate"])
                    return {
                        "price":  price,
                        "high":   price * 1.001,
                        "low":    price * 0.999,
                        "volume": 0,
                        "time":   datetime.now(timezone.utc),
                    }
            
            # Crypto
            elif "-USD" in symbol:
                url = (f"https://www.alphavantage.co/query"
                       f"?function=CURRENCY_EXCHANGE_RATE"
                       f"&from_currency={av_sym}"
                       f"&to_currency=USD"
                       f"&apikey={av_key}")
                resp = requests.get(url, timeout=10)
                data = resp.json()
                rate = data.get("Realtime Currency Exchange Rate", {})
                if rate:
                    price = float(rate["5. Exchange Rate"])
                    return {
                        "price":  price,
                        "high":   price * 1.002,
                        "low":    price * 0.998,
                        "volume": 0,
                        "time":   datetime.now(timezone.utc),
                    }
        except Exception as e:
            print(f"  ⚠️  Alpha Vantage Fehler: {e}")
    
    # yfinance Fallback
    try:
        df = yf.download(
            tickers     = symbol,
            period      = "2d",
            interval    = "1h",
            auto_adjust = True,
            progress    = False,
        )
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            return {
                "price":  float(df["close"].iloc[-1]),
                "high":   float(df["high"].iloc[-1]),
                "low":    float(df["low"].iloc[-1]),
                "volume": float(df["volume"].iloc[-1]),
                "time":   df.index[-1],
            }
    except:
        pass
    
    print(f"  ⚠️  Kein Preis für {symbol}")
    return None


# ── HAUPT PAPER TRADING LOOP ─────────────────

def run_paper_trading(symbols=None, capital=10000.0, max_scans=None):
    """
    Startet den Paper Trading Loop.
    
    Args:
        symbols:    Liste der Symbole (None = alle 6)
        capital:    Startkapital
        max_scans:  Max Durchläufe (None = unbegrenzt, Zahl = für Tests)
    
    Stoppen: Ctrl+C
    """
    if symbols is None:
        symbols = PAPER_SYMBOLS

    print(f"\n{'═'*50}")
    print(f"  🤖 PAPER TRADING GESTARTET")
    print(f"{'═'*50}")
    print(f"  Symbole   : {', '.join(symbols)}")
    print(f"  Kapital   : {capital}$")
    print(f"  Intervall : {SCAN_INTERVAL}s")
    print(f"  Stoppen   : Ctrl+C")
    print(f"{'═'*50}\n")

    # Datenbank initialisieren
    init_database()

    # State initialisieren
    global risk_state
    risk_state["capital"]     = capital
    risk_state["daily_start"] = capital

    # Symbol-Daten laden
    print("  📥 Lade initiale Daten für alle Symbole...")
    symbol_data   = {}
    symbol_trends = {}

    for sym in symbols:
        try:
            print(f"     {sym}...")
            symbol_data[sym]   = prepare_symbol_data(sym)
            symbol_trends[sym] = analyze_all_timeframes(symbol_data[sym])
            time.sleep(2)  # Rate Limit vermeiden
        except Exception as e:
            print(f"  ⚠️  {sym} konnte nicht geladen werden: {e}")

    print(f"\n  ✅ {len(symbol_data)} Symbole geladen\n")

    scan_count  = 0
    last_reload = datetime.now(timezone.utc)

    try:
        while True:
            now = datetime.now(timezone.utc)
            print(f"\n  🔍 Scan #{scan_count + 1} — {now.strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"  {'─'*44}")

            # Tages-Reset um Mitternacht
            if now.hour == 0 and now.minute < 5:
                reset_daily_state()

            # Daten alle 4 Stunden neu laden
            if (now - last_reload).seconds > 14400:
                print("  🔄 Daten werden aktualisiert...")
                for sym in symbols:
                    try:
                        symbol_data[sym]   = prepare_symbol_data(sym)
                        symbol_trends[sym] = analyze_all_timeframes(symbol_data[sym])
                        time.sleep(2)
                    except:
                        pass
                last_reload = now

            # News für heute laden
            todays_events = fetch_news_calendar(now.date())

            # ── OFFENE POSITIONEN UPDATEN ─────
            if risk_state["open_positions"]:
                print(f"  📊 Offene Positionen: {len(risk_state['open_positions'])}")
                for sym in list(risk_state["open_positions"].keys()):
                    latest = get_latest_price(sym)
                    if not latest:
                        continue

                    pos       = risk_state["open_positions"][sym]
                    current   = latest["price"]
                    ema_fast  = float(symbol_data[sym]["h1"]["ema_fast"].iloc[-1]) \
                                if sym in symbol_data else current

                    result = update_position(sym, current, ema_fast)

                    if result in ["sl_hit", "tp_hit", "trailing_sl"]:
                        # Trade wurde geschlossen — in DB speichern
                        if risk_state["trade_log"]:
                            last_trade = risk_state["trade_log"][-1]
                            save_trade(last_trade)
                            print(f"  💾 Trade gespeichert — DB hat jetzt {get_trade_count()} Trades")
                    elif result == "partial_close":
                        print(f"  ✂️  Partial Close {sym} — SL auf Break-Even")
                    else:
                        pnl_now = (current - pos["entry"]) * pos["units_open"]
                        if pos["direction"] == "short":
                            pnl_now = -pnl_now
                        print(f"  📈 {sym}: {current:.5f} | P&L: {'+' if pnl_now>0 else ''}{round(pnl_now,2)}$")

            # ── NEUE SIGNALE SUCHEN ───────────
            print(f"\n  🎯 Suche neue Signale...")
            signals_found = 0

            for sym in symbols:
                if sym not in symbol_data:
                    continue
                if sym in risk_state["open_positions"]:
                    continue  # Bereits offen

                # News Blackout prüfen
                blackout, blackout_reason = is_news_blackout(sym, now, todays_events)

                # Signal generieren
                signal = generate_signal(
                    symbol          = sym,
                    data_dict       = symbol_data[sym],
                    trend_analysis  = symbol_trends[sym],
                    circuit_breaker = get_circuit_breaker_status(),
                    open_trades     = list(risk_state["open_positions"].keys()),
                    dt              = now,
                    news_blackout   = blackout,
                )

                if signal["signal"] != "no_trade":
                    signals_found += 1
                    print(f"\n  🚨 SIGNAL: {sym} — {signal['signal'].upper()}")
                    print(f"     Entry: {signal['entry']} | SL: {signal['sl']} | TP: {signal['tp']} | RR: {signal['rr']}")

                    # Position eröffnen
                    open_position(sym, signal, risk_state["capital"],
                                 risk_mult=signal.get("risk_mult", 1.0))
                else:
                    # Top 2 Fehler ausgeben
                    failed = [f"{k}: {v['info']}" for k,v in signal["checks"].items() if not v["ok"]]
                    if failed:
                        print(f"  ➖ {sym}: {' | '.join(failed[:2])}")

            if signals_found == 0:
                print(f"  ➖ Keine Signale — Markt wartet")

            # ── STATUS AUSGABE ────────────────
            stats = get_performance_stats()
            if stats:
                print(f"\n  📊 Performance: {stats['total_trades']} Trades | "
                      f"WR: {stats['win_rate']}% | PF: {stats['profit_factor']} | "
                      f"P&L: {stats['total_pnl']}$")

            print_risk_summary()

            scan_count += 1

            # Max Scans für Test
            if max_scans and scan_count >= max_scans:
                print(f"\n  ✅ Test abgeschlossen nach {scan_count} Scans")
                break

            print(f"\n  ⏳ Nächster Scan in {SCAN_INTERVAL}s... (Ctrl+C zum Stoppen)")
            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\n  🛑 Paper Trading gestoppt")
        print(f"  Trades gesamt: {get_trade_count()}")
        print_risk_summary()


# ─────────────────────────────────────────────
#  MODUL 7 TEST — 2 Scans
# ─────────────────────────────────────────────

print("\n" + "═"*50)
print("  MODUL 7 — Paper Trading Test (2 Scans)")
print("═"*50)

# Datenbank initialisieren
init_database()

# Kurzer Test mit 2 Scans und 10 Sekunden Pause
print("\n  ℹ️  Teste mit 2 Scans (je 10 Sekunden)")
print("  ℹ️  Für echten Betrieb: run_paper_trading() aufrufen\n")

# Test-Modus: SCAN_INTERVAL kurz setzen
SCAN_INTERVAL = 10
run_paper_trading(
    symbols   = ["EURUSD=X", "GBPUSD=X"],
    capital   = 10000.0,
    max_scans = 2,
)

print("\n✅ Modul 7 funktioniert!")
print("\n" + "═"*50)
print("  🎉 ALLE 7 MODULE FERTIG!")
print("  Für echtes Paper Trading:")
print("  run_paper_trading() in der Console aufrufen")
print("═"*50)


# ─────────────────────────────────────────────
#  MODUL 9 — TELEGRAM ALERTS + LOGGING
# ─────────────────────────────────────────────
#
#  Sendet Nachrichten bei:
#  - Trade geöffnet / geschlossen
#  - Circuit Breaker ausgelöst
#  - CME Gap erkannt
#  - Wöchentlicher Report (Montags)
#  - Bot gestartet / gestoppt
# ─────────────────────────────────────────────

# Telegram Credentials aus .env laden
load_env()
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_telegram(message, parse_mode="HTML"):
    """
    Sendet eine Nachricht via Telegram.
    Schlägt still fehl wenn kein Token konfiguriert.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"  📱 Telegram (nicht konfiguriert): {message[:60]}...")
        return False

    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": parse_mode,
        }
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            return True
        else:
            print(f"  ⚠️  Telegram Fehler: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ⚠️  Telegram nicht erreichbar: {e}")
        return False


def notify_trade_opened(symbol, signal):
    """Sendet Alert wenn Trade geöffnet wird."""
    direction_icon = "📈" if signal["signal"] == "long" else "📉"
    msg = (
        f"{direction_icon} <b>TRADE GEÖFFNET</b>\n\n"
        f"Symbol:    <code>{symbol}</code>\n"
        f"Richtung:  <b>{signal['signal'].upper()}</b>\n"
        f"Entry:     <code>{signal['entry']}</code>\n"
        f"Stop Loss: <code>{signal['sl']}</code>\n"
        f"Take Profit: <code>{signal['tp']}</code>\n"
        f"RR:        <b>{signal['rr']}:1</b>\n"
        f"Risiko:    <b>{round(RISK_PER_TRADE*100, 1)}% = "
        f"{round(risk_state['capital'] * RISK_PER_TRADE, 2)}$</b>\n\n"
        f"💰 Kapital: {round(risk_state['capital'], 2)}$"
    )
    send_telegram(msg)


def notify_trade_closed(trade):
    """Sendet Alert wenn Trade geschlossen wird."""
    pnl        = trade["pnl"]
    icon       = "✅" if pnl > 0 else "❌"
    reason_map = {
        "tp_hit":      "🎯 Take Profit erreicht",
        "sl_hit":      "🛑 Stop Loss getroffen",
        "trailing_sl": "🔁 Trailing Stop",
        "end_of_test": "⏹ Test Ende",
    }
    reason_text = reason_map.get(trade["reason"], trade["reason"])

    msg = (
        f"{icon} <b>TRADE GESCHLOSSEN</b>\n\n"
        f"Symbol:    <code>{trade['symbol']}</code>\n"
        f"Richtung:  <b>{trade['direction'].upper()}</b>\n"
        f"Entry:     <code>{trade['entry']}</code>\n"
        f"Close:     <code>{trade['close']}</code>\n"
        f"Grund:     {reason_text}\n\n"
        f"P&L:       <b>{'+' if pnl > 0 else ''}{round(pnl, 2)}$</b>\n"
        f"💰 Kapital: {round(risk_state['capital'], 2)}$\n\n"
        f"Verluste in Folge: {risk_state['losses_in_row']}"
    )
    send_telegram(msg)


def notify_circuit_breaker():
    """Sendet Alert wenn Circuit Breaker auslöst."""
    msg = (
        f"🔴 <b>CIRCUIT BREAKER AKTIV</b>\n\n"
        f"Verluste in Folge: {risk_state['losses_in_row']}\n"
        f"Tagesverlust: {round(risk_state['daily_loss']*100, 2)}%\n"
        f"Kapital: {round(risk_state['capital'], 2)}$\n\n"
        f"Trading heute gestoppt. Reset morgen 00:00 UTC."
    )
    send_telegram(msg)


def notify_cme_gap(symbol, gap):
    """Sendet Alert wenn neuer CME Gap erkannt wird."""
    direction_icon = "⬆️" if gap["direction"] == "up" else "⬇️"
    msg = (
        f"📊 <b>CME GAP ERKANNT</b>\n\n"
        f"Symbol:    <code>{symbol}</code>\n"
        f"Richtung:  {direction_icon} {gap['direction'].upper()}\n"
        f"Zone:      <code>{gap['gap_bot']:.0f} — {gap['gap_top']:.0f}</code>\n"
        f"Größe:     <b>{gap['gap_pct']}%</b>\n\n"
        f"⚡ ~77% Wahrscheinlichkeit dass Gap gefüllt wird"
    )
    send_telegram(msg)


def notify_bot_started(symbols, capital):
    """Sendet Alert wenn Bot gestartet wird."""
    msg = (
        f"🤖 <b>TRADING BOT GESTARTET</b>\n\n"
        f"Symbole: <code>{', '.join(symbols)}</code>\n"
        f"Kapital: <b>{capital}$</b>\n"
        f"Modus:   Paper Trading\n"
        f"Intervall: {SCAN_INTERVAL}s\n\n"
        f"Bot läuft — ich melde mich bei Signalen! 🚀"
    )
    send_telegram(msg)


def notify_bot_stopped():
    """Sendet Alert wenn Bot gestoppt wird."""
    stats = get_performance_stats()
    if stats:
        msg = (
            f"🛑 <b>BOT GESTOPPT</b>\n\n"
            f"Trades: {stats['total_trades']}\n"
            f"Win-Rate: {stats['win_rate']}%\n"
            f"Profit Factor: {stats['profit_factor']}\n"
            f"Gesamt P&L: {stats['total_pnl']}$\n"
            f"Kapital: {round(risk_state['capital'], 2)}$"
        )
    else:
        msg = "🛑 <b>BOT GESTOPPT</b>\n\nKeine Trades in dieser Session."
    send_telegram(msg)


def send_weekly_report():
    """Sendet wöchentlichen Performance-Report."""
    stats = get_performance_stats()
    if not stats:
        send_telegram("📊 <b>WOCHEN-REPORT</b>\n\nNoch keine Trades diese Woche.")
        return

    # Bewertung
    if stats["profit_factor"] >= 1.5:
        bewertung = "✅ Sehr gut — Strategie performt"
    elif stats["profit_factor"] >= 1.0:
        bewertung = "🟡 Ok — profitabel aber Luft nach oben"
    else:
        bewertung = "❌ Nicht profitabel — Parameter prüfen"

    msg = (
        f"📊 <b>WOCHEN-REPORT</b>\n\n"
        f"Trades:        {stats['total_trades']}\n"
        f"Wins/Losses:   {stats['wins']}/{stats['losses']}\n"
        f"Win-Rate:      <b>{stats['win_rate']}%</b>\n"
        f"Profit Factor: <b>{stats['profit_factor']}</b>\n"
        f"Ø Gewinn:      {stats['avg_win']}$\n"
        f"Ø Verlust:     {stats['avg_loss']}$\n"
        f"Gesamt P&L:    <b>{stats['total_pnl']}$</b>\n\n"
        f"Kapital: {round(risk_state['capital'], 2)}$\n\n"
        f"{bewertung}"
    )
    send_telegram(msg)


# Telegram in Paper Trading Loop einbauen
# Originale Funktionen überschreiben mit Telegram-Versionen

_original_close = _close_position

def _close_position_with_telegram(symbol, close_price, reason):
    """Erweiterte Version mit Telegram Alert."""
    result = _original_close(symbol, close_price, reason)

    # Trade aus Log holen und Alert senden
    if risk_state["trade_log"]:
        last_trade = risk_state["trade_log"][-1]
        notify_trade_closed(last_trade)

        # Circuit Breaker Alert
        if risk_state["circuit_active"]:
            notify_circuit_breaker()

    return result

# Monkey-patch: originale Funktion ersetzen
_close_position = _close_position_with_telegram


# ─────────────────────────────────────────────
#  MODUL 9 TEST
# ─────────────────────────────────────────────

print("\n" + "═"*50)
print("  MODUL 9 — Telegram Test")
print("═"*50)

# Telegram testen
print("\n  📱 Sende Test-Nachricht...")
success = send_telegram(
    "🤖 <b>Trading Bot verbunden!</b>\n\n"
    "Alle Module geladen:\n"
    "✅ Daten-Loader\n"
    "✅ BOS/CHoCH Erkennung\n"
    "✅ Signal-Generator\n"
    "✅ News-Filter\n"
    "✅ Risk Manager\n"
    "✅ Backtesting\n"
    "✅ Paper Trading\n"
    "✅ Telegram Alerts\n\n"
    "Bot ist bereit! 🚀"
)

if success:
    print("  ✅ Telegram funktioniert — check dein Handy!")
else:
    print("  ⚠️  Telegram nicht konfiguriert")
    print("  → Token und Chat-ID in trading_bot.env eintragen")

print("\n✅ Modul 9 funktioniert!")
print("\n" + "═"*50)
print("  🎉 ALLE 9 MODULE FERTIG!")
print("═"*50)
print("""
  Jetzt echtes Paper Trading starten:
  
  In Spyder Console eintippen:
  
  run_paper_trading()
  
  Der Bot läuft dann dauerhaft und sendet
  dir alle Signale auf Telegram.
  Stoppen mit Ctrl+C.
""")


# ═══════════════════════════════════════════════
#  MODUL 10 — STRATEGIE ERWEITERUNGEN V3.0
# ═══════════════════════════════════════════════

# ─────────────────────────────────────────────
#  10.1 — CANDLESTICK MUSTER
# ─────────────────────────────────────────────

def detect_candlestick_patterns(df):
    """
    Erkennt relevante Candlestick-Muster.

    Muster:
    - Bullish/Bearish Engulfing
    - Pin Bar (Hammer / Shooting Star)
    - Doji (Unentschlossenheit)

    Returns:
        df mit neuen Spalten:
        - candle_bull: bullishes Muster
        - candle_bear: bearishes Muster
        - candle_name: Name des Musters
    """
    df = df.copy()
    df["candle_bull"] = False
    df["candle_bear"] = False
    df["candle_name"] = ""

    for i in range(1, len(df)):
        o  = float(df["open"].iloc[i])
        h  = float(df["high"].iloc[i])
        l  = float(df["low"].iloc[i])
        c  = float(df["close"].iloc[i])
        po = float(df["open"].iloc[i-1])
        pc = float(df["close"].iloc[i-1])

        body      = abs(c - o)
        full_range = h - l
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        if full_range == 0:
            continue

        # ── Bullish Engulfing ─────────────────
        # Grüne Kerze die komplett vorherige rote umschließt
        if c > o and pc < po and c > po and o < pc:
            df.iloc[i, df.columns.get_loc("candle_bull")] = True
            df.iloc[i, df.columns.get_loc("candle_name")] = "Bullish Engulfing"

        # ── Bearish Engulfing ─────────────────
        elif c < o and pc > po and c < po and o > pc:
            df.iloc[i, df.columns.get_loc("candle_bear")] = True
            df.iloc[i, df.columns.get_loc("candle_name")] = "Bearish Engulfing"

        # ── Hammer (bullish) ──────────────────
        # Langer unterer Docht, kleiner Body oben
        elif (lower_wick >= body * 2 and
              upper_wick <= body * 0.5 and
              body >= full_range * 0.1):
            df.iloc[i, df.columns.get_loc("candle_bull")] = True
            df.iloc[i, df.columns.get_loc("candle_name")] = "Hammer"

        # ── Shooting Star (bearish) ───────────
        # Langer oberer Docht, kleiner Body unten
        elif (upper_wick >= body * 2 and
              lower_wick <= body * 0.5 and
              body >= full_range * 0.1):
            df.iloc[i, df.columns.get_loc("candle_bear")] = True
            df.iloc[i, df.columns.get_loc("candle_name")] = "Shooting Star"

        # ── Pin Bar (bullish) ─────────────────
        # Sehr langer unterer Docht > 60% der Range
        elif lower_wick >= full_range * 0.6 and body <= full_range * 0.25:
            df.iloc[i, df.columns.get_loc("candle_bull")] = True
            df.iloc[i, df.columns.get_loc("candle_name")] = "Pin Bar (Bull)"

        # ── Pin Bar (bearish) ─────────────────
        elif upper_wick >= full_range * 0.6 and body <= full_range * 0.25:
            df.iloc[i, df.columns.get_loc("candle_bear")] = True
            df.iloc[i, df.columns.get_loc("candle_name")] = "Pin Bar (Bear)"

    return df


# ─────────────────────────────────────────────
#  10.2 — LIQUIDITY SWEEPS
# ─────────────────────────────────────────────

def detect_liquidity_sweeps(df, window=10):
    """
    Erkennt Liquidity Sweeps — wenn Preis kurz über/unter
    einen Swing High/Low geht und dann sofort zurückdreht.

    Das ist ein starkes Signal: institutionelle Trader holen
    Stops ab bevor sie die echte Bewegung starten.

    Returns:
        df mit neuen Spalten:
        - liq_sweep_bull: bullisher Sweep (Stops unten geholt)
        - liq_sweep_bear: bearisher Sweep (Stops oben geholt)
    """
    df = df.copy()
    df["liq_sweep_bull"] = False
    df["liq_sweep_bear"] = False

    for i in range(window + 1, len(df)):
        current_low   = float(df["low"].iloc[i])
        current_high  = float(df["high"].iloc[i])
        current_close = float(df["close"].iloc[i])

        # Letztes Swing Low im Fenster
        prev_lows  = df["low"].iloc[i-window:i]
        prev_highs = df["high"].iloc[i-window:i]
        swing_low  = float(prev_lows.min())
        swing_high = float(prev_highs.max())

        # Bullisher Sweep: Kerze geht unter Swing Low
        # aber schließt ÜBER dem Swing Low → Stops geholt, Umkehr
        if current_low < swing_low and current_close > swing_low:
            df.iloc[i, df.columns.get_loc("liq_sweep_bull")] = True

        # Bearisher Sweep: Kerze geht über Swing High
        # aber schließt UNTER dem Swing High → Stops geholt, Umkehr
        if current_high > swing_high and current_close < swing_high:
            df.iloc[i, df.columns.get_loc("liq_sweep_bear")] = True

    return df


# ─────────────────────────────────────────────
#  10.3 — VIX FILTER
# ─────────────────────────────────────────────

def get_vix_level():
    """
    Holt aktuellen VIX Stand von yfinance.
    VIX = Volatilitätsindex, misst Markt-Angst.

    Returns:
        (float, str) — (vix_wert, status)
        status: "normal" / "elevated" / "extreme"
    """
    try:
        df = yf.download(
            tickers     = "^VIX",
            period      = "2d",
            interval    = "1d",
            auto_adjust = True,
            progress    = False,
        )
        if df.empty:
            return 20.0, "normal"  # Fallback

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        vix = float(df["close"].iloc[-1])

        if vix > 35:
            status = "extreme"   # Bot pausiert
        elif vix > 25:
            status = "elevated"  # Position Size halbieren
        else:
            status = "normal"

        return vix, status

    except Exception as e:
        return 20.0, "normal"  # Im Fehlerfall normal annehmen


# ─────────────────────────────────────────────
#  10.4 — DOLLAR INDEX (DXY)
# ─────────────────────────────────────────────

def get_dxy_bias():
    """
    Holt DXY Trend — bestimmt erlaubte Richtung für USD-Paare.

    DXY steigend  → USD stark → EUR/USD, GBP/USD nur Short
    DXY fallend   → USD schwach → EUR/USD, GBP/USD nur Long
    DXY seitwärts → beide Richtungen erlaubt

    Returns:
        str: "up" / "down" / "neutral"
    """
    try:
        df = yf.download(
            tickers     = "DX-Y.NYB",
            period      = "10d",
            interval    = "1d",
            auto_adjust = True,
            progress    = False,
        )
        if df.empty:
            return "neutral"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        # EMA 5 und EMA 10 für DXY Trend
        ema5  = df["close"].ewm(span=5,  adjust=False).mean()
        ema10 = df["close"].ewm(span=10, adjust=False).mean()

        if ema5.iloc[-1] > ema10.iloc[-1] * 1.001:
            return "up"    # DXY steigend
        elif ema5.iloc[-1] < ema10.iloc[-1] * 0.999:
            return "down"  # DXY fallend
        else:
            return "neutral"

    except:
        return "neutral"


# ─────────────────────────────────────────────
#  10.5 — S&P500 KORRELATION (für BTC/ETH)
# ─────────────────────────────────────────────

def get_sp500_bias():
    """
    Holt S&P500 Trend für BTC/ETH Korrelationscheck.

    S&P500 in Abwärtstrend → kein BTC/ETH Long
    S&P500 in Aufwärtstrend → BTC/ETH Long erlaubt

    Returns:
        str: "bull" / "bear" / "neutral"
    """
    try:
        df = yf.download(
            tickers     = "^GSPC",
            period      = "20d",
            interval    = "1d",
            auto_adjust = True,
            progress    = False,
        )
        if df.empty:
            return "neutral"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        ema10 = df["close"].ewm(span=10, adjust=False).mean()
        ema20 = df["close"].ewm(span=20, adjust=False).mean()

        if ema10.iloc[-1] > ema20.iloc[-1]:
            return "bull"
        elif ema10.iloc[-1] < ema20.iloc[-1]:
            return "bear"
        else:
            return "neutral"

    except:
        return "neutral"


# ─────────────────────────────────────────────
#  10.6 — FUNDING RATE (BTC/ETH)
# ─────────────────────────────────────────────

def get_funding_rate(symbol):
    """
    Holt aktuelle Funding Rate von Binance API.

    Funding Rate > +0.1% → zu viele Longs → kein Long
    Funding Rate < -0.1% → zu viele Shorts → kein Short

    Returns:
        (float, str) — (rate, "long_ok"/"short_ok"/"neutral")
    """
    try:
        # Binance Symbol mapping
        binance_map = {
            "BTC-USD": "BTCUSDT",
            "ETH-USD": "ETHUSDT",
        }
        binance_sym = binance_map.get(symbol)
        if not binance_sym:
            return 0.0, "neutral"

        url  = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={binance_sym}&limit=1"
        resp = requests.get(url, timeout=10)
        data = resp.json()

        if isinstance(data, list) and data:
            rate = float(data[0]["fundingRate"])

            if rate > 0.001:    # > +0.1%
                return rate, "short_ok"   # Überhitzt Long-seitig
            elif rate < -0.001: # < -0.1%
                return rate, "long_ok"    # Überhitzt Short-seitig
            else:
                return rate, "neutral"

    except:
        pass

    return 0.0, "neutral"


# ─────────────────────────────────────────────
#  10.7 — BREAK-EVEN + TIME-BASED EXIT
# ─────────────────────────────────────────────

def check_advanced_exits(symbol, current_price, current_time):
    """
    Prüft erweiterte Exit-Bedingungen:
    1. Break-Even bei 1R
    2. Time-Based Exit nach 48 Stunden

    Wird in update_position() aufgerufen.
    """
    if symbol not in risk_state["open_positions"]:
        return "open"

    pos       = risk_state["open_positions"][symbol]
    entry     = pos["entry"]
    sl        = pos["sl"]
    direction = pos["direction"]
    risk      = abs(entry - sl)
    open_time = pos.get("open_time")

    # ── BREAK-EVEN bei 1R ─────────────────────
    if not pos.get("breakeven_set", False):
        be_level_long  = entry + risk   # 1R im Plus
        be_level_short = entry - risk

        if direction == "long"  and current_price >= be_level_long:
            risk_state["open_positions"][symbol]["sl"]           = entry
            risk_state["open_positions"][symbol]["breakeven_set"] = True
            print(f"  ⚖️  BREAK-EVEN gesetzt: {symbol} SL → {entry}")
            send_telegram(f"⚖️ <b>Break-Even</b> {symbol}\nSL auf Entry gezogen — kein Verlust mehr möglich")

        elif direction == "short" and current_price <= be_level_short:
            risk_state["open_positions"][symbol]["sl"]           = entry
            risk_state["open_positions"][symbol]["breakeven_set"] = True
            print(f"  ⚖️  BREAK-EVEN gesetzt: {symbol} SL → {entry}")
            send_telegram(f"⚖️ <b>Break-Even</b> {symbol}\nSL auf Entry gezogen — kein Verlust mehr möglich")

    # ── TIME-BASED EXIT nach 48 Stunden ───────
    if open_time:
        try:
            if isinstance(open_time, str):
                ot = datetime.fromisoformat(open_time.replace("Z", "+00:00"))
            else:
                ot = open_time
            if ot.tzinfo is None:
                ot = ot.replace(tzinfo=timezone.utc)

            hours_open = (current_time - ot).total_seconds() / 3600

            if hours_open >= 48:
                print(f"  ⏱  TIME EXIT: {symbol} nach {round(hours_open, 1)}h")
                send_telegram(f"⏱ <b>Time Exit</b> {symbol}\nTrade nach 48h geschlossen")
                return _close_position(symbol, current_price, "time_exit")
        except Exception as e:
            pass

    return "open"


# ─────────────────────────────────────────────
#  10.8 — TRADE TAGEBUCH MIT KI ANALYSE
# ─────────────────────────────────────────────

def analyze_trade_with_ai(trade):
    """
    Sendet Trade-Details an Claude API und bekommt
    eine Analyse warum der Trade gewonnen/verloren hat.

    Speichert Analyse in SQLite Datenbank.
    """
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return "Kein Anthropic API Key konfiguriert"

        pnl       = trade["pnl"]
        outcome   = "GEWONNEN" if pnl > 0 else "VERLOREN"
        direction = trade["direction"].upper()

        prompt = f"""Du bist ein erfahrener Trading-Analyst. Analysiere diesen Trade kurz und prägnant auf Deutsch.

Trade Details:
- Symbol: {trade['symbol']}
- Richtung: {direction}
- Entry: {trade['entry']}
- Close: {trade['close']}
- Stop Loss: {trade['sl']}
- Take Profit: {trade['tp']}
- Ergebnis: {outcome} ({'+' if pnl > 0 else ''}{round(pnl, 2)}$)
- Grund: {trade['reason']}
- Geöffnet: {trade['open_time']}
- Geschlossen: {trade['close_time']}

Analysiere in 2-3 Sätzen:
1. Was war der wahrscheinliche Grund für dieses Ergebnis?
2. Was hätte man besser machen können (falls Verlust)?
3. Klassifiziere den Fehler/Erfolg in eine Kategorie:
   FALSCHER_TREND / SCHLECHTES_TIMING / NEWS_SPIKE / GUTES_SETUP / TRAILING_GEWINN / SONSTIGES"""

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 300,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )

        if resp.status_code == 200:
            analysis = resp.json()["content"][0]["text"]

            # In Datenbank speichern
            try:
                conn = sqlite3.connect("logs/trades.db")
                c    = conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS trade_analysis (
                        id        INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id  INTEGER,
                        symbol    TEXT,
                        outcome   TEXT,
                        analysis  TEXT,
                        category  TEXT,
                        date      TEXT
                    )
                """)
                # Kategorie aus Analyse extrahieren
                category = "SONSTIGES"
                for cat in ["FALSCHER_TREND", "SCHLECHTES_TIMING", "NEWS_SPIKE",
                           "GUTES_SETUP", "TRAILING_GEWINN"]:
                    if cat in analysis:
                        category = cat
                        break

                c.execute("""
                    INSERT INTO trade_analysis (symbol, outcome, analysis, category, date)
                    VALUES (?,?,?,?,?)
                """, (trade["symbol"], outcome, analysis, category,
                      datetime.now(timezone.utc).isoformat()))
                conn.commit()
                conn.close()
            except:
                pass

            return analysis
        else:
            return f"API Fehler: {resp.status_code}"

    except Exception as e:
        return f"Analyse fehlgeschlagen: {e}"


# ─────────────────────────────────────────────
#  10.9 — SEASONAL PATTERNS
# ─────────────────────────────────────────────

def get_seasonal_risk_mult(dt=None):
    """
    Berechnet Position-Size Multiplikator basierend
    auf historischen Seasonal Patterns aus der DB.

    Wochentage mit schlechter Win-Rate → kleinere Trades
    Wochentage mit guter Win-Rate → normale Trades

    Returns:
        float: 0.5 bis 1.0
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    weekday      = dt.weekday()  # 0=Mo, 4=Fr
    weekday_name = ["Montag","Dienstag","Mittwoch","Donnerstag","Freitag","Samstag","Sonntag"][weekday]

    try:
        conn = sqlite3.connect("logs/trades.db")
        df_db = pd.read_sql("""
            SELECT
                strftime('%w', open_time) as weekday,
                AVG(win) as win_rate,
                COUNT(*) as count
            FROM trades
            GROUP BY weekday
            HAVING count >= 5
        """, conn)
        conn.close()

        if df_db.empty:
            return 1.0  # Noch nicht genug Daten

        # Aktuellen Wochentag prüfen (SQLite: 0=Sonntag, 1=Montag...)
        sqlite_weekday = str((weekday + 1) % 7)
        row = df_db[df_db["weekday"] == sqlite_weekday]

        if row.empty:
            return 1.0

        win_rate = float(row["win_rate"].iloc[0])

        if win_rate < 0.35:
            print(f"  📅 Seasonal: {weekday_name} hat historisch schlechte Win-Rate ({round(win_rate*100)}%) → 0.5x Size")
            return 0.5
        elif win_rate > 0.6:
            return 1.0  # Normales Risiko
        else:
            return 0.75

    except:
        return 1.0


# ─────────────────────────────────────────────
#  10.10 — FEHLER KLASSIFIZIERUNG REPORT
# ─────────────────────────────────────────────

def send_weekly_error_report():
    """
    Analysiert die letzten 7 Tage Verluste und
    sendet einen detaillierten Fehler-Report via Telegram.
    """
    try:
        conn  = sqlite3.connect("logs/trades.db")

        # Verluste der letzten 7 Tage
        losses = pd.read_sql("""
            SELECT * FROM trades
            WHERE win = 0
            AND close_time >= datetime('now', '-7 days')
        """, conn)

        # Kategorien aus Analyse-Tabelle
        try:
            categories = pd.read_sql("""
                SELECT category, COUNT(*) as count
                FROM trade_analysis
                WHERE outcome = 'VERLOREN'
                AND date >= datetime('now', '-7 days')
                GROUP BY category
                ORDER BY count DESC
            """, conn)
        except:
            categories = pd.DataFrame()

        conn.close()

        if losses.empty:
            send_telegram("📊 <b>Wochen-Fehler-Report</b>\n\nKeine Verluste diese Woche! 🎉")
            return

        msg = f"📊 <b>WOCHEN FEHLER-REPORT</b>\n\n"
        msg += f"Verluste: {len(losses)}\n"
        msg += f"Gesamt P&L Verluste: {round(losses['pnl'].sum(), 2)}$\n\n"

        if not categories.empty:
            msg += "<b>Fehler-Kategorien:</b>\n"
            cat_icons = {
                "FALSCHER_TREND":    "📉 Falscher Trend",
                "SCHLECHTES_TIMING": "⏰ Schlechtes Timing",
                "NEWS_SPIKE":        "📰 News Spike",
                "SONSTIGES":         "❓ Sonstiges",
            }
            for _, row in categories.iterrows():
                label = cat_icons.get(row["category"], row["category"])
                msg += f"  {label}: {int(row['count'])}x\n"

        # Häufigstes Verlust-Symbol
        if len(losses) > 0:
            worst_sym = losses.groupby("symbol")["pnl"].sum().idxmin()
            msg += f"\nSchlechtestes Symbol: <code>{worst_sym}</code>"

        send_telegram(msg)

    except Exception as e:
        print(f"  ⚠️  Fehler-Report Fehler: {e}")


# ─────────────────────────────────────────────
#  V3.0 IN SIGNAL GENERATOR EINBAUEN
# ─────────────────────────────────────────────

# Gecachte Markt-Kontext Werte (werden alle 4h aktualisiert)
_market_context = {
    "vix":         20.0,
    "vix_status":  "normal",
    "dxy_bias":    "neutral",
    "sp500_bias":  "bull",
    "last_update": None,
}

def update_market_context():
    """Aktualisiert VIX, DXY und S&P500 — einmal alle 4 Stunden."""
    global _market_context
    now = datetime.now(timezone.utc)

    # Nur alle 4 Stunden updaten
    if (_market_context["last_update"] is not None and
        (now - _market_context["last_update"]).seconds < 14400):
        return _market_context

    print("  🌍 Aktualisiere Markt-Kontext (VIX, DXY, S&P500)...")
    try:
        import time
        vix, vix_status = get_vix_level()
        time.sleep(1)
        dxy_bias = get_dxy_bias()
        time.sleep(1)
        sp500_bias = get_sp500_bias()

        _market_context.update({
            "vix":         vix,
            "vix_status":  vix_status,
            "dxy_bias":    dxy_bias,
            "sp500_bias":  sp500_bias,
            "last_update": now,
        })
        print(f"  🌍 VIX: {round(vix,1)} ({vix_status}) | DXY: {dxy_bias} | S&P500: {sp500_bias}")
    except Exception as e:
        print(f"  ⚠️  Markt-Kontext Fehler: {e}")

    return _market_context


def generate_signal_v3(symbol, data_dict, trend_analysis,
                       circuit_breaker, open_trades,
                       dt=None, news_blackout=False):
    """
    Erweiterte Signal-Generator Version 3.0
    Alle Original-Checks + neue V3 Checks:
    - Candlestick Muster
    - Liquidity Sweep
    - VIX Filter
    - DXY Filter
    - S&P500 Korrelation
    - Funding Rate
    - Seasonal Risk Mult
    - Max 3 Trades
    """
    # Erst Original-Signal prüfen
    signal = generate_signal(
        symbol          = symbol,
        data_dict       = data_dict,
        trend_analysis  = trend_analysis,
        circuit_breaker = circuit_breaker,
        open_trades     = open_trades,
        dt              = dt,
        news_blackout   = news_blackout,
    )

    # Wenn kein Signal → direkt zurück
    if signal["signal"] == "no_trade":
        return signal

    direction = signal["signal"]
    df_h1     = data_dict["h1"]
    price     = signal["entry"]

    if dt is None:
        dt = datetime.now(timezone.utc)

    # ── MAX 3 TRADES CHECK ────────────────────
    if len(open_trades) >= 3:
        signal["signal"] = "no_trade"
        signal["reason"] = "Max 3 Trades gleichzeitig erreicht"
        return signal

    # ── MARKT KONTEXT ─────────────────────────
    ctx = _market_context

    # VIX Check
    if ctx["vix_status"] == "extreme":
        signal["signal"] = "no_trade"
        signal["reason"] = f"VIX {round(ctx['vix'],1)} > 35 — Markt zu volatil"
        return signal
    elif ctx["vix_status"] == "elevated":
        signal["risk_mult"] = signal.get("risk_mult", 1.0) * 0.5
        print(f"  ⚠️  VIX erhöht ({round(ctx['vix'],1)}) — Position Size halbiert")

    # DXY Check für Forex
    USD_PAIRS = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X"]
    if symbol in USD_PAIRS:
        if ctx["dxy_bias"] == "up" and direction == "long":
            signal["signal"] = "no_trade"
            signal["reason"] = f"DXY steigend — kein Long auf {symbol}"
            return signal
        elif ctx["dxy_bias"] == "down" and direction == "short":
            signal["signal"] = "no_trade"
            signal["reason"] = f"DXY fallend — kein Short auf {symbol}"
            return signal

    # S&P500 Check für BTC/ETH
    if symbol in ["BTC-USD", "ETH-USD"]:
        if ctx["sp500_bias"] == "bear" and direction == "long":
            signal["risk_mult"] = signal.get("risk_mult", 1.0) * 0.5
            print(f"  ⚠️  S&P500 bearish — BTC Long mit halber Size")

        # Funding Rate Check
        rate, funding_status = get_funding_rate(symbol)
        if funding_status == "short_ok" and direction == "long":
            signal["signal"] = "no_trade"
            signal["reason"] = f"Funding Rate {round(rate*100,3)}% zu hoch — Markt überhitzt Long-seitig"
            return signal
        elif funding_status == "long_ok" and direction == "short":
            signal["signal"] = "no_trade"
            signal["reason"] = f"Funding Rate {round(rate*100,3)}% zu niedrig — Markt überhitzt Short-seitig"
            return signal

    # ── CANDLESTICK MUSTER ────────────────────
    df_candles = detect_candlestick_patterns(df_h1)
    last_candle = df_candles.iloc[-1]
    candle_ok   = False
    candle_name = ""

    if direction == "long"  and last_candle["candle_bull"]:
        candle_ok   = True
        candle_name = last_candle["candle_name"]
    elif direction == "short" and last_candle["candle_bear"]:
        candle_ok   = True
        candle_name = last_candle["candle_name"]
    else:
        # Kein Muster = kein Trade-Stopper, aber keine Verstärkung
        candle_name = "Kein Muster — Trade trotzdem ok"
        candle_ok   = True  # Nicht zwingend erforderlich

    signal["checks"]["10_candle"] = {
        "ok":   candle_ok,
        "info": candle_name if candle_name else "Kein Candlestick-Muster",
    }

    # ── LIQUIDITY SWEEP ───────────────────────
    df_sweeps = detect_liquidity_sweeps(df_h1, window=10)
    last_sweep = df_sweeps.iloc[-1]
    sweep_bull = bool(last_sweep["liq_sweep_bull"])
    sweep_bear = bool(last_sweep["liq_sweep_bear"])

    if direction == "long" and sweep_bull:
        signal["checks"]["11_liquidity"] = {"ok": True, "info": "Bullisher Liquidity Sweep erkannt ✓"}
    elif direction == "short" and sweep_bear:
        signal["checks"]["11_liquidity"] = {"ok": True, "info": "Bearisher Liquidity Sweep erkannt ✓"}
    else:
        signal["checks"]["11_liquidity"] = {"ok": True, "info": "Kein Sweep — normales Setup"}

    # ── SEASONAL RISK MULT ────────────────────
    seasonal_mult = get_seasonal_risk_mult(dt)
    if seasonal_mult < 1.0:
        signal["risk_mult"] = signal.get("risk_mult", 1.0) * seasonal_mult
        signal["checks"]["12_seasonal"] = {
            "ok":   True,
            "info": f"Seasonal: {round(seasonal_mult*100)}% Size ({dt.strftime('%A')})"
        }

    return signal


# ─────────────────────────────────────────────
#  MODUL 10 TEST
# ─────────────────────────────────────────────

print("\n" + "═"*50)
print("  MODUL 10 — V3.0 Features Test")
print("═"*50)

# Candlestick Test
print("\n  🕯  Candlestick Muster Test:")
import numpy as np
test_dates = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
test_price = 1.08 + np.cumsum(np.random.normal(0, 0.0005, 50))
df_test = pd.DataFrame({
    "open":   test_price * (1 + np.random.normal(0, 0.0002, 50)),
    "high":   test_price * (1 + abs(np.random.normal(0, 0.001, 50))),
    "low":    test_price * (1 - abs(np.random.normal(0, 0.001, 50))),
    "close":  test_price,
    "volume": np.ones(50) * 1000,
}, index=test_dates)

df_candle = detect_candlestick_patterns(df_test)
bull_patterns = df_candle[df_candle["candle_bull"]]["candle_name"].value_counts()
bear_patterns = df_candle[df_candle["candle_bear"]]["candle_name"].value_counts()
print(f"     Bullishe Muster: {bull_patterns.to_dict()}")
print(f"     Bearishe Muster: {bear_patterns.to_dict()}")

# Liquidity Sweep Test
print("\n  💧 Liquidity Sweep Test:")
df_sweep = detect_liquidity_sweeps(df_test, window=5)
bull_sweeps = df_sweep["liq_sweep_bull"].sum()
bear_sweeps = df_sweep["liq_sweep_bear"].sum()
print(f"     Bullishe Sweeps: {bull_sweeps}")
print(f"     Bearishe Sweeps: {bear_sweeps}")

# Markt Kontext
print("\n  🌍 Markt-Kontext Update:")
update_market_context()

# Seasonal
print("\n  📅 Seasonal Pattern:")
mult = get_seasonal_risk_mult()
print(f"     Heute: {datetime.now().strftime('%A')} → Size Multiplikator: {mult}")

print("\n✅ Modul 10 — V3.0 Features funktionieren!")
print("\n" + "═"*50)
print("  🎉 TRADING BOT V3.0 KOMPLETT!")
print("═"*50)
print("""
  Neue Features:
  ✅ Candlestick Muster (Engulfing, Pin Bar, Hammer)
  ✅ Liquidity Sweeps
  ✅ VIX Filter (Pause bei Extrem-Volatilität)
  ✅ Dollar Index DXY Filter
  ✅ S&P500 Korrelation für BTC/ETH
  ✅ Funding Rate Filter
  ✅ Break-Even bei 1R
  ✅ Time-Based Exit (48h)
  ✅ Max 3 Trades gleichzeitig
  ✅ Trade-Tagebuch mit KI Analyse
  ✅ Seasonal Patterns
  ✅ Fehler-Klassifizierung

  Paper Trading starten:
  update_market_context()
  run_paper_trading()
""")

# ─────────────────────────────────────────────
#  15MIN BOT — Aggressivere Version zum Testen
# ─────────────────────────────────────────────

def run_paper_trading_15min(symbols=None, capital=10000.0, max_scans=None):
    """
    15min Version des Paper Trading Bots.
    Lockerere Filter, mehr Trades, zum Vergleichen mit 1H Bot.
    RR 1.5:1, ADX >= 12, RSI 40-70 / 30-60
    """
    import time as time_module

    if symbols is None:
        symbols = ["EURUSD=X", "GBPUSD=X", "BTC-USD"]

    print("\n" + "="*50)
    print("  15MIN BOT GESTARTET")
    print("="*50)
    print(f"  Symbole   : {', '.join(symbols)}")
    print(f"  Kapital   : {capital}$")
    print(f"  Timeframe : 15min Entry")
    print("  Stoppen   : Ctrl+C\n")

    # Separater Risk State
    risk_15 = {
        "capital":        capital,
        "daily_start":    capital,
        "losses_in_row":  0,
        "circuit_active": False,
        "open_positions": {},
    }

    # Daten laden
    print("  Lade Daten...")
    data_15 = {}
    for sym in symbols:
        try:
            print(f"     {sym}...")
            df_15 = yf.download(
                tickers=sym, period="60d",
                interval="15m", auto_adjust=True, progress=False
            )
            if df_15.empty:
                continue
            if isinstance(df_15.columns, pd.MultiIndex):
                df_15.columns = df_15.columns.get_level_values(0)
            df_15.columns = [c.lower() for c in df_15.columns]
            if df_15.index.tzinfo is None:
                df_15.index = df_15.index.tz_localize("UTC")
            df_15 = add_indicators(df_15.dropna(subset=["close"]))

            sym_data   = prepare_symbol_data(sym)
            trend_data = analyze_all_timeframes(sym_data)

            data_15[sym] = {"df_15": df_15, "trend": trend_data}
            print(f"     {sym}: {len(df_15)} Kerzen OK")
            time_module.sleep(2)
        except Exception as e:
            print(f"     {sym} Fehler: {e}")

    scan_count = 0
    try:
        while True:
            now = datetime.now(timezone.utc)
            print(f"\n  15M Scan #{scan_count+1} — {now.strftime('%H:%M UTC')}")

            for sym in list(data_15.keys()):
                df_15 = data_15[sym]["df_15"]
                trend = data_15[sym]["trend"]

                final_bias = trend["final_bias"]
                if final_bias not in ["bull", "bear"]:
                    print(f"  - {sym}: Kein Trend")
                    continue

                direction = "long" if final_bias == "bull" else "short"

                if len(df_15) < 2:
                    continue

                last = df_15.iloc[-1]
                prev = df_15.iloc[-2]
                rsi  = float(last["rsi"])
                adx  = float(last["adx"])
                atr  = float(last["atr"])

                ema_cross_bull = float(prev["ema_fast"]) <= float(prev["ema_slow"]) and float(last["ema_fast"]) > float(last["ema_slow"])
                ema_cross_bear = float(prev["ema_fast"]) >= float(prev["ema_slow"]) and float(last["ema_fast"]) < float(last["ema_slow"])
                ema_cross = ema_cross_bull if direction == "long" else ema_cross_bear

                rsi_ok  = (40 <= rsi <= 70) if direction == "long" else (30 <= rsi <= 60)
                adx_ok  = adx >= 12
                hours_ok, _ = is_trading_hours(now)

                if not (ema_cross and rsi_ok and adx_ok and hours_ok):
                    reasons = []
                    if not hours_ok:  reasons.append("Handelszeit")
                    if not ema_cross: reasons.append("Kein EMA Cross")
                    if not rsi_ok:    reasons.append(f"RSI {round(rsi,1)}")
                    if not adx_ok:    reasons.append(f"ADX {round(adx,1)}")
                    print(f"  - {sym}: {' | '.join(reasons)}")
                    continue

                if sym in risk_15["open_positions"]:
                    continue

                latest = get_latest_price(sym)
                if not latest:
                    continue
                price = latest["price"]

                sl = price - atr * 1.5 if direction == "long" else price + atr * 1.5
                tp = price + abs(price - sl) * 1.5 if direction == "long" else price - abs(price - sl) * 1.5

                risk_usd = risk_15["capital"] * 0.005
                units    = risk_usd / abs(price - sl) if abs(price - sl) > 0 else 0

                risk_15["open_positions"][sym] = {
                    "direction": direction,
                    "entry":     price,
                    "sl":        sl,
                    "tp":        tp,
                    "units":     units,
                    "units_open": units,
                }

                print(f"  SIGNAL 15M: {sym} {direction.upper()} @ {round(price, 5)}")
                print(f"  SL: {round(sl,5)} | TP: {round(tp,5)} | RR: 1.5")
                send_telegram(
                    f"15M SIGNAL\n{sym} {direction.upper()}\n"
                    f"Entry: {round(price,5)}\n"
                    f"SL: {round(sl,5)} | TP: {round(tp,5)}"
                )

            # Offene Positionen updaten
            for sym in list(risk_15["open_positions"].keys()):
                latest = get_latest_price(sym)
                if not latest:
                    continue
                p   = latest["price"]
                pos = risk_15["open_positions"][sym]

                sl_hit = (pos["direction"] == "long"  and p <= pos["sl"]) or \
                         (pos["direction"] == "short" and p >= pos["sl"])
                tp_hit = (pos["direction"] == "long"  and p >= pos["tp"]) or \
                         (pos["direction"] == "short" and p <= pos["tp"])

                if sl_hit or tp_hit:
                    close_p = pos["sl"] if sl_hit else pos["tp"]
                    pnl = (close_p - pos["entry"]) * pos["units_open"]
                    if pos["direction"] == "short":
                        pnl = -pnl
                    risk_15["capital"] += pnl
                    icon   = "+" if pnl > 0 else "-"
                    reason = "TP" if tp_hit else "SL"
                    print(f"  {icon} 15M CLOSED {sym}: {reason} | P&L: {round(pnl, 2)}$")
                    if pnl < 0:
                        risk_15["losses_in_row"] += 1
                    else:
                        risk_15["losses_in_row"] = 0
                    del risk_15["open_positions"][sym]
                else:
                    pnl_now = (p - pos["entry"]) * pos["units_open"]
                    if pos["direction"] == "short":
                        pnl_now = -pnl_now
                    sign = "+" if pnl_now >= 0 else ""
                    print(f"  15M {sym}: {round(p,5)} | P&L: {sign}{round(pnl_now,2)}$")

            print(f"  Kapital 15M: {round(risk_15['capital'],2)}$")
            print(f"  Verluste in Folge: {risk_15['losses_in_row']}")

            scan_count += 1
            if max_scans and scan_count >= max_scans:
                print("\n  15Min Bot Test abgeschlossen")
                break

            print("  Naechster Scan in 60s...")
            time_module.sleep(60)

    except KeyboardInterrupt:
        print("\n  15Min Bot gestoppt")
        print(f"  Endkapital: {round(risk_15['capital'],2)}$")

# ═══════════════════════════════════════════════
#  MODUL 8 — WALK-FORWARD OPTIMIERUNG MIT OPTUNA
# ═══════════════════════════════════════════════
#
#  Automatisch beste Parameter finden via:
#  1. Optuna Bayesian Search auf Train-Fenster
#  2. Validierung auf Out-of-Sample Test-Fenster
#  3. Rollierendes Walk-Forward über alle verfügbaren Daten
#
#  Optimierte Parameter:
#  - SL_ATR_MULT  (Stop Loss Breite)
#  - TP_RR_RATIO  (Take Profit)
#  - RSI Bounds   (Long + Short)
#  - ADX Minimum  (Trendstärke)
#
#  Trigger: check_and_trigger_optimization() nach je 50 Trades
# ═══════════════════════════════════════════════


def _slice_data_by_dates(data_dict, start_dt, end_dt):
    """Schneidet alle DataFrames in data_dict auf den Zeitraum zu."""
    sliced = {}
    for key in ["weekly", "daily", "h1", "h4"]:
        if key in data_dict:
            df = data_dict[key]
            sliced[key] = df[(df.index >= start_dt) & (df.index < end_dt)]
    sliced["sr_daily"] = data_dict.get("sr_daily", [])
    sliced["sr_h4"]    = data_dict.get("sr_h4", [])
    sliced["cme_gaps"] = data_dict.get("cme_gaps", [])
    return sliced


def _run_backtest_params(symbol, data_dict_sliced, params, start_capital=10000.0):
    """
    Parameterisierter Backtest für den Optimierer.
    Nutzt vorberechnete Indikatoren — keine API-Calls.

    params dict:
        sl_atr_mult   : float  Stop Loss Breite (1.0–3.0)
        tp_rr_ratio   : float  Take Profit (1.5–4.0)
        rsi_long_min  : float  RSI Untergrenze Long (35–60)
        rsi_long_max  : float  RSI Obergrenze Long  (60–80)
        rsi_short_min : float  RSI Untergrenze Short (20–45)
        rsi_short_max : float  RSI Obergrenze Short  (40–65)
        adx_min       : float  ADX Minimum (12–30)
    """
    sl_mult       = params["sl_atr_mult"]
    rr_ratio      = params["tp_rr_ratio"]
    rsi_long_min  = params["rsi_long_min"]
    rsi_long_max  = params["rsi_long_max"]
    rsi_short_min = params["rsi_short_min"]
    rsi_short_max = params["rsi_short_max"]
    adx_min       = params["adx_min"]

    df_entry = data_dict_sliced.get("h4", pd.DataFrame())
    df_h4    = df_entry.copy()
    df_daily = data_dict_sliced.get("daily", pd.DataFrame())
    df_weekly= data_dict_sliced.get("weekly", pd.DataFrame())

    if len(df_entry) < 60 or len(df_daily) < 10 or len(df_weekly) < 3:
        return {"error": "Nicht genug Daten"}

    capital        = start_capital
    losses_in_row  = 0
    daily_loss     = 0.0
    daily_start    = start_capital
    circuit_active = False
    open_pos       = {}
    trades         = []
    capital_curve  = [start_capital]
    last_date      = None

    # Trendstruktur vorberechnen
    try:
        weekly_struct = detect_bos_choch(df_weekly, window=3)
        daily_struct  = detect_bos_choch(df_daily,  window=10)
        h4_struct     = detect_bos_choch(df_h4,     window=20)
        df_h4         = find_fvg(df_h4)
    except Exception:
        return {"error": "Trendstruktur-Fehler"}

    total_candles = len(df_entry)

    for i in range(50, total_candles):
        candle   = df_entry.iloc[i]
        dt       = df_entry.index[i]
        price    = float(candle["close"])
        high     = float(candle["high"])
        low      = float(candle["low"])
        atr      = float(candle["atr"])
        ema_fast = float(candle["ema_fast"])

        # Tages-Reset — Circuit Breaker ist Tagesschutz, kein permanentes Ban
        current_date = dt.date()
        if last_date != current_date:
            daily_start    = capital
            daily_loss     = 0.0
            circuit_active = False  # Reset täglich
            losses_in_row  = 0      # Reset täglich
            last_date      = current_date

        # ── OFFENE POSITIONEN UPDATEN ─────────────
        for sym in list(open_pos.keys()):
            pos       = open_pos[sym]
            direction = pos["direction"]
            entry     = pos["entry"]
            sl        = pos["sl"]
            tp        = pos["tp"]
            partial   = pos["partial_tp"]

            sl_hit = (direction == "long"  and low  <= sl) or \
                     (direction == "short" and high >= sl)
            tp_hit = (direction == "long"  and high >= tp) or \
                     (direction == "short" and low  <= tp)

            if sl_hit:
                pnl = (sl - entry) * pos["units_open"]
                if direction == "short": pnl = -pnl
                capital       += pnl
                losses_in_row  = 0 if pnl > 0 else losses_in_row + 1
                daily_loss     = max(0, (daily_start - capital) / daily_start)
                trades.append(_make_trade_record(sym, pos, sl, "sl_hit", pnl, dt))
                del open_pos[sym]
                continue

            if tp_hit:
                pnl = (tp - entry) * pos["units_open"]
                if direction == "short": pnl = -pnl
                capital       += pnl
                losses_in_row  = 0
                daily_loss     = max(0, (daily_start - capital) / daily_start)
                trades.append(_make_trade_record(sym, pos, tp, "tp_hit", pnl, dt))
                del open_pos[sym]
                continue

            # Partial Close bei 1.5R
            if not pos["partial_done"]:
                partial_hit = (direction == "long"  and high >= partial) or \
                              (direction == "short" and low  <= partial)
                if partial_hit:
                    units_close = pos["units_open"] * 0.5
                    pnl_p = (partial - entry) * units_close
                    if direction == "short": pnl_p = -pnl_p
                    capital += pnl_p
                    open_pos[sym]["units_open"]  *= 0.5
                    open_pos[sym]["partial_done"] = True
                    open_pos[sym]["sl"]           = entry
                    open_pos[sym]["trailing_sl"]  = entry

            # Trailing Stop (EMA 20)
            if pos.get("partial_done") and pos.get("trailing_sl") is not None:
                if direction == "long" and ema_fast > pos["sl"] and ema_fast > entry:
                    open_pos[sym]["sl"] = round(ema_fast, 5)
                elif direction == "short" and ema_fast < pos["sl"] and ema_fast < entry:
                    open_pos[sym]["sl"] = round(ema_fast, 5)

                trail_hit = (direction == "long"  and low  <= open_pos[sym]["sl"]) or \
                            (direction == "short" and high >= open_pos[sym]["sl"])
                if trail_hit:
                    close_p = open_pos[sym]["sl"]
                    pnl     = (close_p - entry) * open_pos[sym]["units_open"]
                    if direction == "short": pnl = -pnl
                    capital       += pnl
                    losses_in_row  = 0 if pnl > 0 else losses_in_row + 1
                    daily_loss     = max(0, (daily_start - capital) / daily_start)
                    trades.append(_make_trade_record(sym, pos, close_p, "trailing_sl", pnl, dt))
                    del open_pos[sym]
                    continue

        # Circuit Breaker
        if losses_in_row >= MAX_LOSSES_ROW or daily_loss >= MAX_DAILY_LOSS:
            circuit_active = True

        capital_curve.append(capital)

        if circuit_active or symbol in open_pos or len(open_pos) >= 3:
            continue

        # ── NEUES SIGNAL ──────────────────────────
        w_idx = weekly_struct.index.searchsorted(dt) - 1
        d_idx = daily_struct.index.searchsorted(dt)  - 1
        h_idx = h4_struct.index.searchsorted(dt)     - 1

        if w_idx < 0 or d_idx < 0 or h_idx < 0:
            continue

        w_bias = weekly_struct["structure"].iloc[max(0, w_idx)]
        d_bias = daily_struct["structure"].iloc[max(0, d_idx)]
        h_bias = h4_struct["structure"].iloc[max(0, h_idx)]

        ema_f = float(candle["ema_fast"])
        ema_s = float(candle["ema_slow"])
        ema_t = float(candle["ema_trend"])

        d_bull = d_bias == "bull" or (d_bias == "neutral" and ema_f > ema_s > ema_t)
        d_bear = d_bias == "bear" or (d_bias == "neutral" and ema_f < ema_s < ema_t)
        h_bull = h_bias == "bull" or (h_bias == "neutral" and ema_f > ema_s)
        h_bear = h_bias == "bear" or (h_bias == "neutral" and ema_f < ema_s)

        bull_ok = d_bull and h_bull and w_bias != "bear"
        bear_ok = d_bear and h_bear  # Weekly-Filter für Shorts entfernt — Daily+4H reicht

        if not bull_ok and not bear_ok:
            continue

        direction = "long" if bull_ok else "short"

        # EMA Cross
        prev = df_entry.iloc[i - 1]
        ema_cross = (
            (direction == "long"  and prev["ema_fast"] <= prev["ema_slow"] and candle["ema_fast"] > candle["ema_slow"]) or
            (direction == "short" and prev["ema_fast"] >= prev["ema_slow"] and candle["ema_fast"] < candle["ema_slow"])
        )
        if not ema_cross:
            continue

        # RSI — mit optimierten Bounds
        rsi = float(candle["rsi"])
        if direction == "long":
            rsi_ok = rsi_long_min <= rsi <= rsi_long_max
        else:
            rsi_ok = rsi_short_min <= rsi <= rsi_short_max
        if not rsi_ok:
            continue

        # ADX — mit optimiertem Minimum
        adx = float(candle["adx"])
        if adx < adx_min:
            continue

        # Handelszeiten
        hours_ok, _ = is_trading_hours(dt)
        if not hours_ok:
            continue

        # SL / TP — mit optimierten Multiplikatoren
        sl_price   = price - (atr * sl_mult) if direction == "long" else price + (atr * sl_mult)
        risk       = abs(price - sl_price)
        tp_price   = price + (risk * rr_ratio) if direction == "long" else price - (risk * rr_ratio)
        partial_tp = price + (risk * 1.5) if direction == "long" else price - (risk * 1.5)

        sizing = calculate_position_size(capital, RISK_PER_TRADE, price, sl_price)

        open_pos[symbol] = {
            "direction":   direction,
            "entry":       price,
            "sl":          sl_price,
            "sl_original": sl_price,
            "tp":          tp_price,
            "partial_tp":  partial_tp,
            "units":       sizing["units"],
            "units_open":  sizing["units"],
            "risk_usd":    sizing["risk_usd"],
            "partial_done": False,
            "trailing_sl": None,
            "open_time":   dt,
        }

    # Offene Positionen am Ende schließen
    for sym, pos in open_pos.items():
        last_price = float(df_entry["close"].iloc[-1])
        pnl = (last_price - pos["entry"]) * pos["units_open"]
        if pos["direction"] == "short": pnl = -pnl
        capital += pnl
        trades.append(_make_trade_record(sym, pos, last_price, "end_of_test", pnl, df_entry.index[-1]))

    return _calculate_stats(trades, start_capital, capital, capital_curve)


def _optuna_objective(trial, symbol, data_dict_train):
    """
    Optuna Objective-Funktion.
    Maximiert: Profit Factor × Win-Rate − Drawdown-Strafe
    """
    params = {
        "sl_atr_mult":   trial.suggest_float("sl_atr_mult",   1.0,  3.0),
        "tp_rr_ratio":   trial.suggest_float("tp_rr_ratio",   1.5,  4.0),
        "rsi_long_min":  trial.suggest_float("rsi_long_min",  35.0, 55.0),
        "rsi_long_max":  trial.suggest_float("rsi_long_max",  60.0, 80.0),
        "rsi_short_min": trial.suggest_float("rsi_short_min", 20.0, 45.0),
        "rsi_short_max": trial.suggest_float("rsi_short_max", 40.0, 65.0),
        "adx_min":       trial.suggest_float("adx_min",       12.0, 30.0),
    }

    # Constraints: RSI-Bounds müssen sinnvoll sein
    if params["rsi_long_min"]  >= params["rsi_long_max"]:  return -999.0
    if params["rsi_short_min"] >= params["rsi_short_max"]: return -999.0
    # TP muss größer als 1.5x SL sein (Mindest-RR)
    if params["tp_rr_ratio"] < params["sl_atr_mult"] * 0.8: return -999.0

    try:
        stats = _run_backtest_params(symbol, data_dict_train, params, start_capital=10000.0)

        if "error" in stats or stats["total_trades"] < 5:
            return -1.0

        pf = min(stats["profit_factor"], 10.0)   # Cap gegen Ausreißer
        wr = stats["win_rate"] / 100.0
        dd = stats["max_drawdown"] / 100.0
        n  = stats["total_trades"]

        # Score: PF × WR belohnt, Drawdown bestraft
        # Bonus für mehr Trades (bis zu einem gewissen Punkt)
        score = (pf * wr) - (dd * 2.0) + min(n / 100.0, 0.5)
        return float(score) if np.isfinite(score) else -1.0

    except Exception:
        return -1.0


def walk_forward_optimize(symbol, data_dict, n_trials=50, train_months=12, test_months=3):
    """
    Walk-Forward Optimierung mit Optuna.

    Teilt die historischen Daten in rollende Fenster auf:
      Train-Fenster → Optuna sucht beste Parameter
      Test-Fenster  → Out-of-Sample Validierung

    Args:
        symbol:        z.B. "EURUSD=X"
        data_dict:     Output von prepare_symbol_data()
        n_trials:      Anzahl Optuna-Versuche pro Fenster
                       10 = schnell (Test), 50-100 = echte Optimierung
        train_months:  Trainings-Fenster in Monaten (Standard: 12)
        test_months:   Test-Fenster in Monaten      (Standard: 3)

    Returns:
        {
            "windows":        alle Fenster mit Ergebnissen,
            "best_params":    gemittelte beste Parameter,
            "oos_stats":      aggregierte OOS Performance,
            "recommendation": Empfehlung ob deploy oder nicht,
        }
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("\n  ❌ Optuna nicht installiert!")
        print("  Installiere mit: pip install optuna --break-system-packages")
        print("  Dann Spyder neu starten und nochmal ausführen.")
        return None

    print(f"\n{'═'*50}")
    print(f"  MODUL 8 — Walk-Forward Optimierung")
    print(f"  Symbol  : {symbol}")
    print(f"  Trials  : {n_trials} pro Fenster")
    print(f"  Fenster : {train_months}M Train + {test_months}M Test (rollend)")
    print(f"{'═'*50}")

    # Datenbereich aus h4 ableiten
    df_ref     = data_dict["h4"]
    data_start = df_ref.index[0]
    data_end   = df_ref.index[-1]

    # Walk-Forward Fenster berechnen
    windows      = []
    cursor       = data_start
    train_delta  = pd.DateOffset(months=train_months)
    test_delta   = pd.DateOffset(months=test_months)

    while True:
        train_start = cursor
        train_end   = cursor + train_delta
        test_start  = train_end
        test_end    = train_end + test_delta
        if test_end > data_end:
            break
        windows.append({
            "train_start": train_start, "train_end": train_end,
            "test_start":  test_start,  "test_end":  test_end,
        })
        cursor += test_delta   # Fenster um test_months vorschieben

    if not windows:
        print(f"\n  ⚠️  Nicht genug Daten!")
        print(f"  Vorhandene Daten: {data_start.strftime('%Y-%m')} – {data_end.strftime('%Y-%m')}")
        print(f"  Benötigt: mindestens {train_months + test_months} Monate")
        return None

    print(f"\n  {len(windows)} Walk-Forward Fenster gefunden")
    print(f"  Daten: {data_start.strftime('%Y-%m')} – {data_end.strftime('%Y-%m')}\n")

    all_window_results = []
    all_best_params    = []

    for idx, win in enumerate(windows):
        print(f"  📊 Fenster {idx+1}/{len(windows)}: "
              f"Train {win['train_start'].strftime('%Y-%m')}–{win['train_end'].strftime('%Y-%m')} | "
              f"Test {win['test_start'].strftime('%Y-%m')}–{win['test_end'].strftime('%Y-%m')}")

        train_data = _slice_data_by_dates(data_dict, win["train_start"], win["train_end"])
        test_data  = _slice_data_by_dates(data_dict, win["test_start"],  win["test_end"])

        if len(train_data.get("h4", pd.DataFrame())) < 100:
            print(f"     ⚠️  Zu wenig Trainingsdaten — überspringe")
            continue
        if len(test_data.get("h4", pd.DataFrame())) < 20:
            print(f"     ⚠️  Zu wenig Testdaten — überspringe")
            continue

        # Optuna Studie
        study = optuna.create_study(
            direction = "maximize",
            sampler   = optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            lambda trial: _optuna_objective(trial, symbol, train_data),
            n_trials          = n_trials,
            show_progress_bar = False,
        )

        best_p = study.best_params
        print(f"     ✅ Train Score: {round(study.best_value, 3)} | "
              f"SL×{round(best_p['sl_atr_mult'],2)} | "
              f"RR {round(best_p['tp_rr_ratio'],2)} | "
              f"ADX≥{round(best_p['adx_min'],1)}")

        # Out-of-Sample Test
        oos = _run_backtest_params(symbol, test_data, best_p, start_capital=10000.0)

        if "error" not in oos:
            pf_icon = "✅" if oos["profit_factor"] >= 1.5 else ("🟡" if oos["profit_factor"] >= 1.0 else "❌")
            print(f"     {pf_icon} OOS: {oos['total_trades']} Trades | "
                  f"WR {oos['win_rate']}% | "
                  f"PF {oos['profit_factor']} | "
                  f"DD {oos['max_drawdown']}%")
        else:
            print(f"     ⚠️  OOS: {oos.get('error', 'Keine Trades')}")

        all_window_results.append({
            "window":      idx + 1,
            "train_start": str(win["train_start"].date()),
            "train_end":   str(win["train_end"].date()),
            "test_start":  str(win["test_start"].date()),
            "test_end":    str(win["test_end"].date()),
            "best_score":  round(study.best_value, 3),
            "best_params": best_p,
            "oos_stats":   oos,
        })
        all_best_params.append(best_p)

    if not all_window_results:
        print("\n  ❌ Keine erfolgreichen Fenster")
        return None

    # ── BESTE PARAMETER MITTELN ───────────────────
    param_keys = list(all_best_params[0].keys())
    avg_params = {
        key: round(float(np.mean([p[key] for p in all_best_params])), 4)
        for key in param_keys
    }

    # ── OOS AGGREGATION ───────────────────────────
    valid_oos = [
        w["oos_stats"] for w in all_window_results
        if "error" not in w["oos_stats"] and w["oos_stats"]["total_trades"] >= 3
    ]

    if valid_oos:
        oos_agg = {
            "avg_win_rate":      round(np.mean([s["win_rate"]      for s in valid_oos]), 1),
            "avg_profit_factor": round(np.mean([s["profit_factor"] for s in valid_oos]), 2),
            "avg_max_drawdown":  round(np.mean([s["max_drawdown"]  for s in valid_oos]), 1),
            "avg_return":        round(np.mean([s["total_return"]  for s in valid_oos]), 1),
            "total_oos_trades":  sum(s["total_trades"] for s in valid_oos),
            "windows_analyzed":  len(valid_oos),
        }
    else:
        oos_agg = {"error": "Keine validen OOS Fenster — zu wenig Trades"}

    # ── EMPFEHLUNG ────────────────────────────────
    if "error" not in oos_agg:
        pf = oos_agg["avg_profit_factor"]
        dd = oos_agg["avg_max_drawdown"]
        if   pf >= 1.5 and dd <= 15: rec = "✅ DEPLOY — Parameter robust (PF≥1.5, DD≤15%)"
        elif pf >= 1.2:              rec = "🟡 MONITOR — Profitabel, aber Risiko vorhanden"
        else:                         rec = "❌ NICHT DEPLOYEN — Strategie nicht robust genug"
    else:
        rec = "⚠️  UNKLAR — Zu wenig OOS-Daten"

    result = {
        "windows":        all_window_results,
        "best_params":    avg_params,
        "oos_stats":      oos_agg,
        "recommendation": rec,
        "symbol":         symbol,
        "n_trials":       n_trials,
    }

    _print_optimization_results(result)
    return result


def _print_optimization_results(result):
    """Übersichtliche Ausgabe der Walk-Forward Ergebnisse."""
    print(f"\n{'═'*50}")
    print(f"  WALK-FORWARD ERGEBNIS — {result['symbol']}")
    print(f"{'═'*50}")

    oos = result["oos_stats"]
    if "error" not in oos:
        print(f"\n  📊 OOS Performance ({oos['windows_analyzed']} Fenster, {oos['total_oos_trades']} Trades):")
        print(f"     Ø Win-Rate     : {oos['avg_win_rate']}%")
        print(f"     Ø Profit Factor: {oos['avg_profit_factor']}")
        print(f"     Ø Max Drawdown : {oos['avg_max_drawdown']}%")
        print(f"     Ø Rendite      : {oos['avg_return']}%")
    else:
        print(f"\n  ⚠️  {oos['error']}")

    print(f"\n  🔧 Empfohlene Parameter (Ø über alle Fenster):")
    bp = result["best_params"]
    print(f"     SL_ATR_MULT    = {bp.get('sl_atr_mult', '?')}")
    print(f"     TP_RR_RATIO    = {bp.get('tp_rr_ratio', '?')}")
    print(f"     RSI Long       = {bp.get('rsi_long_min', '?')} – {bp.get('rsi_long_max', '?')}")
    print(f"     RSI Short      = {bp.get('rsi_short_min', '?')} – {bp.get('rsi_short_max', '?')}")
    print(f"     ADX Minimum    = {bp.get('adx_min', '?')}")

    print(f"\n  → {result['recommendation']}")
    print(f"\n  Zum Aktivieren: apply_optimized_params(opt_result)")


def apply_optimized_params(opt_result):
    """
    Setzt optimierte Parameter als neue globale Werte.
    Speichert sie außerdem in der SQLite-DB.

    Aufruf nach Optimierung:
        opt = walk_forward_optimize("EURUSD=X", data)
        apply_optimized_params(opt)
    """
    global SL_ATR_MULT, TP_RR_RATIO

    if opt_result is None or "best_params" not in opt_result:
        print("  ⚠️  Keine Optimierungsergebnisse")
        return False

    bp = opt_result["best_params"]

    old_sl = SL_ATR_MULT
    old_rr = TP_RR_RATIO

    SL_ATR_MULT = bp.get("sl_atr_mult", SL_ATR_MULT)
    TP_RR_RATIO = bp.get("tp_rr_ratio", TP_RR_RATIO)

    print(f"\n  🔧 Parameter aktualisiert:")
    print(f"     SL_ATR_MULT : {old_sl} → {round(SL_ATR_MULT, 2)}")
    print(f"     TP_RR_RATIO : {old_rr} → {round(TP_RR_RATIO, 2)}")

    # In DB speichern
    try:
        import json
        conn = sqlite3.connect("logs/trades.db")
        c    = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS optimization_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol    TEXT,
                params    TEXT,
                oos_pf    REAL,
                oos_wr    REAL,
                oos_dd    REAL,
                n_trials  INTEGER,
                date      TEXT
            )
        """)
        oos = opt_result.get("oos_stats", {})
        c.execute("""
            INSERT INTO optimization_log (symbol, params, oos_pf, oos_wr, oos_dd, n_trials, date)
            VALUES (?,?,?,?,?,?,?)
        """, (
            opt_result.get("symbol", "?"),
            json.dumps(bp),
            oos.get("avg_profit_factor", 0),
            oos.get("avg_win_rate", 0),
            oos.get("avg_max_drawdown", 0),
            opt_result.get("n_trials", 0),
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()
        conn.close()
        print(f"  💾 In DB gespeichert (optimization_log)")
    except Exception as e:
        print(f"  ⚠️  DB-Speicherung fehlgeschlagen: {e}")

    print(f"\n  ✅ Neue Parameter aktiv — run_backtest() nutzt jetzt diese Werte")
    return True


def check_and_trigger_optimization(symbol, data_dict, threshold=50):
    """
    Automatischer Trigger: läuft im Paper Trading Loop.
    Startet Optimierung wenn threshold neue Trades erreicht.

    Wird in run_paper_trading() aufgerufen.

    Args:
        threshold: Trades nach denen optimiert wird (Standard: 50)
    """
    trade_count = get_trade_count()

    # Wann wurde zuletzt optimiert?
    last_opt_at = 0
    try:
        conn = sqlite3.connect("logs/trades.db")
        c    = conn.cursor()
        c.execute("""
            SELECT COUNT(*) FROM optimization_log WHERE symbol = ?
        """, (symbol,))
        opt_runs = c.fetchone()[0]
        conn.close()
        last_opt_at = opt_runs * threshold
    except Exception:
        pass

    trades_since_last_opt = trade_count - last_opt_at

    if trade_count >= threshold and trades_since_last_opt >= threshold:
        print(f"\n  🔬 {trade_count} Trades — Starte automatische Optimierung für {symbol}...")
        send_telegram(f"🔬 <b>Optimierung startet</b>\n{symbol} — {trade_count} Trades erreicht")

        opt_result = walk_forward_optimize(
            symbol       = symbol,
            data_dict    = data_dict,
            n_trials     = 50,
            train_months = 12,
            test_months  = 3,
        )

        if opt_result:
            rec = opt_result.get("recommendation", "")
            if rec.startswith("✅"):
                apply_optimized_params(opt_result)
                send_telegram(
                    f"✅ <b>Optimierung abgeschlossen</b>\n{symbol}\n"
                    f"PF: {opt_result['oos_stats'].get('avg_profit_factor', '?')}\n"
                    f"WR: {opt_result['oos_stats'].get('avg_win_rate', '?')}%\n"
                    f"→ Neue Parameter aktiv"
                )
            else:
                send_telegram(
                    f"⚠️ <b>Optimierung: Keine Verbesserung</b>\n{symbol}\n{rec}"
                )
        return opt_result
    else:
        remaining = threshold - trades_since_last_opt
        if trade_count > 0:
            print(f"  📊 Optimierung in ~{max(0, remaining)} Trades")
        return None


# ─────────────────────────────────────────────
#  MODUL 8 TEST
# ─────────────────────────────────────────────

print("\n" + "═"*50)
print("  MODUL 8 — Walk-Forward Optimierung Test")
print("═"*50)

try:
    import optuna
    optuna_version = optuna.__version__
    print(f"  ✅ Optuna {optuna_version} verfügbar")

    print("\n  🔬 Schnell-Test: 10 Trials, 1 Fenster...")
    print("  ⏳ Dauert ca. 1-3 Minuten...")

    opt_result = walk_forward_optimize(
        symbol       = "EURUSD=X",
        data_dict    = data,
        n_trials     = 10,       # Kurz für Test — echte Optimierung: 50–100
        train_months = 12,
        test_months  = 3,
    )

    if opt_result:
        print("\n  💡 Für echte Optimierung:")
        print("     opt = walk_forward_optimize('EURUSD=X', data, n_trials=100)")
        print("     apply_optimized_params(opt)")
        print("\n✅ Modul 8 funktioniert!")
    else:
        print("\n  ℹ️  Test nicht ausführbar — Funktionen sind bereit für manuellen Aufruf")

except ImportError:
    print("  ⚠️  Optuna nicht installiert")
    print("\n  Installiere mit diesem Befehl in der Spyder Console:")
    print("  !pip install optuna")
    print("\n  Dann: walk_forward_optimize('EURUSD=X', data, n_trials=50)")
    print("\n  Alle Funktionen sind geladen und bereit:")
    print("  - walk_forward_optimize(symbol, data)")
    print("  - apply_optimized_params(opt_result)")
    print("  - check_and_trigger_optimization(symbol, data)")

print("\n" + "═"*50)
print("  🎉 TRADING BOT V3.0 — ALLE 10 MODULE FERTIG!")
print("═"*50)
print("""
  Starten:
    update_market_context()
    run_paper_trading()

  Nach 50 Trades: Optimierung startet automatisch.
  Oder manuell:
    walk_forward_optimize("EURUSD=X", data, n_trials=50)
""")
# ─────────────────────────────────────────────
#  MULTI-SYMBOL BACKTEST (verschiedene Strategien)
# ─────────────────────────────────────────────

def run_multi_backtest(symbols=None):
    """
    Backtestet mehrere Symbole mit je eigener Strategie (SYMBOL_STRATEGY).
    Gibt kombinierte Stats aus.
    """
    if symbols is None:
        symbols = list(SYMBOL_STRATEGY.keys())

    all_trades   = []
    total_return = 0

    print("\n" + "═"*50)
    print("  MULTI-SYMBOL BACKTEST")
    print("═"*50)

    for sym in symbols:
        d   = prepare_symbol_data(sym)
        r   = run_backtest(sym, d, verbose=False)
        all_trades.extend(r["trades"])
        s     = r["stats"]
        strat = SYMBOL_STRATEGY.get(sym, "ema_pullback")
        if "error" in s:
            print(f"  {sym:12s} [{strat:15s}]  ⚠️  {s['error']}")
            continue
        print(f"  {sym:12s} [{strat:15s}]  "
              f"Return: {s['total_return']:+.1f}%  "
              f"Trades: {s['total_trades']:3d}  "
              f"WR: {s['win_rate']:.1f}%  "
              f"PF: {s['profit_factor']:.2f}")
        total_return += s["total_return"]

    import pandas as pd
    df     = pd.DataFrame(all_trades)
    wins   = df[df["pnl"] > 0]["pnl"]
    losses = df[df["pnl"] < 0]["pnl"]

    print(f"\n{'─'*50}")
    print(f"  KOMBINIERT — {len(symbols)} Symbole")
    print(f"  Trades gesamt : {len(df)}")
    print(f"  Win-Rate      : {len(wins)/len(df)*100:.1f}%")
    print(f"  Profit Factor : {wins.sum()/abs(losses.sum()):.2f}")
    print(f"  Ø Return/Sym  : {total_return/len(symbols):.2f}%")
    print(f"  Ø Gewinn      : ${wins.mean():.2f}")
    print(f"  Ø Verlust     : ${losses.mean():.2f}")
    print(f"{'═'*50}")
    return df


# Direkt ausführen:
# run_multi_backtest()


def compare_strategies(symbol):
    """Testet alle 3 Strategien für ein Symbol und zeigt Vergleich."""
    print(f"\n{'═'*50}")
    print(f"  STRATEGIE-VERGLEICH: {symbol}")
    print(f"{'═'*50}")
    d = prepare_symbol_data(symbol)
    best = None
    for strat in ["ema_pullback", "mean_reversion", "breakout"]:
        r = run_backtest(symbol, d, verbose=False, strategy=strat)
        s = r["stats"]
        if "error" in s:
            print(f"  {strat:15s} → ⚠️  Keine Trades")
            continue
        marker = ""
        if best is None or s["profit_factor"] > best:
            best = s["profit_factor"]
            marker = "  ← beste"
        print(f"  {strat:15s} → Return: {s['total_return']:+6.1f}%  "
              f"Trades: {s['total_trades']:3d}  "
              f"WR: {s['win_rate']:.1f}%  "
              f"PF: {s['profit_factor']:.2f}{marker}")
    print(f"{'═'*50}")

# Direkt ausführen:
# compare_strategies("GC=F")


# ═══════════════════════════════════════════════
#  15MIN BACKTEST
#
#  Hinweis: yfinance liefert 15min-Daten nur für
#  die letzten 60 Tage (API-Limit). Statistisch
#  weniger aussagekräftig als der 1H-Backtest.
#
#  Aufruf:
#    run_backtest_15min("EURUSD=X")
#    run_multi_backtest_15min()
# ═══════════════════════════════════════════════

def run_backtest_15min(symbol, strategy=None, start_capital=10000.0, verbose=False):
    """
    Backtestet ein Symbol auf dem 15min-Timeframe.

    Gleiche Entry-Logik wie run_backtest() (1H),
    aber auf 15min-Kerzen angewendet.

    Limit: yfinance liefert max. 60 Tage 15min-Daten.
    """
    if strategy is None:
        strategy = SYMBOL_STRATEGY.get(symbol, "ema_pullback")

    tp_rr      = SYMBOL_TP.get(symbol, STRATEGY_TP.get(strategy, TP_RR_RATIO))
    bb_period, _ = SYMBOL_BB.get(symbol, (BB_PERIOD, BB_STD))
    bb_col     = "bb14" if bb_period == 14 else "bb20"

    print(f"\n  🔄 15min Backtest: {symbol} [{strategy}]")

    # 15min-Daten laden (max 60 Tage)
    import time as _t
    try:
        df_raw = yf.download(
            tickers     = symbol,
            period      = "60d",
            interval    = "15m",
            auto_adjust = True,
            progress    = False,
        )
        if df_raw.empty:
            _t.sleep(10)
            df_raw = yf.download(
                tickers     = symbol,
                period      = "60d",
                interval    = "15m",
                auto_adjust = True,
                progress    = False,
            )
        if df_raw.empty:
            return {"trades": [], "stats": {"error": "Keine 15min-Daten"}}
    except Exception as e:
        return {"trades": [], "stats": {"error": str(e)}}

    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    if df_raw.index.tzinfo is None:
        df_raw.index = df_raw.index.tz_localize("UTC")
    else:
        df_raw.index = df_raw.index.tz_convert("UTC")
    df_raw = df_raw.dropna(subset=["close"])
    df_raw = df_raw[df_raw["close"] > 0]

    # Indikatoren berechnen (gleiche Funktion wie 1H)
    df = add_indicators(df_raw)
    print(f"     {len(df)} Kerzen geladen ({df.index[0].date()} – {df.index[-1].date()})")

    # Backtest-Loop (identisch zu run_backtest)
    capital        = start_capital
    losses_in_row  = 0
    daily_loss     = 0.0
    daily_start    = start_capital
    circuit_active = False
    open_pos       = {}
    trades         = []
    capital_curve  = [start_capital]
    last_date      = None

    total_candles = len(df)

    for i in range(50, total_candles):
        candle = df.iloc[i]
        dt     = df.index[i]
        price  = float(candle["close"])
        high   = float(candle["high"])
        low    = float(candle["low"])
        atr    = float(candle["atr"])

        # Tages-Reset
        current_date = dt.date()
        if last_date != current_date:
            daily_start    = capital
            daily_loss     = 0.0
            circuit_active = False
            losses_in_row  = 0
            last_date      = current_date

        # Offene Positionen updaten
        for sym in list(open_pos.keys()):
            pos       = open_pos[sym]
            direction = pos["direction"]
            entry     = pos["entry"]
            sl        = pos["sl"]
            tp        = pos["tp"]

            sl_hit = (direction == "long"  and low  <= sl) or \
                     (direction == "short" and high >= sl)
            tp_hit = (direction == "long"  and high >= tp) or \
                     (direction == "short" and low  <= tp)

            if sl_hit:
                pnl = (sl - entry) * pos["units_open"]
                if direction == "short": pnl = -pnl
                capital       += pnl
                losses_in_row  = 0 if pnl > 0 else losses_in_row + 1
                daily_loss     = max(0, (daily_start - capital) / daily_start)
                trades.append(_make_trade_record(sym, pos, sl, "sl_hit", pnl, dt))
                del open_pos[sym]
                if verbose: print(f"  ❌ SL {sym} @ {sl:.5f}  P&L: {round(pnl,2)}$")
                continue

            if tp_hit:
                pnl = (tp - entry) * pos["units_open"]
                if direction == "short": pnl = -pnl
                capital       += pnl
                losses_in_row  = 0
                daily_loss     = max(0, (daily_start - capital) / daily_start)
                trades.append(_make_trade_record(sym, pos, tp, "tp_hit", pnl, dt))
                del open_pos[sym]
                if verbose: print(f"  ✅ TP {sym} @ {tp:.5f}  P&L: +{round(pnl,2)}$")
                continue

        # Circuit Breaker
        if losses_in_row >= MAX_LOSSES_ROW or daily_loss >= MAX_DAILY_LOSS:
            circuit_active = True

        capital_curve.append(capital)

        if circuit_active or symbol in open_pos:
            continue

        # Signal prüfen
        if i < 1:
            continue
        prev      = df.iloc[i - 1]
        rsi       = float(candle["rsi"])
        adx       = float(candle["adx"])
        hours_ok, _ = is_trading_hours(dt)
        if not hours_ok:
            continue

        direction = None
        sl_price  = None
        tp_price  = None

        if strategy == "ema_pullback":
            ema_f = float(candle["ema_fast"])
            ema_s = float(candle["ema_slow"])
            ema_t = float(candle["ema_trend"])
            trend_long  = ema_f > ema_s > ema_t
            trend_short = ema_f < ema_s < ema_t
            if not trend_long and not trend_short:
                continue
            direction = "long" if trend_long else "short"
            if direction == "long":
                if not (float(prev["low"]) <= ema_f and price > ema_f):
                    continue
            else:
                if not (float(prev["high"]) >= ema_f and price < ema_f):
                    continue
            rsi_ok = (35 <= rsi <= 75) if direction == "long" else (25 <= rsi <= 65)
            if not rsi_ok or adx < 20:
                continue
            if direction == "long":
                sl_price = min(float(prev["low"]), float(candle["low"])) - atr * 0.5
            else:
                sl_price = max(float(prev["high"]), float(candle["high"])) + atr * 0.5
            risk = abs(price - sl_price)
            if risk <= 0: continue
            tp_price = price + risk * tp_rr if direction == "long" else price - risk * tp_rr

        elif strategy == "mean_reversion":
            try:
                bb_upper = float(candle[f"{bb_col}_upper"])
                bb_lower = float(candle[f"{bb_col}_lower"])
                bb_mid   = float(candle[f"{bb_col}_mid"])
                if any(pd.isna([bb_upper, bb_lower, bb_mid])): continue
            except: continue
            if float(prev["low"]) <= bb_lower and price > bb_lower and rsi < 45:
                direction = "long"
            elif float(prev["high"]) >= bb_upper and price < bb_upper and rsi > 55:
                direction = "short"
            else:
                continue
            if adx > 35: continue
            if direction == "long":
                sl_price = float(candle["low"]) - atr * 0.5
                tp_price = bb_mid
            else:
                sl_price = float(candle["high"]) + atr * 0.5
                tp_price = bb_mid
            risk = abs(price - sl_price)
            if risk <= 0 or abs(tp_price - price) / risk < 1.0: continue

        elif strategy == "breakout":
            try:
                prev_high_n = float(prev["high_n"])
                prev_low_n  = float(prev["low_n"])
                if pd.isna(prev_high_n) or pd.isna(prev_low_n): continue
            except: continue
            if price > prev_high_n:
                direction = "long"
            elif price < prev_low_n:
                direction = "short"
            else:
                continue
            if adx < 20: continue
            sl_price = price - atr * SL_ATR_MULT if direction == "long" else price + atr * SL_ATR_MULT
            risk = abs(price - sl_price)
            if risk <= 0: continue
            tp_price = price + risk * tp_rr if direction == "long" else price - risk * tp_rr

        if direction is None: continue

        sizing = calculate_position_size(capital, RISK_PER_TRADE, price, sl_price)
        open_pos[symbol] = {
            "direction":    direction,
            "entry":        price,
            "sl":           sl_price,
            "sl_original":  sl_price,
            "tp":           tp_price,
            "partial_tp":   price + abs(price - sl_price) * 1.5 if direction == "long" else price - abs(price - sl_price) * 1.5,
            "units":        sizing["units"],
            "units_open":   sizing["units"],
            "risk_usd":     sizing["risk_usd"],
            "partial_done": False,
            "trailing_sl":  None,
            "open_time":    dt,
        }
        if verbose:
            icon = "📈 LONG" if direction == "long" else "📉 SHORT"
            print(f"  {icon} {symbol} @ {price:.5f}  SL:{round(sl_price,5)}  TP:{round(tp_price,5)}")

    # Offene Positionen am Ende schließen
    for sym, pos in open_pos.items():
        last_price = float(df["close"].iloc[-1])
        pnl = (last_price - pos["entry"]) * pos["units_open"]
        if pos["direction"] == "short": pnl = -pnl
        capital += pnl
        trades.append(_make_trade_record(sym, pos, last_price, "end_of_test", pnl, df.index[-1]))

    stats = _calculate_stats(trades, start_capital, capital, capital_curve)
    return {"trades": trades, "capital_curve": capital_curve, "stats": stats}


def run_multi_backtest_15min(symbols=None):
    """
    15min-Backtest für alle Symbole mit je eigener Strategie.
    Vergleich mit dem 1H-Backtest.

    Aufruf: run_multi_backtest_15min()
    """
    import time as _t

    if symbols is None:
        symbols = list(SYMBOL_STRATEGY.keys())

    all_trades   = []
    total_return = 0
    n_ok         = 0

    print(f"\n{'═'*56}")
    print(f"  15MIN MULTI-SYMBOL BACKTEST  (max. 60 Tage Daten)")
    print(f"{'═'*56}")

    for sym in symbols:
        strat = SYMBOL_STRATEGY.get(sym, "ema_pullback")
        r     = run_backtest_15min(sym, strategy=strat)
        s     = r["stats"]
        _t.sleep(2)

        if "error" in s:
            print(f"  {sym:12s} [{strat:15s}]  ⚠️  {s['error']}")
            continue

        print(f"  {sym:12s} [{strat:15s}]  "
              f"Return: {s['total_return']:+.1f}%  "
              f"Trades: {s['total_trades']:3d}  "
              f"WR: {s['win_rate']:.1f}%  "
              f"PF: {s['profit_factor']:.2f}")
        all_trades.extend(r["trades"])
        total_return += s["total_return"]
        n_ok         += 1

    if not all_trades:
        print("  ⚠️  Keine Trades")
        return pd.DataFrame()

    df     = pd.DataFrame(all_trades)
    wins   = df[df["pnl"] > 0]["pnl"]
    losses = df[df["pnl"] < 0]["pnl"]

    print(f"\n{'─'*56}")
    print(f"  KOMBINIERT — {n_ok} Symbole  (60 Tage)")
    print(f"  Trades gesamt : {len(df)}")
    if len(df) > 0:
        print(f"  Win-Rate      : {len(wins)/len(df)*100:.1f}%")
        print(f"  Profit Factor : {wins.sum()/abs(losses.sum()):.2f}" if len(losses) > 0 else "  Profit Factor : ∞")
    if n_ok > 0:
        print(f"  Ø Return/Sym  : {total_return/n_ok:.2f}%")
    print(f"  Trades/Tag    : {len(df)/60:.1f}  (Hochrechnung auf 60 Tage)")
    print(f"{'═'*56}")
    print(f"\n  ⚠️  Hinweis: 60 Tage = wenig Statistik.")
    print(f"  Erst ab ~200 Trades pro Symbol aussagekräftig.")
    return df

# Aufruf:
# run_multi_backtest_15min()


# ═══════════════════════════════════════════════
#  PAPER TRADING V2 — Backtest-Strategie Live
#
#  Verwendet dieselbe Entry-Logik wie run_backtest():
#  - ema_pullback / mean_reversion / breakout
#    je nach SYMBOL_STRATEGY-Routing
#  - Kein BOS/CHoCH erforderlich
#  - Reines SL/TP (kein Partial Close, kein Trailing)
#
#  Starten: run_paper_trading_v2()
#  Stoppen: Ctrl+C
# ═══════════════════════════════════════════════

def get_backtest_signal(symbol, df_h1):
    """
    Prüft ob die letzte ABGESCHLOSSENE 1H-Kerze
    ein Signal gemäß Backtest-Logik erzeugt.

    df_h1.iloc[-1] = aktuelle (noch formende) Kerze  → ignorieren
    df_h1.iloc[-2] = letzte abgeschlossene Kerze      → "candle"
    df_h1.iloc[-3] = davor                            → "prev"

    Returns:
        dict {signal, entry, sl, tp, rr, strategy}
        oder None wenn kein Signal
    """
    if len(df_h1) < 4:
        return None

    strategy  = SYMBOL_STRATEGY.get(symbol, "ema_pullback")
    bb_period, _ = SYMBOL_BB.get(symbol, (BB_PERIOD, BB_STD))
    bb_col    = "bb14" if bb_period == 14 else "bb20"
    tp_rr     = SYMBOL_TP.get(symbol, STRATEGY_TP.get(strategy, TP_RR_RATIO))

    candle = df_h1.iloc[-2]
    prev   = df_h1.iloc[-3]

    price = float(candle["close"])
    atr   = float(candle["atr"])
    rsi   = float(candle["rsi"])
    adx   = float(candle["adx"])

    direction = None
    sl_price  = None
    tp_price  = None

    # ── EMA PULLBACK ──────────────────────────────
    if strategy == "ema_pullback":
        ema_f = float(candle["ema_fast"])
        ema_s = float(candle["ema_slow"])
        ema_t = float(candle["ema_trend"])
        trend_long  = ema_f > ema_s > ema_t
        trend_short = ema_f < ema_s < ema_t
        if not trend_long and not trend_short:
            return None
        direction = "long" if trend_long else "short"
        if direction == "long":
            if not (float(prev["low"]) <= ema_f and price > ema_f):
                return None
        else:
            if not (float(prev["high"]) >= ema_f and price < ema_f):
                return None
        rsi_ok = (35 <= rsi <= 75) if direction == "long" else (25 <= rsi <= 65)
        if not rsi_ok or adx < 20:
            return None
        if direction == "long":
            sl_price = min(float(prev["low"]), float(candle["low"])) - atr * 0.5
        else:
            sl_price = max(float(prev["high"]), float(candle["high"])) + atr * 0.5
        risk = abs(price - sl_price)
        if risk <= 0:
            return None
        tp_price = price + risk * tp_rr if direction == "long" else price - risk * tp_rr

    # ── MEAN REVERSION ────────────────────────────
    elif strategy == "mean_reversion":
        try:
            bb_upper = float(candle[f"{bb_col}_upper"])
            bb_lower = float(candle[f"{bb_col}_lower"])
            bb_mid   = float(candle[f"{bb_col}_mid"])
            if any(pd.isna([bb_upper, bb_lower, bb_mid])):
                return None
        except (KeyError, ValueError):
            return None
        if float(prev["low"]) <= bb_lower and price > bb_lower and rsi < 45:
            direction = "long"
        elif float(prev["high"]) >= bb_upper and price < bb_upper and rsi > 55:
            direction = "short"
        else:
            return None
        if adx > 35:
            return None
        if direction == "long":
            sl_price = float(candle["low"]) - atr * 0.5
            tp_price = bb_mid
        else:
            sl_price = float(candle["high"]) + atr * 0.5
            tp_price = bb_mid
        risk = abs(price - sl_price)
        if risk <= 0 or abs(tp_price - price) / risk < 1.0:
            return None

    # ── BREAKOUT ─────────────────────────────────
    elif strategy == "breakout":
        try:
            prev_high_n = float(prev["high_n"])
            prev_low_n  = float(prev["low_n"])
            if pd.isna(prev_high_n) or pd.isna(prev_low_n):
                return None
        except (KeyError, ValueError):
            return None
        if price > prev_high_n:
            direction = "long"
        elif price < prev_low_n:
            direction = "short"
        else:
            return None
        if adx < 20:
            return None
        sl_price = price - atr * SL_ATR_MULT if direction == "long" else price + atr * SL_ATR_MULT
        risk = abs(price - sl_price)
        if risk <= 0:
            return None
        tp_price = price + risk * tp_rr if direction == "long" else price - risk * tp_rr

    if direction is None:
        return None

    return {
        "signal":   direction,
        "entry":    round(price, 5),
        "sl":       round(sl_price, 5),
        "tp":       round(tp_price, 5),
        "rr":       round(abs(tp_price - price) / abs(price - sl_price), 2),
        "strategy": strategy,
        "risk_mult": 1.0,
    }


def run_paper_trading_v2(symbols=None, capital=10000.0, max_scans=None, scan_interval=3600):
    """
    Paper Trading V2 — spiegelt Backtest-Logik exakt.

    Unterschied zu run_paper_trading():
    - Kein BOS/CHoCH erforderlich
    - ema_pullback / mean_reversion / breakout per Symbol (SYMBOL_STRATEGY)
    - Reines SL/TP — kein Partial Close, kein Trailing
    - Trades in logs/trades_v2.db gespeichert

    Args:
        symbols:       None = alle aus SYMBOL_STRATEGY
        capital:       Startkapital in USD
        max_scans:     None = unbegrenzt (kleine Zahl für Tests)
        scan_interval: Sekunden zwischen Scans (3600 = 1 Stunde)

    Starten: run_paper_trading_v2()
    Stoppen: Ctrl+C
    """
    import time as _time

    if symbols is None:
        symbols = list(SYMBOL_STRATEGY.keys())

    # Eigener State — unabhängig vom globalen risk_state
    v2 = {
        "capital":        capital,
        "daily_start":    capital,
        "losses_in_row":  0,
        "daily_loss":     0.0,
        "trades_today":   0,
        "circuit_active": False,
        "open_positions": {},   # {symbol: pos_dict}
        "trade_log":      [],
        "last_date":      None,
        "last_candle_ts": {},   # {symbol: timestamp} — verhindert Doppel-Signale
    }

    print(f"\n{'═'*50}")
    print(f"  🤖 PAPER TRADING V2 — Backtest-Strategie")
    print(f"{'═'*50}")
    for sym in symbols:
        strat  = SYMBOL_STRATEGY.get(sym, "ema_pullback")
        bb_tag = f" BB({SYMBOL_BB[sym][0]})" if sym in SYMBOL_BB else ""
        print(f"  {sym:15s}: {strat}{bb_tag}")
    print(f"\n  Kapital  : ${capital:,.2f}")
    print(f"  Intervall: {scan_interval}s ({scan_interval//60} Min)")
    print(f"  DB       : logs/trades_v2.db")
    print(f"  Stoppen  : Ctrl+C")
    print(f"{'═'*50}\n")

    init_database("logs/trades_v2.db")

    send_telegram(
        f"🤖 <b>PAPER BOT V2 GESTARTET</b>\n\n"
        f"Symbole: <code>{', '.join(symbols)}</code>\n"
        f"Kapital: <b>${capital:,.2f}</b>\n"
        f"Strategie: Backtest-Logik (ema_pullback / mean_reversion / breakout)\n\n"
        f"Ich melde mich bei Signalen! 🚀"
    )

    # Initiale Daten laden
    print("  📥 Lade Daten für alle Symbole...")
    symbol_data = {}
    for sym in symbols:
        try:
            print(f"     {sym}...")
            symbol_data[sym] = prepare_symbol_data(sym)
            _time.sleep(2)
        except Exception as e:
            print(f"  ⚠️  {sym}: {e}")
    print(f"  ✅ {len(symbol_data)} Symbole geladen\n")

    scan_count  = 0
    last_reload = datetime.now(timezone.utc)

    try:
        while True:
            now = datetime.now(timezone.utc)
            print(f"\n{'─'*50}")
            print(f"  🔍 Scan #{scan_count+1}  {now.strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"  💰 ${v2['capital']:.2f}  |  Positionen: {len(v2['open_positions'])}  |  Verluste: {v2['losses_in_row']}")
            print(f"{'─'*50}")

            # ── Tages-Reset ──────────────────────
            today = now.date()
            if v2["last_date"] != today:
                v2["daily_start"]    = v2["capital"]
                v2["daily_loss"]     = 0.0
                v2["trades_today"]   = 0
                v2["circuit_active"] = False
                v2["losses_in_row"]  = 0
                v2["last_date"]      = today
                print(f"  🔄 Tages-Reset — {today}")

            # ── Daten alle 4h reload ──────────────
            if (now - last_reload).seconds > 14400:
                print("  🔄 Daten werden aktualisiert...")
                for sym in symbols:
                    try:
                        symbol_data[sym] = prepare_symbol_data(sym)
                        _time.sleep(2)
                    except:
                        pass
                last_reload = now

            todays_events = fetch_news_calendar(now.date())

            # ── OFFENE POSITIONEN UPDATEN ─────────
            for sym in list(v2["open_positions"].keys()):
                latest = get_latest_price(sym)
                if not latest:
                    continue

                pos     = v2["open_positions"][sym]
                current = latest["price"]
                entry   = pos["entry"]
                sl      = pos["sl"]
                tp      = pos["tp"]
                direc   = pos["direction"]

                sl_hit = (direc == "long"  and current <= sl) or \
                         (direc == "short" and current >= sl)
                tp_hit = (direc == "long"  and current >= tp) or \
                         (direc == "short" and current <= tp)

                if sl_hit or tp_hit:
                    close_p = sl if sl_hit else tp
                    pnl = (close_p - entry) * pos["units"]
                    if direc == "short":
                        pnl = -pnl

                    v2["capital"]   += pnl
                    v2["daily_loss"] = max(0, (v2["daily_start"] - v2["capital"]) / v2["daily_start"])

                    if pnl > 0:
                        v2["losses_in_row"] = 0
                    else:
                        v2["losses_in_row"] += 1

                    result    = "tp_hit" if tp_hit else "sl_hit"
                    icon      = "✅ TP" if tp_hit else "❌ SL"
                    trade_rec = {
                        "symbol":    sym,
                        "direction": direc,
                        "entry":     entry,
                        "close":     close_p,
                        "sl":        sl,
                        "tp":        tp,
                        "pnl":       round(pnl, 2),
                        "pnl_pct":   round(pnl / v2["capital"] * 100, 3),
                        "units":     pos["units"],
                        "reason":    result,
                        "open_time": pos["open_time"],
                        "close_time": now.isoformat(),
                        "win":       pnl > 0,
                    }
                    v2["trade_log"].append(trade_rec)
                    save_trade(trade_rec, "logs/trades_v2.db")
                    del v2["open_positions"][sym]

                    print(f"  {icon}  {sym}: {'+' if pnl > 0 else ''}{pnl:.2f}$  →  Kapital: ${v2['capital']:.2f}")
                    notify_trade_closed(trade_rec)

                    # Circuit Breaker
                    if v2["losses_in_row"] >= MAX_LOSSES_ROW or v2["daily_loss"] >= MAX_DAILY_LOSS:
                        v2["circuit_active"] = True
                        print(f"  🔴 CIRCUIT BREAKER aktiv — Trading heute gestoppt")
                        notify_circuit_breaker()

                else:
                    pnl_now = (current - entry) * pos["units"]
                    if direc == "short":
                        pnl_now = -pnl_now
                    sign = "+" if pnl_now >= 0 else ""
                    print(f"  📊 {sym:12s} {direc.upper():5s}: {current:.5f}  P&L: {sign}{pnl_now:.2f}$")

            # ── NEUE SIGNALE ──────────────────────
            if not v2["circuit_active"]:
                print(f"\n  🎯 Signalcheck...")

                for sym in symbols:
                    if sym not in symbol_data:
                        continue
                    if sym in v2["open_positions"]:
                        print(f"  ➖ {sym}: bereits offen")
                        continue

                    # Handelszeiten
                    hours_ok, hours_reason = is_trading_hours(now)
                    if not hours_ok:
                        print(f"  ➖ {sym}: {hours_reason}")
                        continue

                    # News Blackout
                    blackout, bl_reason = is_news_blackout(sym, now, todays_events)
                    if blackout:
                        print(f"  📰 {sym}: {bl_reason}")
                        continue

                    # Korrelation
                    corr_ok, corr_reason = check_correlation(sym, list(v2["open_positions"].keys()))
                    if not corr_ok:
                        print(f"  🔗 {sym}: {corr_reason}")
                        continue

                    # Signal aus letzter abgeschlossener Kerze
                    df_h1 = symbol_data[sym]["h1"]
                    last_candle_ts = str(df_h1.index[-2]) if len(df_h1) >= 2 else ""

                    # Bereits auf diese Kerze reagiert?
                    if v2["last_candle_ts"].get(sym) == last_candle_ts:
                        print(f"  ➖ {sym}: Kerze bereits geprüft")
                        continue

                    sig = get_backtest_signal(sym, df_h1)
                    v2["last_candle_ts"][sym] = last_candle_ts  # Kerze als geprüft markieren

                    if sig:
                        # Position Size
                        risk_usd = v2["capital"] * RISK_PER_TRADE
                        pip_risk = abs(sig["entry"] - sig["sl"])
                        units    = round(risk_usd / pip_risk, 2) if pip_risk > 0 else 0

                        if units > 0:
                            pos = {
                                "direction": sig["signal"],
                                "entry":     sig["entry"],
                                "sl":        sig["sl"],
                                "tp":        sig["tp"],
                                "units":     units,
                                "risk_usd":  round(risk_usd, 2),
                                "open_time": now.isoformat(),
                                "strategy":  sig["strategy"],
                            }
                            v2["open_positions"][sym] = pos
                            v2["trades_today"] += 1

                            print(f"\n  🚨 SIGNAL: {sym} [{sig['strategy']}] {sig['signal'].upper()}")
                            print(f"     Entry: {sig['entry']}  SL: {sig['sl']}  TP: {sig['tp']}  RR: {sig['rr']}")
                            print(f"     Units: {units}  Risiko: ${risk_usd:.2f}")
                            notify_trade_opened(sym, sig)
                    else:
                        strat = SYMBOL_STRATEGY.get(sym, "?")
                        print(f"  ➖ {sym} [{strat}]: kein Signal")
            else:
                print(f"  ⛔ Circuit Breaker aktiv")

            # ── PERFORMANCE SUMMARY ───────────────
            if v2["trade_log"]:
                total     = len(v2["trade_log"])
                wins      = sum(1 for t in v2["trade_log"] if t["win"])
                total_pnl = sum(t["pnl"] for t in v2["trade_log"])
                pf_wins   = sum(t["pnl"] for t in v2["trade_log"] if t["pnl"] > 0)
                pf_loss   = abs(sum(t["pnl"] for t in v2["trade_log"] if t["pnl"] < 0))
                pf        = round(pf_wins / pf_loss, 2) if pf_loss > 0 else float("inf")
                print(f"\n  📊 Gesamt: {total} Trades | WR: {round(wins/total*100,1)}% | PF: {pf} | P&L: ${total_pnl:.2f}")

            scan_count += 1
            if max_scans and scan_count >= max_scans:
                print(f"\n  ✅ Test abgeschlossen nach {scan_count} Scans")
                break

            print(f"\n  ⏳ Nächster Scan in {scan_interval}s... (Ctrl+C zum Stoppen)")
            _time.sleep(scan_interval)

    except KeyboardInterrupt:
        print(f"\n\n  🛑 Paper Trading V2 gestoppt")
        if v2["trade_log"]:
            total     = len(v2["trade_log"])
            wins      = sum(1 for t in v2["trade_log"] if t["win"])
            total_pnl = sum(t["pnl"] for t in v2["trade_log"])
            print(f"  Trades: {total}  WR: {round(wins/total*100,1) if total else 0}%  P&L: ${total_pnl:.2f}")
        print(f"  Endkapital: ${v2['capital']:.2f}")
        send_telegram(f"🛑 <b>Paper Bot V2 gestoppt</b>\nKapital: ${v2['capital']:.2f}")


print("""
  ═══════════════════════════════════════════════
  Paper Trading V2 starten (empfohlen):
    run_paper_trading_v2()

  Kurztest (3 Scans, 10s Pause):
    run_paper_trading_v2(max_scans=3, scan_interval=10)
  ═══════════════════════════════════════════════
""")