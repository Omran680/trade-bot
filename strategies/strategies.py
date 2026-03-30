"""
strategies/strategies.py — Les 6 stratégies de scalping XAU/USD
Chaque stratégie fonctionne seule via signal(df).
Fonctionne seul : python -m strategies.strategies
"""

import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from strategies.base import BaseStrategy, Signal


class EMACrossover(BaseStrategy):
    """EMA9 / EMA21 crossover + filtre VWAP."""
    name = "ema_crossover"

    def signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 22:
            return "HOLD"
        last, prev = df.iloc[-1], df.iloc[-2]
        buy = (prev["EMA9"] < prev["EMA21"] and
               last["EMA9"] > last["EMA21"] and
               40 < last["RSI"] < 65 and
               last["Close"] > last["VWAP"])
        sell = (prev["EMA9"] > prev["EMA21"] and
                last["EMA9"] < last["EMA21"]) or last["RSI"] > 70
        return "BUY" if buy else "SELL" if sell else "HOLD"


class RSIMeanReversion(BaseStrategy):
    """RSI survendu/suracheté + filtre EMA200."""
    name = "rsi_mean_reversion"

    def signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 200:
            return "HOLD"
        last = df.iloc[-1]
        buy  = last["RSI"] < 30 and last["Close"] > last["EMA200"]
        sell = last["RSI"] > 70 and last["Close"] < last["EMA200"]
        return "BUY" if buy else "SELL" if sell else "HOLD"


class BollingerBreakout(BaseStrategy):
    """Prix à la bande Bollinger + confirmation RSI."""
    name = "bollinger_breakout"

    def signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 22:
            return "HOLD"
        last = df.iloc[-1]
        buy  = last["Close"] <= last["BB_lower"] and last["RSI"] < 40
        sell = last["Close"] >= last["BB_upper"] and last["RSI"] > 60
        return "BUY" if buy else "SELL" if sell else "HOLD"


class VWAPMomentum(BaseStrategy):
    """Retour au VWAP avec confirmation de volume."""
    name = "vwap_momentum"

    def signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 20:
            return "HOLD"
        last, prev = df.iloc[-1], df.iloc[-2]
        avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
        strong  = last["Volume"] > 1.2 * avg_vol
        buy  = (prev["Close"] < prev["VWAP"] and
                last["Close"] > last["VWAP"] and strong)
        sell = (prev["Close"] > prev["VWAP"] and
                last["Close"] < last["VWAP"] and strong)
        return "BUY" if buy else "SELL" if sell else "HOLD"


class MACDScalp(BaseStrategy):
    """MACD histogram crossover + EMA21 trend filter."""
    name = "macd_scalp"

    def signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 30:
            return "HOLD"
        last, prev = df.iloc[-1], df.iloc[-2]
        buy  = (prev["MACD_hist"] < 0 and last["MACD_hist"] > 0 and
                last["Close"] > last["EMA21"])
        sell = (prev["MACD_hist"] > 0 and last["MACD_hist"] < 0 and
                last["Close"] < last["EMA21"])
        return "BUY" if buy else "SELL" if sell else "HOLD"


class HeikenAshiChannel(BaseStrategy):
    """Heiken Ashi haussier + price > EMA55."""
    name = "heiken_ashi"

    def signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 56:
            return "HOLD"
        last = df.iloc[-1]
        bull_streak = all(df["HA_Bull"].iloc[-3:])    # 3 bougies HA bull
        bear_streak = not any(df["HA_Bull"].iloc[-3:]) # 3 bougies HA bear
        buy  = bull_streak and last["Close"] > last["EMA55"]
        sell = bear_streak and last["Close"] < last["EMA55"]
        return "BUY" if buy else "SELL" if sell else "HOLD"


# Registre centralisé
ALL_STRATEGIES = [
    EMACrossover(),
    RSIMeanReversion(),
    BollingerBreakout(),
    VWAPMomentum(),
    MACDScalp(),
    HeikenAshiChannel(),
]


# ── Test standalone ───────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf
    import sys
    sys.path.insert(0, "..")
    from core.indicators import add_all
    from core.logger import get_logger
    log = get_logger()

    log.info("=== Test toutes les stratégies ===")
    raw = yf.download("GC=F", period="10d", interval="1m", auto_adjust=True)
    raw.columns = [c[0] for c in raw.columns]
    df = add_all(raw)

    for strat in ALL_STRATEGIES:
        sig = strat.signal(df)
        log.info(f"  {strat.name:<25} → {sig}")
