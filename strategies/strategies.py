# strategies/strategies.py
import pandas as pd
from strategies.base import BaseStrategy, Signal

class EMACrossover(BaseStrategy):
    name = "ema_crossover"
    def signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 22:
            return "HOLD"
        last, prev = df.iloc[-1], df.iloc[-2]
        buy = prev["EMA9"] < prev["EMA21"] and last["EMA9"] > last["EMA21"] and last["Close"] > last["VWAP"]
        sell = prev["EMA9"] > prev["EMA21"] and last["EMA9"] < last["EMA21"]
        return "BUY" if buy else "SELL" if sell else "HOLD"

ALL_STRATEGIES = [EMACrossover()]