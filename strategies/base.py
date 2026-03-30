# strategies/base.py
import pandas as pd
import numpy as np

class TradeResult:
    def __init__(self, strategy, signal, entry, exit, pnl, pnl_pct):
        self.strategy = strategy
        self.signal = signal
        self.entry = entry
        self.exit = exit
        self.pnl = pnl
        self.pnl_pct = pnl_pct

class BaseStrategy:
    def __init__(self, name):
        self.name = name
        
    def signal(self, df):
        """Retourne BUY, SELL ou HOLD"""
        return "HOLD"

class EMACrossoverStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("ema_crossover")
        
    def signal(self, df):
        if df.empty or len(df) < 20:
            return "HOLD"
        
        # EMA 9 et EMA 21
        ema9 = df['close'].ewm(span=9, adjust=False).mean()
        ema21 = df['close'].ewm(span=21, adjust=False).mean()
        
        last_ema9 = ema9.iloc[-1]
        last_ema21 = ema21.iloc[-1]
        prev_ema9 = ema9.iloc[-2]
        prev_ema21 = ema21.iloc[-2]
        
        if prev_ema9 <= prev_ema21 and last_ema9 > last_ema21:
            return "BUY"
        elif prev_ema9 >= prev_ema21 and last_ema9 < last_ema21:
            return "SELL"
        return "HOLD"

class RSIStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("rsi_mean_reversion")
        
    def signal(self, df):
        if df.empty or 'RSI' not in df.columns:
            return "HOLD"
        
        rsi = df['RSI'].iloc[-1]
        
        if rsi < 30:
            return "BUY"
        elif rsi > 70:
            return "SELL"
        return "HOLD"

class BollingerStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("bollinger_breakout")
        
    def signal(self, df):
        if df.empty or len(df) < 20:
            return "HOLD"
        
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        
        last_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        if prev_close <= upper.iloc[-2] and last_close > upper.iloc[-1]:
            return "SELL"
        elif prev_close >= lower.iloc[-2] and last_close < lower.iloc[-1]:
            return "BUY"
        return "HOLD"

class VWAPStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("vwap_momentum")
        
    def signal(self, df):
        return "HOLD"

class MACDStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("macd_scalp")
        
    def signal(self, df):
        if df.empty or 'MACD' not in df.columns or 'MACD_signal' not in df.columns:
            return "HOLD"
        
        macd = df['MACD'].iloc[-1]
        signal = df['MACD_signal'].iloc[-1]
        prev_macd = df['MACD'].iloc[-2]
        prev_signal = df['MACD_signal'].iloc[-2]
        
        if prev_macd <= prev_signal and macd > signal:
            return "BUY"
        elif prev_macd >= prev_signal and macd < signal:
            return "SELL"
        return "HOLD"

class HeikenAshiStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("heiken_ashi")
        
    def signal(self, df):
        return "HOLD"

# Factory pour créer toutes les stratégies
def create_strategies():
    return [
        EMACrossoverStrategy(),
        RSIStrategy(),
        BollingerStrategy(),
        VWAPStrategy(),
        MACDStrategy(),
        HeikenAshiStrategy()
    ]