# strategies/base.py
from typing import Literal
import pandas as pd

Signal = Literal["BUY", "SELL", "HOLD"]

class BaseStrategy:
    name: str = "base"

    def signal(self, df: pd.DataFrame) -> Signal:
        return "HOLD"