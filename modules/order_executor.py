# modules/order_executor.py
import config
from modules.ibkr_connector import IBKRConnector
from core.logger import get_logger
from datetime import datetime

log = get_logger()

class OrderExecutor:
    def __init__(self):
        self.connector = IBKRConnector()
        self.trade_history = []

    def get_price(self):
        return self.connector.get_price()

    def available_cash(self):
        return self.connector.available_cash()

    def has_position(self):
        return self.connector.has_position()

    def get_position_info(self):
        return self.connector.get_position_info()

    def enter(self, side, price, atr):
        size = max(1, int(config.CAPITAL * config.RISK_PCT / max(atr*1.5,5)))
        success = self.connector.place_order(side.upper(), size)
        if success:
            self.record_trade({"type": side.upper(), "size": size, "price": price, "timestamp": datetime.now()})
        return success

    def close(self):
        success = self.connector.close_position()
        if success:
            self.record_trade({"type": "CLOSE", "timestamp": datetime.now()})
        return success

    def record_trade(self, trade_info):
        self.trade_history.append(trade_info)
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

    def get_trade_history(self, limit=None):
        return self.trade_history[-limit:] if limit else self.trade_history

    def disconnect(self):
        self.connector.disconnect()