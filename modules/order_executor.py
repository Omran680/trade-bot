# modules/order_executor.py
import time
from datetime import datetime
from typing import Optional, Dict, List

import config
from core.logger import get_logger
from modules.ibkr_connector import IBKRConnector

log = get_logger()

class OrderExecutor:
    def __init__(self):
        self.connector = IBKRConnector()
        self.trade_history = []
        log.info("OrderExecutor initialise avec IBKR")
        
    def get_price(self) -> float:
        return self.connector.get_price()
    
    def available_cash(self) -> float:
        return self.connector.available_cash()
    
    def has_position(self) -> bool:
        return self.connector.has_position()
    
    def get_position_info(self) -> Optional[Dict]:
        return self.connector.get_position_info()
    
    def enter(self, side: str, price: float, atr: float) -> bool:
        try:
            size = self._calculate_position_size(price, atr)
            if size <= 0:
                log.warning(f"Taille invalide: {size}")
                return False
            action = "BUY" if side == "buy" else "SELL"
            success = self.connector.place_order(action, size)
            if success:
                self.record_trade({
                    'type': side.upper(),
                    'size': size,
                    'price': price,
                    'timestamp': datetime.now(),
                    'status': 'ENTERED'
                })
                log.info(f"Ordre {action} {size} oz place a ${price:.2f}")
                return True
            return False
        except Exception as e:
            log.error(f"Erreur enter: {e}")
            return False
    
    def close(self) -> bool:
        try:
            success = self.connector.close_position()
            if success:
                self.record_trade({
                    'type': 'CLOSE',
                    'timestamp': datetime.now(),
                    'status': 'CLOSED'
                })
                log.info("Position fermee")
                return True
            return False
        except Exception as e:
            log.error(f"Erreur close: {e}")
            return False
    
    def _calculate_position_size(self, price: float, atr: float) -> int:
        try:
            risk_amount = config.CAPITAL * config.RISK_PCT
            stop_distance = max(atr * 1.5, 5)
            size = int(risk_amount / stop_distance)
            return max(1, min(size, 10))
        except:
            return 1
    
    def record_trade(self, trade_info: Dict):
        if not hasattr(self, 'trade_history'):
            self.trade_history = []
        self.trade_history.append(trade_info)
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    def get_trade_history(self, limit: int = None) -> List[Dict]:
        if not hasattr(self, 'trade_history'):
            self.trade_history = []
        if limit:
            return self.trade_history[-limit:]
        return self.trade_history
    
    def disconnect(self):
        self.connector.disconnect()