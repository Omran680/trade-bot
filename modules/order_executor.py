# modules/order_executor.py
import time
from datetime import datetime
from pathlib import Path
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from core.logger import get_logger
from modules.ibkr_connector import IBKRConnector

log = get_logger()

class OrderExecutor:
    def __init__(self):
        try:
            self.ib = IBKRConnector()
            log.info("OrderExecutor initialise")
        except Exception as e:
            log.error(f"Erreur initialisation: {e}")
            raise
        
    def get_price(self) -> float:
        try:
            return self.ib.get_price()
        except Exception as e:
            log.error(f"Erreur get_price: {e}")
            return 2650.0
        
    def get_position(self) -> float:
        try:
            return self.ib.get_position()
        except Exception as e:
            log.error(f"Erreur get_position: {e}")
            return 0.0
        
    def has_position(self) -> bool:
        return self.get_position() != 0
        
    def get_position_info(self):
        try:
            return self.ib.get_position_info()
        except Exception as e:
            log.error(f"Erreur get_position_info: {e}")
            return None
        
    def available_cash(self) -> float:
        try:
            cash = self.ib.available_cash()
            return cash if cash > 0 else config.CAPITAL
        except Exception as e:
            log.error(f"Erreur available_cash: {e}")
            return config.CAPITAL
        
  # modules/order_executor.py - Modification de calc_qty

def calc_qty(self, price: float, atr: float = None) -> float:
    """Calcule la quantité basée sur le risque - retourne un entier"""
    cash = min(self.available_cash(), 100000)
    risk_amount = cash * config.RISK_PCT
    
    if atr is None or atr <= 0:
        atr = 1.5
    
    # Calcul de la quantité
    qty = max(1, round(risk_amount / atr))  # Arrondir à l'entier le plus proche, minimum 1
    
    # Limiter à la quantité max possible
    max_qty = int(cash / price) if price > 0 else 1
    qty = min(qty, max_qty)
    
    # S'assurer que c'est un entier
    qty = int(qty)
    
    # Minimum 1 unité
    if qty < 1:
        qty = 1
        
    return qty
        
    def enter(self, side: str, price: float, atr: float) -> bool:
        qty = self.calc_qty(price, atr)
        
        if qty <= 0:
            log.error("Quantite invalide")
            return False
            
        log.info(f"{'BUY' if side=='buy' else 'SELL'} {qty} oz @ ${price:.2f}")
        
        try:
            success = self.ib.place_order(side.upper(), qty)
            if success:
                log.info("Ordre execute")
                self._log_trade(side, qty, price)
                return True
            else:
                log.error("Ordre non execute")
                return False
        except Exception as e:
            log.error(f"Erreur ordre: {e}")
            return False
            
    def close(self) -> bool:
        try:
            success = self.ib.close_position()
            if success:
                log.info("Position fermee")
            return success
        except Exception as e:
            log.error(f"Erreur fermeture: {e}")
            return False
            
    def get_trade_history(self, count=3):
        try:
            trades_file = Path("data/trades.json")
            if trades_file.exists():
                trades = json.loads(trades_file.read_text())
                return trades[-count:] if trades else []
        except Exception as e:
            log.error(f"Erreur lecture historique: {e}")
        return []
        
    def _log_trade(self, side, qty, price):
        trades_file = Path("data/trades.json")
        trades_file.parent.mkdir(exist_ok=True)
        try:
            trades = json.loads(trades_file.read_text()) if trades_file.exists() else []
        except Exception:
            trades = []
            
        trades.append({
            "ts": datetime.utcnow().isoformat(),
            "symbol": config.SYMBOL,
            "side": side,
            "qty": qty,
            "open_price": price,
            "close_price": None,
            "profit": 0,
            "units": qty,
            "type": side.upper()
        })
        
        trades_file.write_text(json.dumps(trades, indent=2))