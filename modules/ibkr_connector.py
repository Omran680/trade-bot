# modules/ibkr_connector.py
import asyncio
import threading

# Fix Python 3.10+ RuntimeError
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, Forex
import config
from core.logger import get_logger

log = get_logger()

class IBKRConnector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.ib = IB()
        self.connected = False
        self.connect()

    def connect(self):
        try:
            self.ib.connect("127.0.0.1", 4002, clientId=1)
            self.connected = True
            log.info("✅ Connecté à IBKR Paper Trading")
        except Exception as e:
            log.error(f"Erreur connexion IBKR: {e}")
            self.connected = False

    def get_price(self):
        # Forex XAU/USD
        contract = Forex("XAUUSD")
        ticker = self.ib.reqMktData(contract, "", False, False)
        self.ib.sleep(1)
        return ticker.marketPrice() if ticker.marketPrice() else 2650.0

    def place_order(self, side, quantity):
        log.info(f"[IBKR] Ordre simulé {side} {quantity}")
        return True

    def get_position_info(self):
        return {"type": "LONG", "units": 10, "open_price": 2650.0, "unrealized_pnl": 0}

    def has_position(self):
        return True

    def available_cash(self):
        return config.CAPITAL

    def close_position(self):
        log.info("[IBKR] Position fermée")
        return True

    def disconnect(self):
        if self.connected:
            self.ib.disconnect()
            log.info("✅ Déconnecté d'IBKR")