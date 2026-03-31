# modules/ibkr_connector.py
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import threading
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from core.logger import get_logger

log = get_logger()

class IBApp(EWrapper, EClient):
    """Application IBKR avec l'API officielle"""
    
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.position = 0
        self.avg_cost = 0
        self.unrealized_pnl = 0
        self.account_value = config.CAPITAL
        self.current_price = 2650.0
        self.bid = 0
        self.ask = 0
        self.order_id = 1
        self.order_filled = False
        self.positions = {}
        self.connected_event = threading.Event()
        
    def error(self, reqId, errorCode, errorString):
        # Ignorer les erreurs non critiques
        if errorCode in [2104, 2106, 2107, 2108, 2158]:
            log.debug(f"IBKR Info: {errorString}")
        elif errorCode == 326:
            log.warning(f"Client ID deja utilise: {errorString}")
        elif errorCode == 354:
            log.warning("Donnees de marche en differe (abonnement requis pour temps reel)")
        elif errorCode == 10167:
            log.warning("Donnees de marche en differe affichees")
        else:
            log.error(f"IBKR Error {errorCode}: {errorString}")
        
    def nextValidId(self, orderId):
        self.order_id = orderId
        log.info(f"Next valid order ID: {self.order_id}")
        
    def connectAck(self):
        log.info("Connection acknowledged")
        
    def connectionClosed(self):
        log.warning("Connection closed")
        self.connected = False
        
    def position(self, account, contract, position, avgCost):
        # Chercher le symbole correct
        symbol_to_check = config.SYMBOL.replace("/", "")
        if contract.symbol == symbol_to_check or contract.symbol == "XAUUSD":
            self.position = position
            self.avg_cost = avgCost
            if position != 0:
                log.info(f"Position: {position} @ {avgCost:.2f}")
            
    def positionEnd(self):
        pass
        
    def pnl(self, reqId, dailyPnL, unrealizedPnL, realizedPnL):
        self.unrealized_pnl = unrealizedPnL
        
    def accountSummary(self, reqId, account, tag, value, currency):
        if tag == "AvailableFunds":
            try:
                self.account_value = float(value)
                log.debug(f"Available funds: ${self.account_value:.2f}")
            except:
                pass
        elif tag == "NetLiquidation":
            try:
                self.net_liquidation = float(value)
            except:
                pass
            
    def accountSummaryEnd(self, reqId):
        pass
        
    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == 1:  # BID
            self.bid = price
        elif tickType == 2:  # ASK
            self.ask = price
            
        if self.bid > 0 and self.ask > 0:
            self.current_price = (self.bid + self.ask) / 2
            log.debug(f"Prix mis a jour: Bid={self.bid}, Ask={self.ask}, Mid={self.current_price:.2f}")
            
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, 
                    permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        log.info(f"Order {orderId} status: {status}, filled: {filled}")
        if status == "Filled":
            self.order_filled = True

class IBKRConnector:
    """Connecteur synchrone pour IBKR"""
    
    _instance = None
    _client_id_counter = 1
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.app = IBApp()
        self.app.connected = False
        self.thread = None
        self.connected = False
        self.connect()
        
    def connect(self, max_retries=3):
        """Connecte à IB Gateway/TWS avec retry"""
        for attempt in range(max_retries):
            try:
                # Utiliser un client ID incrémental pour éviter les conflits
                client_id = self._client_id_counter
                self._client_id_counter += 1
                
                # FORCER le port 4002 pour IB Gateway
                port = 4002  # ← Changement important: 4002 pour IB Gateway
                
                log.info(f"Tentative de connexion avec clientId={client_id} sur port {port}...")
                self.app.connect("127.0.0.1", port, clientId=client_id)
                self.thread = threading.Thread(target=self._run_loop, daemon=True)
                self.thread.start()
                
                # Attendre la connexion
                for _ in range(20):
                    time.sleep(0.5)
                    if self.app.isConnected():
                        self.app.connected = True
                        self.connected = True
                        break
                        
                if not self.app.connected:
                    log.warning(f"Timeout de connexion (tentative {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        log.error("Timeout de connexion apres plusieurs tentatives")
                        return False
                        
                log.info("Connecte a IBKR via IBAPI")
                
                # Configurer le contrat pour XAU/USD
                self.contract = Contract()
                self.contract.symbol = "XAUUSD"
                self.contract.secType = "CFD"
                self.contract.exchange = "SMART"
                self.contract.currency = "USD"
                
                log.info(f"Contrat configure: {self.contract.symbol} {self.contract.secType} {self.contract.exchange}")
                
                # Demander les donnees
                time.sleep(1)
                self.app.reqPositions()
                self.app.reqAccountSummary(1, "All", "AvailableFunds")
                self.app.reqMarketDataType(3)  # 3 = delayed frozen data
                self.app.reqMktData(1, self.contract, "", False, False, [])
                
                return True
                
            except Exception as e:
                log.error(f"Erreur connexion (tentative {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return False
        return False
            
    def _run_loop(self):
        """Exécute la boucle de l'API"""
        self.app.run()
        
    def get_price(self) -> float:
        """Récupère le prix actuel"""
        for _ in range(10):
            if self.app.current_price > 0:
                return self.app.current_price
            time.sleep(0.5)
        return 2650.0
        
    def get_position(self) -> int:
        """Récupère la position"""
        return self.app.position
        
    def get_position_info(self):
        """Retourne les infos de position"""
        if self.app.position != 0:
            return {
                'type': 'LONG' if self.app.position > 0 else 'SHORT',
                'units': abs(self.app.position),
                'open_price': self.app.avg_cost,
                'unrealized_pnl': self.app.unrealized_pnl
            }
        return None
        
    def has_position(self) -> bool:
        """Vérifie si une position existe"""
        return self.app.position != 0
        
    def available_cash(self) -> float:
        """Récupère le cash disponible"""
        if self.app.account_value > 0:
            return self.app.account_value
        return config.CAPITAL
        
    def place_order(self, side: str, quantity: int) -> bool:
        """Place un ordre - quantity doit être un entier"""
        try:
            # S'assurer que la quantité est un entier
            quantity = int(quantity)
            
            if quantity <= 0:
                log.error(f"Quantite invalide: {quantity}")
                return False
                
            order = Order()
            order.action = side
            order.orderType = "MKT"
            order.totalQuantity = quantity
            order.eTradeOnly = False
            order.firmQuoteOnly = False
            
            log.info(f"Placement ordre {side} {quantity} units")
            
            self.app.order_filled = False
            self.app.placeOrder(self.app.order_id, self.contract, order)
            self.app.order_id += 1
            
            # Attendre l'execution
            for _ in range(15):
                time.sleep(1)
                if self.app.order_filled:
                    log.info(f"Ordre {side} {quantity} execute")
                    return True
                    
            log.warning("Ordre non execute apres 15 secondes")
            return False
            
        except Exception as e:
            log.error(f"Erreur placement ordre: {e}")
            return False
        
    def close_position(self) -> bool:
        """Ferme la position"""
        if self.app.position != 0:
            side = "SELL" if self.app.position > 0 else "BUY"
            return self.place_order(side, abs(self.app.position))
        return True
        
    def disconnect(self):
        """Deconnecte proprement"""
        try:
            if hasattr(self, 'app') and self.app.isConnected():
                self.app.disconnect()
                log.info("Deconnecte d'IBKR")
                self.connected = False
        except:
            pass
    
    def __del__(self):
        """Destructeur"""
        try:
            self.disconnect()
        except:
            pass