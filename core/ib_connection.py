# core/ib_connection.py
import asyncio
import threading
from ib_insync import IB, Forex

class IBConnectionManager:
    """Gestionnaire de connexion IBKR avec boucle d'événements dédiée"""
    
    _instance = None
    _loop = None
    _thread = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialise la boucle d'événements dans un thread dédié"""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            # Attendre que la boucle soit prête
            time.sleep(0.1)
    
    def _run_loop(self):
        """Exécute la boucle d'événements"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
    
    def run_sync(self, coro):
        """Exécute une coroutine de manière synchrone"""
        if self._loop is None:
            self._initialize()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=10)
    
    def get_ib(self):
        """Retourne une instance IB connectée"""
        return self.run_sync(self._connect_ib())
    
    async def _connect_ib(self):
        """Coroutine de connexion"""
        ib = IB()
        await ib.connectAsync('127.0.0.1', 4002, clientId=1)
        return ib

import time