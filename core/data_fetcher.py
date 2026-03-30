# core/data_fetcher.py
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from core.logger import get_logger

log = get_logger()

def fetch_bars(period="1 D", bar_size="1 min"):
    """Génère des données simulées pour les indicateurs"""
    # Créer des données de test
    end = datetime.now()
    start = end - timedelta(days=1)
    
    dates = pd.date_range(start=start, end=end, freq='1min')
    base_price = 2650.0
    
    # Simuler des variations de prix
    np.random.seed(42)
    returns = np.random.randn(len(dates)) * 0.0005
    prices = base_price * (1 + np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * 0.9995,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'close': prices,
        'volume': np.random.randint(100, 1000, len(dates))
    }, index=dates)
    
    return df

def get_price():
    """Récupère le prix depuis l'executor"""
    try:
        from modules.order_executor import OrderExecutor
        ex = OrderExecutor()
        return ex.get_price()
    except Exception as e:
        log.error(f"Erreur get_price: {e}")
        return 2650.0

def market_is_open():
    """Vérifie si le marché est ouvert"""
    now = datetime.utcnow()
    
    if now.weekday() >= 5:
        if now.weekday() == 6 and now.hour >= 23:
            return True
        return False
    
    if now.weekday() == 4 and now.hour >= 23:
        return False
    
    return True