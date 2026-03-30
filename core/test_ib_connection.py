# test_ib_sync.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Importer et tester directement
from modules.order_executor import OrderExecutor

print("Test de connexion IBKR...")
try:
    ex = OrderExecutor()
    print("✅ Connexion établie!")
    
    price = ex.get_price()
    print(f"✅ Prix actuel: ${price:.2f}")
    
    cash = ex.available_cash()
    print(f"✅ Cash disponible: ${cash:.2f}")
    
    pos = ex.get_position_info()
    if pos:
        print(f"✅ Position: {pos['type']} {pos['units']} oz")
    else:
        print("✅ Pas de position ouverte")
        
    print("✅ Test réussi!")
    
except Exception as e:
    print(f"❌ Erreur: {e}")