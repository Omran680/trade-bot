# test_ib_sync.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ib_insync import IB, Forex

print("Test de connexion synchrone...")
ib = IB()

try:
    ib.connect('127.0.0.1', 4002, clientId=3, timeout=10)
    print("✅ Connecté à IB Gateway!")
    
    contract = Forex("XAUUSD")
    ib.qualifyContracts(contract)
    print(f"✅ Contrat qualifié: {contract}")
    
    # Test prix
    ticker = ib.reqMktData(contract, '', False, False)
    ib.sleep(2)
    
    if ticker.bid and ticker.ask:
        print(f"✅ Prix bid: {ticker.bid:.2f}, ask: {ticker.ask:.2f}")
        print(f"✅ Mid price: {(ticker.bid + ticker.ask)/2:.2f}")
    
    ib.disconnect()
    print("✅ Déconnecté")
    
except Exception as e:
    print(f"❌ Erreur: {e}")