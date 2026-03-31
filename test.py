# test_ibkr_simple.py
import time
import sys
sys.path.insert(0, '.')

print("=" * 50)
print("Test de connexion API IBKR")
print("=" * 50)

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    import threading
    
    class TestApp(EWrapper, EClient):
        def __init__(self):
            EClient.__init__(self, self)
            self.connected = False
            
        def error(self, reqId, errorCode, errorString):
            if errorCode in [2104, 2106, 2158]:
                print(f"[INFO] {errorString}")
            elif errorCode == 502:
                print("[ERREUR] Cannot connect to IB Gateway/TWS")
            elif errorCode == 504:
                print("[ERREUR] Not connected")
            else:
                print(f"[ERROR] {errorCode}: {errorString}")
                
        def connectAck(self):
            print("[OK] Connection acknowledged")
            
        def nextValidId(self, orderId):
            print(f"[OK] Next valid order ID: {orderId}")
            self.connected = True
            
        def connectionClosed(self):
            print("[WARNING] Connection closed")
            self.connected = False
    
    print("1. Creation de l'application...")
    app = TestApp()
    
    print("2. Connexion a IB Gateway (port 4002)...")
    app.connect("127.0.0.1", 4002, clientId=1)
    
    print("3. Demarrage du thread...")
    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()
    
    print("4. Attente de la connexion (15 secondes)...")
    for i in range(15):
        time.sleep(1)
        print(f"   ... {i+1}/15")
        if app.connected:
            break
    
    if app.connected:
        print("\n[SUCCES] Connexion etablie avec IB Gateway!")
        
        # Tester un contrat
        print("\n5. Demande de contrat XAUUSD...")
        contract = Contract()
        contract.symbol = "XAUUSD"
        contract.secType = "CFD"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        app.reqContractDetails(1, contract)
        time.sleep(2)
        
        print("\n6. Demande de donnees de marche...")
        app.reqMarketDataType(3)  # Delayed data
        app.reqMktData(1, contract, "", False, False, [])
        time.sleep(3)
        
        print("\n[SUCCES] Test complete!")
        
    else:
        print("\n[ECHEC] Impossible de se connecter")
        print("\nVerifiez dans IB Gateway:")
        print("  - Etes-vous connecte (cercle vert en bas)?")
        print("  - API activee? (File -> Global Configuration -> API)")
        print("  - Port correct? 4002")
        print("  - 'Enable ActiveX and Socket Clients' coché?")
        
    app.disconnect()
    
except Exception as e:
    print(f"[ERREUR] {e}")
    import traceback
    traceback.print_exc()