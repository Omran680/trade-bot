import time
import sys
import os
import asyncio

# Fix asyncio pour Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.dirname(__file__))

from modules.order_executor import OrderExecutor
from modules.ibkr_connector import IBKRConnector

def main():
    print("=== TEST PAPER TRADING XAU/USD ===")

    # Initialiser l'executor
    executor = OrderExecutor()

    # Vérifier la connexion
    if not executor.connector.connected:
        print("⚠️  Pas connecté à IBKR, vérifier IB Gateway / TWS")
        return

    print("✅ Connecté à IBKR Paper Trading")

    # Afficher le prix actuel
    price = executor.get_price()
    print(f"Prix actuel XAU/USD : ${price:.2f}")

    # Forcer un signal BUY pour test
    signal = "BUY"
    atr = 10  # valeur arbitraire pour le calcul de taille
    print(f"Signal forcé : {signal}")

    success = executor.enter(signal.lower(), price, atr)
    if success:
        print(f"✅ Ordre {signal} exécuté avec succès")
    else:
        print(f"❌ Échec de l'ordre {signal}")

    # Afficher info position
    pos_info = executor.get_position_info()
    if pos_info:
        print("📊 Info position :")
        print(pos_info)
    else:
        print("📊 Pas de position ouverte")

    # Attendre 10 secondes avant fermeture
    print("⏳ Attente 10s avant fermeture...")
    time.sleep(10)

    # Fermer la position pour reset
    print("🛑 Fermeture de la position")
    executor.close()

    # Déconnexion propre
    executor.disconnect()
    print("✅ Déconnecté d'IBKR")

if __name__ == "__main__":
    main()