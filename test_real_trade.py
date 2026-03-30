# test_real_trade.py
from modules.order_executor import OrderExecutor
import time

ex = OrderExecutor()

# Vérifier le cash disponible
cash = ex.available_cash()
print(f"Cash disponible: ${cash:.2f}")

# Vérifier la position actuelle
pos = ex.get_position_info()
print(f"Position: {pos}")

# Demander confirmation
print("\n" + "="*50)
print("ATTENTION: Ceci va passer un VRAI ordre sur le marché!")
response = input("Voulez-vous tester un ordre BUY? (oui/non): ")

if response.lower() == "oui":
    price = ex.get_price()
    atr = price * 0.002
    success = ex.enter("buy", price, atr)
    
    if success:
        print("✅ Ordre BUY envoyé!")
        time.sleep(3)
        
        # Vérifier la nouvelle position
        new_pos = ex.get_position_info()
        print(f"Nouvelle position: {new_pos}")
        
        # Fermer après 10 secondes
        print("Attente 10 secondes avant fermeture...")
        time.sleep(10)
        
        print("Fermeture de la position...")
        ex.close()
        print("✅ Ordre de fermeture envoyé!")
    else:
        print("❌ Échec de l'ordre")