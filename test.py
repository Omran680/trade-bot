# test_trade.py - Script de test simplifié
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("TEST DE TRADING - XAU/USD")
print("=" * 60)

# Utiliser l'executor
from modules.order_executor import OrderExecutor

print("\n1. Connexion à IB Gateway...")
ex = OrderExecutor()
time.sleep(2)

print("\n2. Récupération des informations...")
price = ex.get_price()
cash = ex.available_cash()
pos = ex.get_position_info()

print(f"   Prix XAU/USD: ${price:.2f}")
print(f"   Cash disponible: ${cash:.2f}")
print(f"   Position actuelle: {pos if pos else 'Aucune'}")

print("\n" + "=" * 60)
print("ATTENTION: Ceci va passer un ORDRE DE MARCHÉ!")
print("   Quantité: 0.01 oz (minimum pour test)")
print("   Type: BUY puis fermeture après 10 secondes")
print("=" * 60)

response = input("\nVoulez-vous continuer? (oui/non): ")

if response.lower() == "oui":
    print("\n3. Passage de l'ordre BUY...")
    atr = price * 0.002
    success = ex.enter("buy", price, atr)
    
    if success:
        print("   ✅ Ordre BUY envoyé!")
        time.sleep(3)
        
        # Vérifier la position
        new_pos = ex.get_position_info()
        print(f"   Nouvelle position: {new_pos}")
        
        print("\n4. Attente 10 secondes...")
        for i in range(10, 0, -1):
            print(f"   Fermeture dans {i} secondes...", end="\r")
            time.sleep(1)
        
        print("\n5. Fermeture de la position...")
        ex.close()
        print("   ✅ Ordre de fermeture envoyé!")
        
        # Vérifier après fermeture
        time.sleep(2)
        final_pos = ex.get_position_info()
        print(f"   Position finale: {final_pos if final_pos else 'Aucune'}")
        
    else:
        print("   ❌ Échec de l'ordre")
else:
    print("\nTest annulé")

print("\n" + "=" * 60)
print("Test terminé")
print("=" * 60)