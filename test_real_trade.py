# test_bot.py
import time
import sys
sys.path.insert(0, '.')

print("Test du bot avec IBKR")
print("=" * 40)

from modules.order_executor import OrderExecutor

print("1. Creation de l'order executor...")
ex = OrderExecutor()

print("2. Attente de connexion...")
time.sleep(5)

print("3. Test des fonctions...")
price = ex.get_price()
print(f"   Prix: ${price:.2f}")

cash = ex.available_cash()
print(f"   Cash: ${cash:.2f}")

has_pos = ex.has_position()
print(f"   Position: {has_pos}")

print("\n[SUCCES] Le bot est pret!")