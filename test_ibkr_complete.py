#!/usr/bin/env python3
"""
test_trading_aggressive.py - Forcer les trades pour tester
Lance des ordres d'achat/vente pour valider l'exécution
"""

import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.logger import get_logger
from modules.order_executor import OrderExecutor

log = get_logger()

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_aggressive_trading():
    """Lance des trades agressifs pour tester"""
    
    print_header("🔥 TEST TRADING AGRESSIF - FORCER LES TRADES")
    
    try:
        # Initialiser executor
        print("→ Initialisation OrderExecutor...")
        ex = OrderExecutor()
        print("✅ OrderExecutor prêt\n")
        
        # Récupérer infos
        print("→ Récupération des infos...")
        price = ex.get_price()
        cash = ex.available_cash()
        has_pos = ex.has_position()
        
        print(f"  Prix actuel: ${price:.2f}")
        print(f"  Cash disponible: ${cash:.2f}")
        print(f"  Position ouverte: {'OUI' if has_pos else 'NON'}\n")
        
        if price <= 0:
            print("❌ Prix invalide! Impossible de trader.")
            return False
        
        if cash < 500:
            print("❌ Cash insuffisant! Minimum 500 USD requis.")
            return False
        
        # ════════════════════════════════════════════════
        # TRADE 1: BUY (Achat)
        # ════════════════════════════════════════════════
        
        print_header("📈 TRADE 1: ACHAT (BUY)")
        
        print(f"Conditions:")
        print(f"  - Prix: ${price:.2f}")
        print(f"  - Cash: ${cash:.2f}")
        print(f"  - ATR: {price * 0.002:.2f}")
        
        # Calculer taille petite pour test
        atr = price * 0.002
        risk_amount = 100  # Risquer 100 USD
        size = int(risk_amount / (atr * 1.5))
        size = max(1, min(size, 5))  # Entre 1 et 5 oz
        
        print(f"\n→ Placement ordre BUY {size} oz...")
        
        buy_ok = ex.enter("buy", price, atr)
        
        if buy_ok:
            print(f"✅ ACHAT RÉUSSI!")
            print(f"   Ordre BUY {size} oz placé à ${price:.2f}")
            
            # Vérifier position
            time.sleep(2)
            pos_info = ex.get_position_info()
            if pos_info:
                print(f"   Position confirmée: {pos_info['type']} {pos_info['units']} oz")
                print(f"   Entrée: ${pos_info['open_price']:.2f}")
            
            # Attendre un peu avant fermeture
            print(f"\n→ Attente 10 secondes avant fermeture...")
            for i in range(10):
                print(f"   {10-i}s...", end="\r")
                time.sleep(1)
            print()
            
            # ════════════════════════════════════════════════
            # TRADE 2: CLOSE (Fermeture)
            # ════════════════════════════════════════════════
            
            print_header("📉 TRADE 2: FERMETURE (CLOSE)")
            
            new_price = ex.get_price()
            print(f"Prix actuel: ${new_price:.2f}")
            
            print(f"\n→ Fermeture de la position...")
            close_ok = ex.close()
            
            if close_ok:
                print(f"✅ FERMETURE RÉUSSIE!")
                
                # Vérifier qu'il n'y a plus de position
                time.sleep(2)
                has_pos_after = ex.has_position()
                if not has_pos_after:
                    print(f"   Position bien fermée ✅")
                    
                    # Calculer P&L
                    pnl = (new_price - price) * size
                    pnl_pct = (pnl / (price * size)) * 100
                    
                    print(f"\n📊 RÉSULTAT DU TRADE:")
                    print(f"   Entrée:  ${price:.2f}")
                    print(f"   Sortie:  ${new_price:.2f}")
                    print(f"   Variation: ${new_price - price:.2f}")
                    print(f"   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                else:
                    print(f"   ⚠️ Position encore ouverte après fermeture")
                
                return True
            else:
                print(f"❌ Erreur fermeture")
                return False
        else:
            print(f"❌ ACHAT ÉCHOUÉ")
            return False
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_trades():
    """Lancer plusieurs trades en succession"""
    
    print_header("🔄 MODE: TESTS MULTIPLES")
    
    print("Ce mode lance 3 cycles de buy/sell pour valider")
    print("à chaque cycle:\n")
    
    try:
        ex = OrderExecutor()
        
        for cycle in range(1, 4):
            print_header(f"🔄 CYCLE {cycle}/3")
            
            price = ex.get_price()
            print(f"Prix: ${price:.2f}")
            
            # Buy
            print(f"\n→ Buy {cycle}...")
            ex.enter("buy", price, price * 0.002)
            time.sleep(3)
            
            # Close
            print(f"→ Close {cycle}...")
            ex.close()
            time.sleep(3)
            
            print(f"✅ Cycle {cycle} complété")
            
            if cycle < 3:
                print(f"\n⏳ Attente 5s avant cycle suivant...")
                time.sleep(5)
        
        print_header("✅ TOUS LES CYCLES COMPLÉTÉS")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Menu principal"""
    
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║  🔥 TEST TRADING AGRESSIF - FORCER LES TRADES 🔥  ║")
    print("║" + " "*58 + "║")
    print("║  Mode: Lancer des ordres BUY/SELL pour tester       ║")
    print("╚" + "="*58 + "╝")
    
    print("\nOptions:")
    print("  1 = Test unique (1 BUY + 1 CLOSE)")
    print("  2 = Tests multiples (3 cycles)")
    print("  3 = Mode manuel (entrer commandes)")
    print("  Q = Quitter")
    
    choice = input("\nChoix (1-3/Q): ").strip().upper()
    
    if choice == "1":
        success = test_aggressive_trading()
        if success:
            print("\n✅ TEST RÉUSSI!")
        else:
            print("\n❌ TEST ÉCHOUÉ")
        return 0 if success else 1
    
    elif choice == "2":
        success = test_multiple_trades()
        return 0 if success else 1
    
    elif choice == "3":
        print_header("MODE MANUEL")
        try:
            ex = OrderExecutor()
            
            while True:
                cmd = input("\nCommande (buy/sell/close/price/pos/quit): ").strip().lower()
                
                if cmd == "quit":
                    break
                elif cmd == "price":
                    price = ex.get_price()
                    print(f"Prix: ${price:.2f}")
                elif cmd == "pos":
                    pos = ex.get_position_info()
                    if pos:
                        print(f"Position: {pos['type']} {pos['units']} oz @ ${pos['open_price']:.2f}")
                    else:
                        print("Pas de position")
                elif cmd in ["buy", "sell"]:
                    price = ex.get_price()
                    size_str = input(f"Taille en oz (défaut 1): ").strip()
                    size = int(size_str) if size_str else 1
                    
                    print(f"→ {cmd.upper()} {size} oz...")
                    ex.enter(cmd, price, price * 0.002)
                    time.sleep(2)
                    print("✅ Ordre placé")
                elif cmd == "close":
                    print(f"→ Fermeture...")
                    ex.close()
                    time.sleep(2)
                    print("✅ Position fermée")
                else:
                    print("Commande inconnue")
        
        except KeyboardInterrupt:
            print("\n⚠️ Arrêt")
        except Exception as e:
            print(f"❌ Erreur: {e}")
        
        return 0
    
    elif choice == "Q":
        print("Au revoir!")
        return 0
    
    else:
        print("❌ Choix invalide")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)