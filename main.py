# main.py — Gold Scalper Bot IBKR (XAU/USD)
import time
import signal
import sys
from datetime import datetime, timezone

import config
from core.logger import get_logger
from core.data_fetcher import fetch_bars, get_price, market_is_open
from core.indicators import add_all
from modules.load_balancer import LoadBalancer
from modules.order_executor import OrderExecutor
from modules.session_filter import is_tradeable, session_info
from modules.risk_manager import RiskManager
from strategies.base import TradeResult

log = get_logger()
LOOP_SECONDS = 60

def handle_exit(sig, frame):
    log.info("Arret du bot (Ctrl+C)")
    sys.exit(0)
    
signal.signal(signal.SIGINT, handle_exit)

def run():
    log.info("=" * 55)
    log.info("  GOLD SCALPER BOT - INTERACTIVE BROKERS (IBKR)")
    log.info(f"  Symbole  : {config.SYMBOL}")
    log.info(f"  Capital  : ${config.CAPITAL}")
    log.info(f"  Risque   : {config.RISK_PCT*100:.0f}% / trade")
    log.info("=" * 55)

    lb = LoadBalancer()
    
    # Gérer l'initialisation de l'executor avec les données différées
    try:
        ex = OrderExecutor()
        log.info("OrderExecutor initialise avec succes")
    except Exception as e:
        log.error(f"Erreur initialisation OrderExecutor: {e}")
        return
    
    rm = RiskManager()

    cycle = 0

    while True:
        try:
            cycle += 1
            now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            log.info(f"\n-- Cycle #{cycle}  {now} --")

            log.info(session_info())
            if not is_tradeable():
                log.info("Hors session - attente 5 min...")
                time.sleep(300)
                continue

            if not market_is_open():
                log.info("Marche ferme (weekend) - attente...")
                time.sleep(300)
                continue

            # Récupérer les données
            bars = fetch_bars()
            if bars.empty:
                log.warning("Pas de donnees - retry dans 30s")
                time.sleep(30)
                continue

            # Ajouter les indicateurs
            try:
                df = add_all(bars)
            except Exception as e:
                log.error(f"Erreur calcul indicateurs: {e}")
                time.sleep(30)
                continue
                
            # Récupérer le prix (avec données différées)
            try:
                price = ex.get_price()
                if price <= 0:
                    price = 2650.0
            except Exception as e:
                log.error(f"Erreur get_price: {e}")
                price = 2650.0
                
            # Calculer ATR
            try:
                atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not df.empty else price * 0.002
            except:
                atr = price * 0.002

            # Récupérer la position
            try:
                pos = ex.get_position_info()
                if pos:
                    log.info(f"Position ouverte : {pos['type']} {pos['units']} oz | "
                             f"Entree: ${pos['open_price']:.2f}")
            except Exception as e:
                log.error(f"Erreur get_position_info: {e}")
                pos = None

            # Vérifier le cash disponible
            try:
                cash = ex.available_cash()
                if cash <= 0:
                    cash = config.CAPITAL
            except Exception as e:
                log.error(f"Erreur available_cash: {e}")
                cash = config.CAPITAL
                
            if not rm.can_trade(cash):
                log.warning(f"Trading bloque. {rm.status()}")
                time.sleep(LOOP_SECONDS)
                continue

            # Obtenir le signal
            strategy = lb.best_strategy()
            try:
                signal_ = strategy.signal(df)
            except Exception as e:
                log.error(f"Erreur signal: {e}")
                signal_ = "HOLD"
                
            log.info(f"Strategie : {strategy.name}  |  Signal : {signal_}  |  Prix : ${price:.2f}")

            # Vérifier la position
            try:
                has_pos = ex.has_position()
            except Exception as e:
                log.error(f"Erreur has_position: {e}")
                has_pos = False

            # Exécuter les trades
            if signal_ == "BUY" and not has_pos:
                try:
                    ok = ex.enter("buy", price, atr)
                    if ok:
                        rm.record_trade(0)
                except Exception as e:
                    log.error(f"Erreur execution BUY: {e}")

            elif signal_ == "SELL" and not has_pos:
                try:
                    ok = ex.enter("sell", price, atr)
                    if ok:
                        rm.record_trade(0)
                except Exception as e:
                    log.error(f"Erreur execution SELL: {e}")

            elif (signal_ in ("BUY", "SELL")) and has_pos and pos:
                current_type = pos["type"]
                if (signal_ == "BUY" and current_type == "SELL") or \
                   (signal_ == "SELL" and current_type == "BUY"):
                    log.info("Signal inverse - fermeture + re-entree")
                    try:
                        ex.close()
                        time.sleep(2)
                        side = "buy" if signal_ == "BUY" else "sell"
                        ex.enter(side, ex.get_price(), atr)
                    except Exception as e:
                        log.error(f"Erreur fermeture/re-entree: {e}")

            elif signal_ == "HOLD":
                log.info("En attente d'un signal...")

            # Historique
            try:
                history = ex.get_trade_history(count=3)
                if history:
                    last = history[0]
                    log.info(f"Dernier trade clos : {last['type']}")
            except Exception as e:
                log.error(f"Erreur historique: {e}")

            log.info(f"Marge dispo : ${cash:.2f}  |  {rm.status()}")
            time.sleep(LOOP_SECONDS)
            
        except KeyboardInterrupt:
            log.info("Arret demande")
            break
        except Exception as e:
            log.error(f"Erreur boucle principale: {e}")
            time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    run()