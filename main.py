# main.py
import time
from core.logger import get_logger
from modules.order_executor import OrderExecutor
from strategies.strategies import ALL_STRATEGIES

log = get_logger()
LOOP_SECONDS = 10

def run():
    log.info("=== GOLD SCALPER BOT XAU/USD ===")
    ex = OrderExecutor()

    while True:
        try:
            price = ex.get_price()
            strategy = ALL_STRATEGIES[0]  # pour test, EMACrossover
            sig = strategy.signal(None)
            log.info(f"Signal: {sig} | Prix: {price}")
            
            if sig == "BUY" and not ex.has_position():
                ex.enter("buy", price, atr=5)
            elif sig == "SELL" and ex.has_position():
                ex.close()

            time.sleep(LOOP_SECONDS)

        except KeyboardInterrupt:
            log.info("Arret du bot")
            ex.disconnect()
            break

if __name__ == "__main__":
    run()