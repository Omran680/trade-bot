"""
core/logger.py — Logging centralisé, couleurs terminal + fichier
Fonctionne seul : python -m core.logger
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import config

RESET  = "\033[0m"
COLORS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Vert
    "WARNING":  "\033[33m",   # Jaune
    "ERROR":    "\033[31m",   # Rouge
    "CRITICAL": "\033[35m",   # Magenta
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, RESET)
        ts    = datetime.now().strftime("%H:%M:%S")
        msg   = super().format(record)
        return f"{color}[{ts}] [{record.levelname:8s}] {record.getMessage()}{RESET}"

def get_logger(name: str = "GoldScalper") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    # Terminal
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)

    # Fichier
    Path(config.LOG_FILE).parent.mkdir(exist_ok=True)
    fh = logging.FileHandler(config.LOG_FILE)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    return logger


# ── Test standalone ───────────────────────────────────────────
if __name__ == "__main__":
    log = get_logger()
    log.debug("Debug message")
    log.info("Bot démarré avec succès")
    log.warning("Spread élevé détecté")
    log.error("Erreur connexion Alpaca")
    log.critical("Stop loss global atteint")
