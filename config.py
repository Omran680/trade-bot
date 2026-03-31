"""
config.py — Configuration centrale du bot Gold Scalper
Modifie uniquement ce fichier pour personnaliser le bot.
"""

# ── Instrument ────────────────────────────────────────────────
SYMBOL        = "XAU/USD"  # Or (Gold) — format Massive Broker
TIMEFRAME     = "M1"       # M1 scalping
BAR_LIMIT     = 100        # Bougies chargées par cycle

# ── Capital & risque ──────────────────────────────────────────
CAPITAL       = 200.0      # Capital de départ ($)
RISK_PCT      = 0.02       # 2% risqué par trade
MAX_POSITIONS = 1          # 1 position à la fois

# ── Load Balancer ─────────────────────────────────────────────
LB_EVAL_WINDOW  = 20       # Derniers trades analysés par stratégie
LB_MIN_TRADES   = 5        # Min trades avant notation
LB_UPDATE_EVERY = 10       # Réévalue toutes les N minutes

# ── Sessions de trading (UTC) ─────────────────────────────────
LONDON_OPEN   = 8          # 08:00 UTC
NY_OPEN       = 13         # 13:00 UTC
NY_CLOSE      = 21         # 21:00 UTC

# ── Chemins ───────────────────────────────────────────────────
LOG_FILE      = "logs/bot.log"
TRADES_FILE   = "data/trades.json"
SCORES_FILE   = "data/strategy_scores.json"

# config.py
SYMBOL = "XAU/USD"
CAPITAL = 10000
RISK_PCT = 0.02

DEMO_MODE = True  


# Session de trading (UTC)
SESSION_START = 1  # 1h UTC
SESSION_END = 23   # 23h UTC

# IBKR Connection
IB_HOST = "127.0.0.1"
IB_PORT = 4002
IBKR_PORT = 7497