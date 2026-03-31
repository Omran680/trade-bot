"""
modules/risk_manager.py — Garde-fous : drawdown max, cooldown, position limits
Fonctionne seul : python -m modules.risk_manager
"""

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from core.logger import get_logger

log = get_logger()

RISK_STATE_FILE = "data/risk_state.json"
MAX_DRAWDOWN_PCT = 0.10    # Stoppe le bot si perte > 10% du capital
MAX_DAILY_TRADES = 20      # Max trades par jour
COOLDOWN_MINUTES = 3       # Pause après un trade perdant

class RiskManager:
    def __init__(self, initial_capital: float = 10000):  # Ajout du paramètre initial_capital
        self.start_capital = initial_capital  # ← AJOUT OBLIGATOIRE
        self.initial_capital = initial_capital  # Pour compatibilité
        self.daily_trades = 0
        self.max_daily_trades = 10  # Max 10 trades par jour
        self.max_daily_loss = -500  # Perte max journalière -500$
        self.daily_pnl = 0
        self.cooldown = False
        self._state = self._load()  # Chargement de l'état persistant

    # ── Checks principaux ─────────────────────────────────────
    def can_trade(self, current_value: float) -> bool:
        # Vérification que start_capital existe
        if not hasattr(self, 'start_capital') or self.start_capital is None:
            log.error("❌ start_capital non initialisé dans RiskManager")
            return False
            
        if self._max_drawdown_hit(current_value):
            log.error("🛑 MAX DRAWDOWN ATTEINT — bot en pause")
            return False
        if self._daily_limit_hit():
            log.warning("⚠️  Limite journalière atteinte")
            return False
        if self._in_cooldown():
            log.info("⏳ Cooldown actif après trade perdant")
            return False
        return True

    def record_trade(self, pnl: float):
        self._state["daily_trades"] += 1
        self._state["total_pnl"]    += pnl
        self.daily_trades = self._state["daily_trades"]
        self.daily_pnl = self._state["total_pnl"]
        
        if pnl < 0:
            self._state["last_loss_ts"] = datetime.now(timezone.utc).isoformat()
            self.cooldown = True
        self._save()

    def reset_daily(self):
        self._state["daily_trades"] = 0
        self._state["total_pnl"] = 0.0
        self._state["last_reset"] = datetime.now(timezone.utc).date().isoformat()
        self.daily_trades = 0
        self.daily_pnl = 0
        self._save()
        log.info("Compteurs journaliers réinitialisés")

    # ── Logique interne ───────────────────────────────────────
    def _max_drawdown_hit(self, current: float) -> bool:
        """Vérifie si le drawdown maximum est atteint"""
        if self.start_capital <= 0:
            return False
        loss_pct = (self.start_capital - current) / self.start_capital
        return loss_pct >= MAX_DRAWDOWN_PCT

    def _daily_limit_hit(self) -> bool:
        today = datetime.now(timezone.utc).date().isoformat()
        if self._state.get("last_reset") != today:
            self.reset_daily()
        return self._state["daily_trades"] >= self.max_daily_trades

    def _in_cooldown(self) -> bool:
        ts = self._state.get("last_loss_ts")
        if not ts:
            return False
        last = datetime.fromisoformat(ts)
        return datetime.now(timezone.utc) - last < timedelta(minutes=COOLDOWN_MINUTES)

    # ── Persistance ───────────────────────────────────────────
    def _load(self) -> dict:
        p = Path(RISK_STATE_FILE)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                # Migration des anciennes données
                if "total_pnl" not in data:
                    data["total_pnl"] = 0.0
                return data
            except Exception as e:
                log.warning(f"Erreur chargement état risque: {e}")
        return {
            "daily_trades": 0, 
            "total_pnl": 0.0,
            "last_loss_ts": None, 
            "last_reset": None
        }

    def _save(self):
        try:
            Path(RISK_STATE_FILE).parent.mkdir(exist_ok=True)
            Path(RISK_STATE_FILE).write_text(json.dumps(self._state, indent=2))
        except Exception as e:
            log.error(f"Erreur sauvegarde état risque: {e}")

    def status(self) -> str:
        s = self._state
        return (f"Trades aujourd'hui : {s['daily_trades']}/{self.max_daily_trades} | "
                f"P&L total : ${s['total_pnl']:+.2f} | "
                f"Cooldown : {self._in_cooldown()} | "
                f"Capital : ${self.start_capital:.2f}")
    
    def update_capital(self, new_capital: float):
        """Met à jour le capital courant"""
        self.start_capital = new_capital
        self.initial_capital = new_capital
        
    def get_drawdown(self) -> float:
        """Retourne le drawdown actuel en pourcentage"""
        if self.start_capital <= 0:
            return 0.0
        current = self._state.get("total_pnl", 0)
        return abs(current) / self.start_capital


# ── Test standalone ───────────────────────────────────────────
if __name__ == "__main__":
    log.info("=== Test RiskManager ===")
    rm = RiskManager(initial_capital=10000)  # ← Passer le capital initial
    log.info(rm.status())
    log.info(f"Peut trader (capital=9500$) : {rm.can_trade(9500)}")  # Drawdown 5%
    log.info(f"Peut trader (capital=8900$) : {rm.can_trade(8900)}")  # Drawdown 11% > max
    rm.record_trade(-2.5)
    log.info(f"Après perte → {rm.status()}")