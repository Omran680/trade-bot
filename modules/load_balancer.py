# modules/load_balancer.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.logger import get_logger
from strategies.base import create_strategies

log = get_logger()

class LoadBalancer:
    def __init__(self):
        self.strategies = create_strategies()
        self.scores = {}
        for strategy in self.strategies:
            self.scores[strategy.name] = 0.05
        
    def record(self, result):
        """Enregistre le resultat d'un trade"""
        if result.strategy in self.scores:
            current = self.scores[result.strategy]
            adjustment = result.pnl_pct * 0.1
            self.scores[result.strategy] = max(0.01, min(1.0, current + adjustment))
            
    def best_strategy(self):
        """Retourne la meilleure strategie"""
        if not self.strategies:
            return None
            
        log.info("Load Balancer")
        for strategy in self.strategies:
            score = self.scores.get(strategy.name, 0.05)
            bar_length = int(score * 50)
            bar = "#" * bar_length
            log.info(f"  {strategy.name:<25} score={score:.3f} {bar}")
        
        best = max(self.strategies, key=lambda s: self.scores.get(s.name, 0))
        log.info(f"  Fallback (donnees insuffisantes) : {best.name}")
        return best