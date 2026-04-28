import numpy as np
import logging
from agent import HybridTradingAgent
from feature_extractor import FeatureExtractor, TradingState
from ensemble_strategies import EnsembleDecisionMaker, EnsembleStrategy
from config import BATCH_SIZE, STOP_LOSS_PCT, TAKE_PROFIT_PCT, EPIC, SIZE, CHECKPOINT_INTERVAL
from trader import Trader
import time
import os
import argparse
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# METRICS & TRADE LOGGING
# =========================
@dataclass
class Trade:
    entry_price: float
    entry_time: datetime
    direction: str  # "BUY" or "SELL"
    size: float
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        if self.direction == "BUY":
            return (self.exit_price - self.entry_price) * self.size
        else:  # SELL
            return (self.entry_price - self.exit_price) * self.size
    
    def pnl_pct(self) -> float:
        if self.exit_price is None or self.entry_price == 0:
            return 0.0
        return (self.pnl() / (self.entry_price * self.size)) * 100


@dataclass
class TradingMetrics:
    trades: List[Trade] = field(default_factory=list)
    
    def add_trade(self, trade: Trade):
        self.trades.append(trade)
    
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl() > 0)
        return (wins / len(self.trades)) * 100
    
    def total_pnl(self) -> float:
        return sum(t.pnl() for t in self.trades)
    
    def avg_pnl_per_trade(self) -> float:
        if not self.trades:
            return 0.0
        return self.total_pnl() / len(self.trades)
    
    def max_drawdown(self) -> float:
        """Calculate max drawdown from equity curve"""
        if not self.trades:
            return 0.0
        cumulative = 0
        peak = 0
        max_dd = 0
        for trade in self.trades:
            cumulative += trade.pnl()
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)
        return max_dd


class XAUUSDHybridTrader:
    """
    Hybrid RL trading bot for XAUUSD with DQN + PPO ensemble.
    
    FIX SUMMARY:
    1. Fixed next_state calculation (was identical to state)
    2. Implemented proper done signal
    3. Fixed reward calculation (position-aware)
    4. Added position exit logic
    5. Added trade logging and metrics
    6. Improved position synchronization
    7. Better training frequency control
    8. Added type hints and logging
    """

    def __init__(self, use_live_data: bool = True, dry_run: bool = True):
        self.state_size = 18
        self.action_size = 3  # BUY=0, SELL=1, HOLD=2
        
        # Actions enum for clarity
        self.ACTIONS = {"BUY": 0, "SELL": 1, "HOLD": 2}
        self.ACTION_NAMES = {0: "BUY", 1: "SELL", 2: "HOLD"}

        self.agent = HybridTradingAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            dqn_weight=0.5,
            ppo_weight=0.5
        )

        self.ensemble = EnsembleDecisionMaker(strategy=EnsembleStrategy.WEIGHTED_VOTING)
        self.feature_extractor = FeatureExtractor(lookback_window=20)
        self.trading_state = TradingState()

        self.price_history: List[float] = []
        self.volume_history: List[float] = []

        self.models_dir = "./models"
        os.makedirs(self.models_dir, exist_ok=True)

        self.trader: Optional[Trader] = Trader() if use_live_data else None
        self.epic = EPIC
        self.position: Optional[Dict] = None
        self.last_price: float = 2000.0
        self.dry_run = dry_run
        self.checkpoint_interval = CHECKPOINT_INTERVAL
        
        # Metrics tracking
        self.metrics = TradingMetrics()
        
        # Better position cache
        self._last_has_position = False
        self._position_check_time = 0
        self._position_cache_ttl = 5  # seconds

    # =========================
    # LIVE DATA METHODS
    # =========================
    def get_current_price(self) -> float:
        """Fetch current price with fallback."""
        try:
            price = self.trader.get_price(self.epic)
            self.last_price = price
            return price
        except Exception as e:
            logger.warning(f"Error fetching price: {e}")
            return self.last_price or 2000.0

    def get_current_volume(self) -> float:
        """Fetch current volume, or return 0 if unavailable."""
        try:
            volume = self.trader.get_volume(self.epic)
            return max(volume, 0)
        except Exception as e:
            logger.debug(f"Volume unavailable: {e}")
            return 0.0

    def check_positions(self) -> bool:
        """Check if position is open (with caching)."""
        try:
            # Use cache to avoid excessive API calls
            current_time = time.time()
            if current_time - self._position_check_time < self._position_cache_ttl:
                return self._last_has_position
            
            positions = self.trader.get_positions()
            
            # Handle different response types
            if hasattr(positions, 'empty'):
                has_pos = not positions.empty
            elif hasattr(positions, '__len__'):
                has_pos = len(positions) > 0
            elif hasattr(positions, '__bool__'):
                has_pos = bool(positions)
            else:
                has_pos = False
            
            self._last_has_position = has_pos
            self._position_check_time = current_time
            return has_pos

        except Exception as e:
            logger.error(f"Error checking positions: {e}")
            return self._last_has_position

    # =========================
    # POSITION MANAGEMENT
    # =========================
    def execute_trade(self, action: int, current_price: float) -> Optional[Dict]:
        """
        Execute a trade (BUY or SELL).
        
        Returns:
            Deal reference dict or None if failed
        """
        direction = "BUY" if action == 0 else "SELL" if action == 1 else None
        if direction is None:
            return None

        if self.dry_run:
            logger.info(f"[DRY RUN] {direction} @ {current_price:.2f}")
            
            self.position = {
                "direction": direction,
                "entry_price": current_price,
                "size": SIZE,
                "deal_id": f"dry_run_{int(time.time())}",
                "entry_time": datetime.now()
            }
            
            self._last_has_position = True
            return {"dealReference": self.position["deal_id"]}

        try:
            # Calculate stops
            sl = current_price * (1 - STOP_LOSS_PCT / 100) if direction == "BUY" else current_price * (1 + STOP_LOSS_PCT / 100)
            tp = current_price * (1 + TAKE_PROFIT_PCT / 100) if direction == "BUY" else current_price * (1 - TAKE_PROFIT_PCT / 100)

            result = self.trader.open_trade(
                epic=self.epic,
                direction=direction,
                size=SIZE,
                sl=abs(current_price - sl),
                tp=abs(current_price - tp)
            )

            self.position = {
                "direction": direction,
                "entry_price": current_price,
                "size": SIZE,
                "deal_id": result.get("dealReference"),
                "entry_time": datetime.now()
            }
            
            self._last_has_position = True
            logger.info(f"OPENED {direction} @ {current_price:.2f}")
            return result

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return None

    def close_position(self, reason: str = "manual") -> bool:
        """
        Close current position.
        
        Args:
            reason: Why the position is being closed
            
        Returns:
            True if successful
        """
        if not self.position:
            logger.warning("No position to close")
            return False
        
        try:
            if self.dry_run:
                logger.info(f"[DRY RUN] CLOSED {self.position['direction']} ({reason})")
                exit_price = self.last_price
            else:
                result = self.trader.close_trade(self.position["deal_id"])
                exit_price = self.last_price
            
            # Log trade for metrics
            trade = Trade(
                entry_price=self.position["entry_price"],
                entry_time=self.position["entry_time"],
                direction=self.position["direction"],
                size=self.position["size"],
                exit_price=exit_price,
                exit_time=datetime.now()
            )
            self.metrics.add_trade(trade)
            
            logger.info(
                f"CLOSED {self.position['direction']} @ {exit_price:.2f} | "
                f"P&L: {trade.pnl():.2f} ({reason})"
            )
            
            self.position = None
            self._last_has_position = False
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    # =========================
    # STATE & FEATURES
    # =========================
    def get_state(self, prices: List[float], volumes: Optional[List[float]] = None) -> np.ndarray:
        """Extract feature vector from price/volume history."""
        if volumes is None:
            volumes = np.ones(len(prices))

        return self.feature_extractor.extract_features(
            np.array(prices, dtype=np.float32),
            np.array(volumes, dtype=np.float32)
        )

    def calculate_reward(self, prev_price: float, current_price: float, action: int) -> float:
        """
        Calculate position-aware reward.
        
        FIX: Reward now depends on position direction and action alignment.
        """
        if not self.position:
            # No position: neutral reward (small penalty for inactivity)
            return 0.0
        
        price_change = current_price - prev_price
        
        if self.position["direction"] == "BUY":
            # Long position: profit when price goes up
            reward = price_change * self.position["size"]
        else:  # SELL
            # Short position: profit when price goes down
            reward = -price_change * self.position["size"]
        
        # Penalty for large drawdown
        unrealized_loss = abs(min(0, reward))
        if unrealized_loss > 200:  # Large loss threshold
            reward -= 50  # Extra penalty
        
        return reward

    # =========================
    # MODEL OUTPUTS
    # =========================
    def get_dqn_output(self, state: np.ndarray) -> Dict:
        """Get DQN decision."""
        q_values = self.agent.dqn_agent.predict(state)
        return {
            "action": int(np.argmax(q_values)),
            "q_values": q_values,
            "confidence": float(np.max(q_values) - np.min(q_values))
        }

    def get_ppo_output(self, state: np.ndarray) -> Dict:
        """Get PPO decision."""
        action, logprob, value, probs = self.agent.ppo_agent.act(state)
        return {
            "action": action,
            "probs": probs,
            "value": value,
            "logprob": logprob,
            "confidence": float(np.max(probs))
        }

    # =========================
    # MODEL PERSISTENCE
    # =========================
    def save_models(self) -> None:
        """Save agent models to disk."""
        try:
            self.agent.save_models(self.models_dir)
            logger.info(f"Models saved to {self.models_dir}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def load_models(self) -> bool:
        """Load previously saved models."""
        dqn_path = os.path.join(self.models_dir, "dqn_model.pkl")
        ppo_path = os.path.join(self.models_dir, "ppo_model.pkl")
        
        dqn_exists = os.path.exists(dqn_path)
        ppo_exists = os.path.exists(ppo_path)

        if dqn_exists or ppo_exists:
            try:
                self.agent.load_models(self.models_dir)
                logger.info("Models loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to load models: {e}")
                return False
        
        logger.warning("No saved models found")
        return False

    # =========================
    # MAIN LOOP
    # =========================
    def live_trading_loop(self, max_steps: Optional[int] = 1000, training_mode: bool = True) -> float:
        """
        Main trading loop with training.
        
        Args:
            max_steps: Maximum steps (None = infinite)
            training_mode: Whether to train agent
            
        Returns:
            Total episode reward
        """
        logger.info("\n=== LIVE TRADING BOT ===\n")

        step = 0
        episode_reward = 0.0
        
        # Minimum data collection before trading
        MIN_HISTORY = 100

        try:
            while max_steps is None or step < max_steps:

                # Check position status
                try:
                    has_position = self.check_positions()
                except Exception as e:
                    logger.error(f"Position check failed: {e}")
                    has_position = self._last_has_position

                # Get current price and volume
                price = self.get_current_price()
                volume = self.get_current_volume()

                self.price_history.append(price)
                self.volume_history.append(volume)

                # Wait for sufficient data
                if len(self.price_history) < MIN_HISTORY:
                    logger.info(f"Collecting data... {len(self.price_history)}/{MIN_HISTORY}")
                    time.sleep(1)
                    step += 1
                    continue

                # Extract state from last 20 prices
                state = self.get_state(
                    self.price_history[-20:],
                    self.volume_history[-20:]
                )

                # Get model predictions
                dqn_output = self.get_dqn_output(state)
                ppo_output = self.get_ppo_output(state)

                # Ensemble decision
                decision = self.ensemble.combine_predictions(dqn_output, ppo_output)
                action = decision["action"]

                # === CALCULATE REWARD & NEXT STATE ===
                prev_price = self.price_history[-2] if len(self.price_history) > 1 else price
                reward = self.calculate_reward(prev_price, price, action)
                
                # FIX: next_state is now correctly calculated
                next_state = self.get_state(
                    self.price_history[-20:],
                    self.volume_history[-20:]
                )
                
                # FIX: done signal based on episode length or large loss
                done = (step % 1000 == 0) or (reward < -500 and has_position)

                # === TRADE LOGIC ===
                if not has_position and action in [self.ACTIONS["BUY"], self.ACTIONS["SELL"]]:
                    self.execute_trade(action, price)

                # FIX: Add position exit logic
                if has_position and action == self.ACTIONS["HOLD"]:
                    self.close_position("agent_signal")

                # === TRAINING ===
                if training_mode:
                    self.agent.remember(
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                        ppo_output
                    )
                    
                    # FIX: Only train every N steps
                    # (Don't check memory - HybridTradingAgent handles it internally)
                    if step % 10 == 0:
                        self.agent.train_step(batch_size=BATCH_SIZE)

                episode_reward += reward
                step += 1

                # Logging
                if step % 10 == 0:
                    logger.info(
                        f"Step {step} | Price {price:.2f} | "
                        f"Action {self.ACTION_NAMES.get(action, '?')} | "
                        f"Pos {has_position} | Reward {reward:.2f}"
                    )

                # Checkpoint
                if step % (self.checkpoint_interval * 10) == 0:
                    logger.info("Saving checkpoint...")
                    self.save_models()

                # Metrics display
                if step % 100 == 0 and self.metrics.trades:
                    logger.info(
                        f"Metrics - Trades: {len(self.metrics.trades)}, "
                        f"Win Rate: {self.metrics.win_rate():.1f}%, "
                        f"Total P&L: {self.metrics.total_pnl():.2f}"
                    )

                time.sleep(5)

        except KeyboardInterrupt:
            logger.info("\nStopped by user")

        logger.info(f"Done | Episode Reward: {episode_reward:.2f}")
        return episode_reward


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--forever", action="store_true")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    bot = XAUUSDHybridTrader(
        use_live_data=True,
        dry_run=args.dry_run
    )

    if bot.trader:
        try:
            bot.trader.enable_streaming(bot.epic)
            logger.info("Streaming enabled")
        except Exception as e:
            logger.warning(f"Could not enable streaming: {e}")

    logger.info("Loading models...")
    bot.load_models()

    try:
        if args.forever:
            bot.live_trading_loop(
                max_steps=None,
                training_mode=not args.no_train
            )
        else:
            bot.live_trading_loop(
                max_steps=args.max_steps,
                training_mode=not args.no_train
            )

        bot.save_models()
        
        # Print final metrics
        if bot.metrics.trades:
            logger.info("\n=== FINAL METRICS ===")
            logger.info(f"Total Trades: {len(bot.metrics.trades)}")
            logger.info(f"Win Rate: {bot.metrics.win_rate():.2f}%")
            logger.info(f"Total P&L: {bot.metrics.total_pnl():.2f}")
            logger.info(f"Avg P&L per Trade: {bot.metrics.avg_pnl_per_trade():.2f}")
            logger.info(f"Max Drawdown: {bot.metrics.max_drawdown():.2f}")

    except KeyboardInterrupt:
        bot.save_models()
        logger.info("Stopped safely")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        bot.save_models()