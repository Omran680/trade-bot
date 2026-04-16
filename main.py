import numpy as np
from agent import HybridTradingAgent, DQNAgent, PPOAgent
from model import softmax
from feature_extractor import FeatureExtractor, TradingState
from ensemble_strategies import EnsembleDecisionMaker, EnsembleStrategy
from config import BATCH_SIZE, STOP_LOSS_PCT, TAKE_PROFIT_PCT, EPIC, SIZE, CHECKPOINT_INTERVAL
from trader import Trader
import time
import os


class XAUUSDHybridTrader:
    """
    PPO + DQN Hybrid Trading System with IG Live Data
    """

    def __init__(self, use_live_data=True, dry_run=True):

        self.state_size = 18
        self.action_size = 3  # BUY, SELL, HOLD

        self.agent = HybridTradingAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            dqn_weight=0.5,
            ppo_weight=0.5
        )

        self.ensemble = EnsembleDecisionMaker(strategy=EnsembleStrategy.WEIGHTED_VOTING)

        self.feature_extractor = FeatureExtractor(lookback_window=20)
        self.trading_state = TradingState()

        self.price_history = []
        self.volume_history = []

        self.total_reward = 0
        self.episode_rewards = []
        self.models_dir = "./models"

        # Live trading setup
        self.trader = Trader() if use_live_data else None
        self.epic = EPIC
        self.position = None  # Current position info
        self.last_price = None
        self.dry_run = dry_run  # If True, no real trades
        self.checkpoint_interval = CHECKPOINT_INTERVAL

    # =========================
    # LIVE DATA METHODS
    # =========================
    def get_current_price(self):
        """Fetch current XAU/USD price from IG"""
        try:
            price = self.trader.get_price(self.epic)
            self.last_price = price
            return price
        except Exception as e:
            print(f"Error fetching price: {e}")
            return self.last_price or 2000.0  # Fallback

    def check_positions(self):
        """Check current positions"""
        try:
            positions = self.trader.get_positions()
            # Handle pandas DataFrame or similar objects
            if hasattr(positions, 'empty'):
                return not positions.empty  # DataFrame.empty
            elif hasattr(positions, '__len__'):
                return len(positions) > 0
            elif hasattr(positions, '__bool__'):
                return bool(positions)
            else:
                return False
        except Exception as e:
            print(f"Error checking positions: {e}")
            return False

    def get_account_info(self):
        """Get account information"""
        try:
            accounts = self.trader.get_account_balance()
            # Check if we got valid data
            if hasattr(accounts, 'empty') and not accounts.empty:
                return accounts
            elif hasattr(accounts, '__len__') and len(accounts) > 0:
                return accounts
            else:
                return None
        except Exception as e:
            print(f"Error getting account info: {e}")
            return None

    def execute_trade(self, action, current_price):
        """Execute trade on IG (or simulate in dry run)"""
        if action == 0:  # BUY
            direction = "BUY"
        elif action == 1:  # SELL
            direction = "SELL"
        else:  # HOLD
            return None

        if self.dry_run:
            print(f"[DRY RUN] Would open {direction} position at {current_price}")
            self.position = {
                'direction': direction,
                'entry_price': current_price,
                'size': SIZE,
                'deal_id': 'dry_run'
            }
            return {'dealReference': 'dry_run'}

        try:
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
                'direction': direction,
                'entry_price': current_price,
                'size': SIZE,
                'deal_id': result.get('dealReference')
            }

            print(f"Opened {direction} position at {current_price}")
            return result

        except Exception as e:
            print(f"Trade execution failed: {e}")
            return None

    # =========================
    # STATE (SAFE)
    # =========================
    def get_state(self, prices, volumes=None):

        if volumes is None:
            volumes = np.ones(len(prices))

        prices = np.array(prices, dtype=np.float32)
        volumes = np.array(volumes, dtype=np.float32)

        return self.feature_extractor.extract_features(prices, volumes)

    # =========================
    # DQN
    # =========================
    def get_dqn_output(self, state):
        q_values = self.agent.dqn_agent.predict(state)
        action = int(np.argmax(q_values))
        confidence = float(np.max(q_values) - np.min(q_values))
        return {
            "action": action,
            "q_values": q_values,
            "confidence": confidence
        }

    def get_ppo_output(self, state):
        action, logprob, value, probs = self.agent.ppo_agent.act(state)
        return {
            "action": action,
            "probs": probs,
            "value": value,
            "confidence": float(np.max(probs)),
            "logprob": logprob
        }

    # =========================
    # MODEL SAVE/LOAD
    # =========================
    def save_models(self):
        """Save DQN and PPO models to disk"""
        self.agent.save_models(self.models_dir)

    def load_models(self):
        """Load DQN and PPO models from disk"""
        if os.path.exists(os.path.join(self.models_dir, "dqn_model.pkl")) or os.path.exists(os.path.join(self.models_dir, "ppo_model.pkl")):
            self.agent.load_models(self.models_dir)
            return True
        else:
            print(f"No saved models found in {self.models_dir}")
            return False

    # =========================
    # LIVE TRADING LOOP
    # =========================
    def live_trading_loop(self, max_steps=1000, training_mode=True):
        """Live trading loop with real IG data"""

        print("\n=== LIVE HYBRID PPO + DQN TRADING ===\n")
        print(f"Trading {self.epic} on IG DEMO account")

        # Check account info
        account_info = self.get_account_info()
        if account_info is not None:
            print(f"Account info retrieved successfully")

        print("Press Ctrl+C to stop\n")

        step = 0
        episode_reward = 0

        try:
            while step < max_steps:
                # Check current positions
                has_position = self.check_positions()

                # Fetch current price
                current_price = self.get_current_price()
                volume = 1000.0  # Default volume for features

                # Update price history
                self.price_history.append(current_price)
                self.volume_history.append(volume)

                # Keep only recent history
                if len(self.price_history) > 100:
                    self.price_history = self.price_history[-100:]
                    self.volume_history = self.volume_history[-100:]

                # Need enough history for features
                if len(self.price_history) < 20:
                    print(f"Collecting data... {len(self.price_history)}/20")
                    time.sleep(1)
                    continue

                # Get state features
                state = self.get_state(self.price_history, self.volume_history)

                # Get agent predictions
                dqn_output = self.get_dqn_output(state)
                ppo_output = self.get_ppo_output(state)

                # Ensemble decision
                decision = self.ensemble.combine_predictions(dqn_output, ppo_output)
                action = decision["action"]

                # Calculate reward (based on price movement)
                if len(self.price_history) > 1:
                    price_change = current_price - self.price_history[-2]
                    reward = price_change * (1 if action == 0 else -1 if action == 1 else 0)
                else:
                    reward = 0

                # Execute trade only if no position and action is BUY/SELL
                if not has_position and action in [0, 1]:
                    self.execute_trade(action, current_price)

                # Prepare next state
                next_price = self.get_current_price()  # Fetch again for next state
                next_volume = 1000.0
                next_state = self.get_state(
                    self.price_history + [next_price],
                    self.volume_history + [next_volume]
                )

                done = False  # Continuous learning

                # Train agents
                if training_mode:
                    self.agent.remember(
                        state, action, reward, next_state, done, ppo_output
                    )
                    self.agent.train_step(batch_size=BATCH_SIZE)

                # Update metrics
                episode_reward += reward
                step += 1

                # Print status
                if step % 10 == 0:
                    print(f"Step {step} | Price: {current_price:.2f} | Action: {['BUY', 'SELL', 'HOLD'][action]} | Has Position: {has_position} | Reward: {reward:.4f}")

                # Checkpoint saving
                if step % (self.checkpoint_interval * 10) == 0 and step > 0:
                    print(f"\n💾 Checkpoint at step {step}...")
                    self.save_models()
                    print()

                # Wait before next iteration
                time.sleep(2)  # Respect API rate limits

        except KeyboardInterrupt:
            print("\nStopping live trading...")

        print(f"\nSession complete. Total steps: {step}, Total reward: {episode_reward:.2f}")

        return episode_reward


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("Initializing IG Live Trading Bot...")
    trader = XAUUSDHybridTrader(use_live_data=True, dry_run=True)  # Set dry_run=False for real trading

    # Try to load pre-trained models
    print("Checking for saved models...")
    if trader.load_models():
        print("✓ Loaded pre-trained models. Continuing training...\n")
    else:
        print("📌 No saved models found. Starting fresh training...\n")

    print("Starting live trading with real XAU/USD data...")
    print("⚠️  DRY RUN MODE: No real trades will be executed")
    print("Change dry_run=False in main.py for real trading")
    print("Press Ctrl+C to stop at any time\n")

    try:
        trader.live_trading_loop(max_steps=50, training_mode=True)
        # Save models after training
        print("\n💾 Saving final models...")
        trader.save_models()
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
        print("💾 Saving models before exit...")
        trader.save_models()
    except Exception as e:
        print(f"Error during trading: {e}")
        print("💾 Saving models due to error...")
        trader.save_models()