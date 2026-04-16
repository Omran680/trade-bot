# 🚀 Quick Commands - Hybrid PPO+DQN Trading System

## Installation & Setup

### 1. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import tensorflow; print(f'TensorFlow {tensorflow.__version__}')"
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
```

---

## Training & Backtesting

### Quick Training (5 min)
```bash
python -c "
from main import XAUUSDHybridTrader
trader = XAUUSDHybridTrader()
prices, volumes = trader.generate_synthetic_data(2000)
trader.trading_loop(prices, volumes, episodes=10)
trader.save_models()
"
```

### Full Training (30 min)
```bash
python -c "
from main import XAUUSDHybridTrader
trader = XAUUSDHybridTrader()
prices, volumes = trader.generate_synthetic_data(5000)
trader.trading_loop(prices, volumes, episodes=50)
trader.save_models()
"
```

### Run Backtesting
```bash
python -c "
from main import XAUUSDHybridTrader
trader = XAUUSDHybridTrader()
trader.load_models()
prices, volumes = trader.generate_synthetic_data(1000)
trader.backtest(prices, volumes)
"
```

---

## Examples & Testing

### Run All Examples
```bash
python examples.py
```

### Run Specific Example
```bash
python -c "from examples import example_ensemble_decision_analysis; example_ensemble_decision_analysis()"
```

### Test Individual Components
```bash
# Test Feature Extraction
python -c "
from feature_extractor import FeatureExtractor
extractor = FeatureExtractor()
prices = [2000 + i for i in range(20)]
features = extractor.extract_features(prices)
print(f'Features shape: {features.shape}')
"

# Test Risk Manager
python -c "
from risk_manager import PortfolioRiskManager
rm = PortfolioRiskManager(initial_capital=10000)
size = rm.calculate_position_size(entry_price=2050, stop_loss_price=2045)
print(f'Position size: {size:.3f} lots')
"

# Test Ensemble
python -c "
from ensemble_strategies import EnsembleStrategy, EnsembleDecisionMaker
dm = EnsembleDecisionMaker()
dqn_out = {'action': 0, 'q_values': [1.0, 0.5, 0.3], 'confidence': 0.8}
ppo_out = {'action': 0, 'probs': [0.7, 0.2, 0.1], 'value': 0.5, 'confidence': 0.7}
decision = dm.combine_predictions(dqn_out, ppo_out)
print(f'Decision: {decision}')
"
```

---

## Configuration & Customization

### Change Ensemble Strategy
```bash
python -c "
import config
from main import XAUUSDHybridTrader
from ensemble_strategies import EnsembleStrategy

config.ENSEMBLE_STRATEGY = 'averaging'
trader = XAUUSDHybridTrader()
# ... continue
"
```

### Adjust Hyperparameters
```bash
python -c "
import config

# DQN settings
config.EPSILON_DECAY = 0.99  # Faster decay
config.TARGET_UPDATE_FREQ = 500  # Update more often

# PPO settings
config.PPO_EPOCHS = 1  # Faster training
config.PPO_CLIP_RATIO = 0.1  # More conservative

# Risk settings
config.STOP_LOSS_PCT = 1.0  # Tighter stops
config.TAKE_PROFIT_PCT = 5.0  # Higher targets

# Training
config.BATCH_SIZE = 64  # Larger batches

from main import XAUUSDHybridTrader
trader = XAUUSDHybridTrader()
# ... train with new config
"
```

### Live Trading Setup
```bash
# 1. Create .env file
cat > .env << EOF
IG_USERNAME=your_username
IG_PASSWORD=your_password
IG_API_KEY=your_api_key
EOF

# 2. Run live trader
python -c "
from main import XAUUSDHybridTrader
trader = XAUUSDHybridTrader(use_live_data=True)
trader.load_models()
# ... live trading loop
"
```

---

## Performance Analysis

### Check Training Progress
```bash
python -c "
from main import XAUUSDHybridTrader
import numpy as np

trader = XAUUSDHybridTrader()
prices, volumes = trader.generate_synthetic_data(2000)
trader.trading_loop(prices, volumes, episodes=5)

# Print metrics
print(f'Total Reward: {trader.total_reward:.2f}')
print(f'Avg Reward: {np.mean(trader.episode_rewards):.2f}')
print(f'Best Episode: {np.max(trader.episode_rewards):.2f}')
print(f'Worst Episode: {np.min(trader.episode_rewards):.2f}')
"
```

### Ensemble Performance
```bash
python -c "
from main import XAUUSDHybridTrader

trader = XAUUSDHybridTrader()
prices, volumes = trader.generate_synthetic_data(2000)
trader.trading_loop(prices, volumes, episodes=10)

metrics = trader.ensemble.get_performance_metrics()
print(f'DQN Weight: {metrics[\"current_dqn_weight\"]:.3f}')
print(f'PPO Weight: {metrics[\"current_ppo_weight\"]:.3f}')
print(f'Avg DQN Reward: {metrics[\"avg_dqn_reward\"]:.2f}')
print(f'Avg PPO Reward: {metrics[\"avg_ppo_reward\"]:.2f}')
"
```

### Risk Manager Stats
```bash
python -c "
from risk_manager import TradingSession

session = TradingSession(capital=10000)
session.open_position(2050, 2045, 2060, 'LONG')
session.close_position(0, 2055)

summary = session.get_session_summary()
print(f'Return: {summary[\"return_pct\"]:.2f}%')
print(f'Win Rate: {summary[\"win_rate\"]:.1%}')
print(f'Sharpe: {summary[\"sharpe_ratio\"]:.2f}')
print(f'Max DD: {summary[\"max_drawdown\"]:.2f}%')
"
```

---

## Data & Models

### Generate Data
```bash
python -c "
from main import XAUUSDHybridTrader
trader = XAUUSDHybridTrader()
prices, volumes = trader.generate_synthetic_data(num_steps=5000)
print(f'Generated {len(prices)} price points')
print(f'Price range: {min(prices):.2f} - {max(prices):.2f}')
"
```

### Save Models
```bash
python -c "
from main import XAUUSDHybridTrader
trader = XAUUSDHybridTrader()
prices, volumes = trader.generate_synthetic_data(2000)
trader.trading_loop(prices, volumes, episodes=5)
trader.save_models()
print('Models saved to ./models/')
"
```

### Load Models
```bash
python -c "
from main import XAUUSDHybridTrader
trader = XAUUSDHybridTrader()
trader.load_models()
print('Models loaded successfully')
"
```

---

## Debugging & Troubleshooting

### Check TensorFlow
```bash
python -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
print(f'GPU Available: {tf.config.list_physical_devices(\"GPU\")}')
print(f'CPUs: {tf.config.list_physical_devices(\"CPU\")}')
"
```

### Validate Installation
```bash
python -c "
import sys
packages = ['tensorflow', 'numpy', 'pandas', 'scipy', 'sklearn']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg} NOT INSTALLED')
"
```

### Test DQN Agent
```bash
python -c "
from agent import DQNAgent
agent = DQNAgent(state_size=18, action_size=3)
print('✓ DQN Agent created')
print(f'  Model: {agent.model}')
print(f'  Epsilon: {agent.epsilon}')
"
```

### Test PPO Agent
```bash
python -c "
from agent import PPOAgent
agent = PPOAgent(state_size=18, action_size=3)
print('✓ PPO Agent created')
print(f'  Model: {agent.model}')
"
```

### Test Hybrid Network
```bash
python -c "
from model import HybridNetwork
import tensorflow as tf
import numpy as np

net = HybridNetwork(state_size=18, action_size=3)
state = tf.constant(np.random.randn(1, 18), dtype=tf.float32)
q_vals, logits, value = net(state)
print(f'✓ Hybrid Network working')
print(f'  Q-values shape: {q_vals.shape}')
print(f'  Logits shape: {logits.shape}')
print(f'  Value shape: {value.shape}')
"
```

---

## Directory & File Management

### Create Directories
```bash
mkdir -p models logs data
```

### List Files
```bash
# Windows
dir

# Linux/Mac
ls -la

# Python
python -c "
import os
for item in os.listdir('.'):
    print(item)
"
```

### Clean Cache
```bash
# Remove Python cache
python -c "
import shutil
shutil.rmtree('__pycache__', ignore_errors=True)
print('Cache cleaned')
"

# Windows only
if exist __pycache__ rmdir /s __pycache__

# Linux/Mac only
rm -rf __pycache__
```

### Check Model Files
```bash
python -c "
import os
if os.path.exists('models'):
    files = os.listdir('models')
    print(f'Models found: {len(files)}')
    for f in files:
        print(f'  - {f}')
else:
    print('No models directory yet')
"
```

---

## Development Commands

### Format Code (PEP8)
```bash
pip install autopep8
autopep8 --in-place --aggressive --aggressive *.py
```

### Check Code Quality
```bash
pip install pylint
pylint *.py --disable=all --enable=C,E
```

### Create Requirements
```bash
pip freeze > requirements.txt
```

### Virtual Env Cleanup
```bash
# Deactivate
deactivate

# Remove venv
# Windows
rmdir /s venv

# Linux/Mac
rm -rf venv
```

---

## Useful Python One-Liners

### Check Numpy Array Shape
```python
import numpy as np
arr = np.random.randn(32, 18)
print(f"Shape: {arr.shape}, Size: {arr.size}, Dtype: {arr.dtype}")
```

### Time Function Execution
```python
import time
start = time.time()
# ... code ...
elapsed = time.time() - start
print(f"Elapsed: {elapsed:.3f}s")
```

### Memory Usage
```python
import sys
obj = some_object()
print(f"Size: {sys.getsizeof(obj)} bytes")
```

### List Tensorflow Operations
```python
import tensorflow as tf
print(tf.executing_eagerly())
print(tf.config.list_physical_devices())
```

---

## Common Workflows

### Workflow 1: First Run (30 min)
```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Quick test
python examples.py

# 3. Train
python -c "
from main import XAUUSDHybridTrader
trader = XAUUSDHybridTrader()
prices, volumes = trader.generate_synthetic_data(2000)
trader.trading_loop(prices, volumes, episodes=20)
trader.save_models()
"

# 4. Backtest
python -c "
from main import XAUUSDHybridTrader
trader = XAUUSDHybridTrader()
trader.load_models()
prices, volumes = trader.generate_synthetic_data(1000)
trader.backtest(prices, volumes)
"
```

### Workflow 2: Optimization (2-3 hours)
```bash
# 1. Load and test different strategies
python -c "
from ensemble_strategies import EnsembleStrategy
from main import XAUUSDHybridTrader

for strategy in [EnsembleStrategy.VOTING, EnsembleStrategy.WEIGHTED_VOTING]:
    trader = XAUUSDHybridTrader()
    trader.ensemble.strategy = strategy
    prices, volumes = trader.generate_synthetic_data(2000)
    trader.trading_loop(prices, volumes, episodes=10)
    print(f'{strategy.value}: {trader.total_reward:.2f}')
"

# 2. Adjust config and retrain
python -c "
import config
config.LR = 0.0001
config.EPSILON_DECAY = 0.99
# ... retrain
"

# 3. Validate results
python examples.py
```

### Workflow 3: Production Deploy
```bash
# 1. Setup live credentials
echo "IG_USERNAME=xxx" > .env
echo "IG_PASSWORD=xxx" >> .env
echo "IG_API_KEY=xxx" >> .env

# 2. Load trained model
python -c "
from main import XAUUSDHybridTrader
trader = XAUUSDHybridTrader(use_live_data=True)
trader.load_models()
# ... start trading
"
```

---

## Performance Monitoring

### Real-time Metrics
```bash
python -c "
from main import XAUUSDHybridTrader
import time

trader = XAUUSDHybridTrader()
prices, volumes = trader.generate_synthetic_data(5000)

start = time.time()
trader.trading_loop(prices, volumes, episodes=5)
elapsed = time.time() - start

print(f'Total time: {elapsed:.1f}s')
print(f'Time per episode: {elapsed/5:.1f}s')
print(f'Reward: {trader.total_reward:.2f}')
"
```

### Compare Strategies
```bash
python -c "
from main import XAUUSDHybridTrader
from ensemble_strategies import EnsembleStrategy
import numpy as np

strategies = [EnsembleStrategy.VOTING, EnsembleStrategy.WEIGHTED_VOTING]
prices, volumes = [2000 + np.random.randn()*5 for _ in range(2000)], [2000]*2000

for strat in strategies:
    trader = XAUUSDHybridTrader()
    trader.ensemble.strategy = strat
    trader.trading_loop(prices, volumes, episodes=10)
    print(f'{strat.value}: {np.mean(trader.episode_rewards):.2f}')
"
```

---

**🎯 Quick Reference Guide**
**Last Updated**: 2024
**Status**: Production-Ready

🚀 **Ready to Trade!**
