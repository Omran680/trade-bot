# Guide d'Utilisation - Hybrid PPO+DQN Trading System

## Installation

### 1. Cloner/Préparer le projet
```bash
cd X-DQN-Engine
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Configuration (Optional pour live trading)
Créer un fichier `.env`:
```
IG_USERNAME=your_username
IG_PASSWORD=your_password
IG_API_KEY=your_api_key
```

## Utilisation Simple

### Mode Entraînement Basique
```python
from main import XAUUSDHybridTrader

# Initialiser
trader = XAUUSDHybridTrader(use_live_data=False)

# Générer données de test
prices, volumes = trader.generate_synthetic_data(num_steps=5000)

# Entraîner
trader.trading_loop(prices, volumes, episodes=50, learning_mode=True)

# Sauvegarder
trader.save_models()
```

### Backtesting
```python
# Charger modèles
trader.load_models()

# Générer nouvelles données
prices_test, volumes_test = trader.generate_synthetic_data(num_steps=1000)

# Backtester
trader.backtest(prices_test, volumes_test)
```

### Live Trading (Avec IG Markets)
```python
from main import XAUUSDHybridTrader
import numpy as np

# Initialiser avec données live
trader = XAUUSDHybridTrader(use_live_data=True)
trader.load_models()

# Boucle trading
while True:
    # Récupérer prix actuel
    current_price = trader.trader.get_price("CS.D.IN_GOLD.MFI.IP")
    
    # Préparer état
    state = trader.get_state(trader.price_history, trader.volume_history)
    
    # Décision
    dqn_out = trader.get_dqn_output(state)
    ppo_out = trader.get_ppo_output(state)
    decision = trader.ensemble.combine_predictions(dqn_out, ppo_out)
    
    # Exécuter
    if decision['action'] == 0:  # BUY
        trader.trader.open_trade("CS.D.IN_GOLD.MFI.IP", "BUY", 0.1, 2, 3)
    elif decision['action'] == 1:  # SELL
        trader.trader.open_trade("CS.D.IN_GOLD.MFI.IP", "SELL", 0.1, 2, 3)
    
    time.sleep(60)  # Attendre 1 minute
```

## Configuration Avancée

### 1. Changer la Stratégie d'Ensemble
```python
from ensemble_strategies import EnsembleStrategy

trader = XAUUSDHybridTrader()
trader.ensemble.strategy = EnsembleStrategy.AVERAGING

# Options disponibles:
# - VOTING: Simple majority
# - WEIGHTED_VOTING: Pondéré par confiance (défaut)
# - AVERAGING: Moyenne des scores
# - STACKING: Meta-learner
# - MAJORITY_VOTING: Strictement unanime
```

### 2. Ajuster les Hyperparamètres
Éditer `config.py`:
```python
# DQN
EPSILON_DECAY = 0.99  # Plus rapide (exploration courte)
TARGET_UPDATE_FREQ = 500  # Plus fréquent (stability)

# PPO
PPO_CLIP_RATIO = 0.15  # Plus stricte (moins de variation)
PPO_ENTROPY_COEFF = 0.02  # Plus exploration

# Risk Management
STOP_LOSS_PCT = 1.5  # Plus agressif
TAKE_PROFIT_PCT = 5.0  # Attendre plus
```

### 3. Utiliser Données Réelles
```python
import pandas as pd

# Charger données historiques
df = pd.read_csv('xau_usd_data.csv')
prices = df['Close'].values
volumes = df['Volume'].values

# Entraîner
trader.trading_loop(prices, volumes, episodes=100)
```

## Analyse des Résultats

### Métriques de Performance de l'Ensemble
```python
metrics = trader.ensemble.get_performance_metrics()

print(f"DQN Performance: {metrics['avg_dqn_reward']:.3f}")
print(f"PPO Performance: {metrics['avg_ppo_reward']:.3f}")
print(f"DQN Weight: {metrics['current_dqn_weight']:.3f}")
print(f"PPO Weight: {metrics['current_ppo_weight']:.3f}")
```

### Historique des Épisodes
```python
import matplotlib.pyplot as plt

plt.plot(trader.episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()
```

### Détails des Décisions
```python
# Accéder aux décisions de chaque agent
state = trader.get_state(prices_history)
dqn_decision = trader.get_dqn_output(state)
ppo_decision = trader.get_ppo_output(state)
ensemble_decision = trader.ensemble.combine_predictions(dqn_decision, ppo_decision)

print(f"DQN Action: {dqn_decision['action']} (Q-max: {max(dqn_decision['q_values']):.3f})")
print(f"PPO Action: {ppo_decision['action']} (Prob: {max(ppo_decision['probs']):.3f})")
print(f"Ensemble: {ensemble_decision['action']} (Rationale: {ensemble_decision['rationale']})")
```

## Troubleshooting

### Problème: Models pas trouvés
```python
# Assurez-vous que le répertoire existe
import os
os.makedirs("./models", exist_ok=True)

# Puis relancer l'entraînement
trader.save_models()
```

### Problème: Mémoire insuffisante
```python
# Réduire la taille des buffers
config.MEMORY_SIZE = 2500  # au lieu de 5000
config.BATCH_SIZE = 16  # au lieu de 32
```

### Problème: Entraînement très lent
```python
# Augmenter taille des batches
config.BATCH_SIZE = 64

# Réduire nombre d'epochs PPO
config.PPO_EPOCHS = 1

# Augmenter learning rate légèrement
config.LR = 0.0005
```

### Problème: Rewards décroissent
```python
# Vérifier les poids d'ensemble
print(trader.ensemble.dqn_weight, trader.ensemble.ppo_weight)

# Si très déséquilibré, reset:
trader.ensemble.dqn_weight = 0.5
trader.ensemble.ppo_weight = 0.5

# Ou changer de stratégie
trader.ensemble.strategy = EnsembleStrategy.MAJORITY_VOTING
```

## Optimisation

### Pour Backtesting Rapide
```python
# Désactiver logging verbeux
config.VERBOSE = False

# Utiliser moins d'episodes
trader.trading_loop(prices, volumes, episodes=10, learning_mode=False)
```

### Pour Production
```python
# Activer checkpoints fréquents
config.CHECKPOINT_INTERVAL = 10

# Augmenter période d'évaluation
config.EVAL_INTERVAL = 50

# Utiliser averaging strategy (plus stable)
trader.ensemble.strategy = EnsembleStrategy.AVERAGING
```

### Pour Recherche
```python
# Tester multiples configurations
configs_to_test = [
    {'dqn_weight': 0.3, 'ppo_weight': 0.7, 'strategy': 'WEIGHTED_VOTING'},
    {'dqn_weight': 0.7, 'ppo_weight': 0.3, 'strategy': 'WEIGHTED_VOTING'},
    {'dqn_weight': 0.5, 'ppo_weight': 0.5, 'strategy': 'AVERAGING'},
]

results = []
for config in configs_to_test:
    trader = XAUUSDHybridTrader(
        dqn_weight=config['dqn_weight'],
        ppo_weight=config['ppo_weight']
    )
    # ... entraîner et backtester
    results.append({**config, 'final_reward': trader.total_reward})

# Analyser
best = max(results, key=lambda x: x['final_reward'])
print(f"Best config: {best}")
```

## Fichiers Importants

| Fichier | Description |
|---------|-------------|
| `main.py` | Point d'entrée principal |
| `agent.py` | DQNAgent, PPOAgent, HybridTradingAgent |
| `model.py` | Réseaux de neurones (DQN, PPO, Hybrid) |
| `ensemble_strategies.py` | Stratégies d'agrégation |
| `feature_extractor.py` | Extraction de features + gestion état |
| `config.py` | Configuration globale |
| `replay_buffer.py` | Buffer d'expérience |
| `trader.py` | Interface IG Markets |
| `ARCHITECTURE.md` | Documentation technique |

## Prochaines Étapes

1. **Optimiser pour XAU/USD réel**
   - Ajuster features pour patterns spécifiques
   - Backtester sur données 2023-2024
   - Tester avec commissions réelles

2. **Améliorer Risk Management**
   - Position sizing dynamique
   - Portfolio-level hedging
   - Correlation tracking

3. **Ajouter Signals Externes**
   - Economic calendar
   - Sentiment analysis
   - Technical indicators (RSI, MACD)

4. **Multi-Asset Support**
   - Combiner XAU/USD + EUR/USD + S&P500
   - Cross-asset ensemble learning

---

**Version**: 1.0
**Last Updated**: 17-04-2026
**Status**: Production-Ready
