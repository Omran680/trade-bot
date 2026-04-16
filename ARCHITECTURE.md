# Architecture Hybride PPO+DQN pour le Trading XAU/USD

## Vue d'ensemble

Ce système implémente une **architecture d'ensemble learning** qui combine deux algorithmes d'apprentissage par renforcement (RL) complémentaires pour le trading du gold (XAU/USD):

- **DQN (Deep Q-Network)**: Apprentissage off-policy, recommande les actions avec la meilleure valeur Q estimée
- **PPO (Proximal Policy Optimization)**: Apprentissage on-policy stable, optimise la politique directement

## Architecture Système

```
┌─────────────────────────────────────────────────────────────────┐
│                    ÉTAT D'ENTRÉE (18 features)                   │
│  - Momentum | Volatility | Trend | Mean Reversion | Volume       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     ┌──────────┐    ┌──────────┐    ┌──────────────┐
     │   DQN    │    │   PPO    │    │   HYBRID     │
     │  Network │    │ Network  │    │   NETWORK    │
     └──────────┘    └──────────┘    │ (Shared)     │
          │                │         └──────────────┘
          │ Q-values       │ Logits+Value
          └────────────────┼────────────────┐
                           │                │
                    ┌──────▼─────────────────▼────────┐
                    │  ENSEMBLE DECISION MAKER        │
                    │ - Voting Strategy               │
                    │ - Weighted Aggregation          │
                    │ - Confidence Calculation        │
                    └──────┬────────────────────────────┘
                           │
                    ┌──────▼──────────┐
                    │   ACTION (BUY,  │
                    │   SELL, HOLD)   │
                    └─────────────────┘
```

## Composants Clés

### 1. DQN Agent (agent.py - DQNAgent)

**Caractéristiques:**
- Apprentissage off-policy via Experience Replay
- Target Network pour stabilité
- Epsilon-greedy exploration
- Double DQN implicitly (via target network)

**Forces:**
- Très efficace pour l'exploration-exploitation
- Converge rapidement
- Bonne pour identifier les patterns exploitables

**Faiblesses:**
- Peut être instable avec certains environnements
- Tend à surestimer les Q-values
- Sample inefficient en début

### 2. PPO Agent (agent.py - PPOAgent)

**Caractéristiques:**
- Actor-Critic architecture
- Generalized Advantage Estimation (GAE)
- PPO clipping pour stabilité
- On-policy learning

**Forces:**
- Très stable et robuste
- Sample efficient
- Convergence garantie
- Bonne pour le fine-tuning

**Faiblesses:**
- Moins exploratoire que DQN
- Peut converger vers optimum local
- Plus lent à converger

### 3. Hybrid Network (model.py - HybridNetwork)

**Architecture:**
```
Input State (18)
    │
    ├─→ Shared Feature Extractor (256→256)
    │
    ├─→ DQN Head (128) → Q-values (3)
    │
    └─→ PPO Head (128) → Logits (3)
                    └─→ Value (1)
```

**Avantages:**
- Partage de représentations
- Transfert learning implicite
- Réduction des paramètres
- Convergence plus rapide

### 4. Ensemble Decision Maker (ensemble_strategies.py)

**Stratégies disponibles:**

#### a. **Weighted Voting** (Recommandé)
```python
score_dqn = confidence_dqn * weight_dqn
score_ppo = confidence_ppo * weight_ppo
action = argmax([score_dqn, score_ppo])
```

#### b. **Majority Voting**
```python
if dqn_action == ppo_action:
    action = dqn_action
else:
    action = argmax(confidence_dqn, confidence_ppo)
```

#### c. **Averaging**
```python
combined_scores = (q_values * weight_dqn) + (probs * weight_ppo)
action = argmax(combined_scores)
```

#### d. **Stacking**
```python
meta_features = [dqn_action_onehot, ppo_action_onehot, confidences, values]
action = meta_learner(meta_features)
```

## Feature Extraction (feature_extractor.py)

### 18 Features XAU/USD:

**Momentum (5):**
1. Rate of Change (ROC)
2. Price vs SMA court terme
3. SMA court vs long terme
4. Price vs EMA
5. Momentum (derniers 5 bars)

**Volatility (4):**
6. Volatilité (std dev)
7. Average True Range (ATR)
8. Bollinger Band position
9. Range normalisé

**Trend (4):**
10. Régression linéaire slope
11. Higher Highs pattern
12. Lower Lows pattern
13. Trend strength

**Mean Reversion (3):**
14. RSI (Relative Strength Index)
15. Distance à la moyenne
16. Stochastic Oscillator

**Volume (2):**
17. Volume trend
18. Volume volatilité

## Cycle d'Entraînement

### Phase 1: Collecte d'Expérience
```
Pour chaque step:
  1. État actuel → Features
  2. DQN décide → Q-values
  3. PPO décide → Logits
  4. Ensemble vote → Action finale
  5. Exécuter action → Reward
  6. Stocker transition
```

### Phase 2: Entraînement DQN
```
Tous les 32 steps:
  1. Sampler mini-batch (32 transitions)
  2. Calculer target Q-values
  3. Backprop MSE loss
  4. Mettre à jour target network tous les 1000 steps
  5. Decay epsilon
```

### Phase 3: Entraînement PPO
```
Tous les 64 steps:
  1. Collecter trajectoire (64 steps)
  2. Calculer GAE advantages
  3. Pour 3 epochs:
     - Forward pass
     - Calculer ratio (new policy / old policy)
     - PPO loss = -E[min(ratio*A, clip(ratio)*A)]
     - Value loss = E[(R - V)²]
     - Total loss = policy_loss + 0.5*value_loss - 0.01*entropy
  4. Backprop et update
```

### Phase 4: Adaptation Poids
```
À chaque episode:
  1. Calculer reward moyen DQN
  2. Calculer reward moyen PPO
  3. weight_dqn = reward_dqn / (reward_dqn + reward_ppo)
  4. weight_ppo = reward_ppo / (reward_dqn + reward_ppo)
```

## Avantages de l'Architecture Hybride

### 1. **Diversité des Stratégies**
- DQN: Bon pour patterns court-terme
- PPO: Bon pour tendances long-terme
- Combinaison → Robustesse

### 2. **Réduction du Risque Systématique**
- Erreurs corrélées diminuées
- Votes conflictuels → Signal de prudence (HOLD)
- Votes unanimes → Signal fort

### 3. **Adaptabilité Dynamique**
- Poids ajustés selon performance
- Auto-régulation du système
- Apprentissage continu

### 4. **Stabilité Améliorée**
- PPO stabilise DQN instable
- DQN explore là où PPO stagne
- Feature sharing → Convergence rapide

## Indicateurs de Performance

### Individuels:
```python
metrics = ensemble.get_performance_metrics()
# {
#   'avg_dqn_reward': float,
#   'avg_ppo_reward': float,
#   'dqn_variance': float,
#   'ppo_variance': float,
#   'current_dqn_weight': float,
#   'current_ppo_weight': float
# }
```

### Trading:
```
Total Return = (Final Capital - Initial Capital) / Initial Capital
Win Rate = Winning Trades / Total Trades
Profit Factor = Total Gains / Total Losses
Sharpe Ratio = (Mean Return - Risk-Free Rate) / Std Dev
Max Drawdown = Maximum peak-to-trough decline
```

## Configuration Optimale

```python
# DQN
epsilon_start = 1.0
epsilon_decay = 0.995
target_update = 1000
batch_size = 32

# PPO
clip_ratio = 0.2
epochs = 3
gae_lambda = 0.95
entropy_coeff = 0.01

# Ensemble
strategy = "weighted_voting"
dqn_initial_weight = 0.5
ppo_initial_weight = 0.5

# Stop Loss / Take Profit
stop_loss_pct = 2.0
take_profit_pct = 3.0
```

## Utilisation

### Entraînement:
```python
trader = XAUUSDHybridTrader(use_live_data=False)
prices, volumes = trader.generate_synthetic_data(num_steps=5000)
trader.trading_loop(prices, volumes, episodes=100, learning_mode=True)
trader.save_models()
```

### Backtesting:
```python
trader.load_models()
trader.backtest(prices_test, volumes_test)
```

### Trading en Live:
```python
trader = XAUUSDHybridTrader(use_live_data=True)
trader.load_models()
# Boucle de trading continu...
```

## Améliorations Futures

1. **Multi-timeframe**: Combiner 5min, 15min, 1h
2. **Meta-features**: Inclure sentiment analysis, news
3. **Curriculum Learning**: Progression graduelle
4. **Risk Management**: Position sizing dynamique
5. **Hierarchical RL**: Multiple decision-making levels
6. **Attention Mechanisms**: Sélection de features dynamique

## Références

- [DQN](https://arxiv.org/abs/1312.5602) - Mnih et al., 2013
- [PPO](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [GAE](https://arxiv.org/abs/1506.02438) - Schulman et al., 2015
- [Ensemble Methods](https://papers.nips.cc/paper/2012/hash/4ddb5fda2a58bfb13e3bb0b3fbcfb75f-Abstract.html)

---

**Auteur**: Hybrid Trading System
**Date**: 2024
**Version**: 1.0
