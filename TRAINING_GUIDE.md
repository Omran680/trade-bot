# 📚 Guide de Sauvegarde et Entraînement Continu

## 🎯 Vue d'ensemble

Le système X-DQN-Engine supporte maintenant la **sauvegarde automatique** et le **chargement des modèles** pour permettre un entraînement continu et la persistence de l'apprentissage.

## 💾 Système de Sauvegarde

### Structure des fichiers sauvegardés

```
models/
├── dqn_model.pkl        # Poids du réseau DQN
├── ppo_model.pkl        # Poids du réseau PPO
└── .gitkeep             # Git tracking
```

### Types de sauvegarde

#### 1. **Checkpoint Automatique** 
- Intervalle: Tous les `CHECKPOINT_INTERVAL * 10` steps (défaut: 50 * 10 = 500 steps)
- Survient automatiquement pendant `live_trading_loop()`
- Affiche: `💾 Checkpoint at step X...`

#### 2. **Sauvegarde à la Fin de Session**
- Automatiquement quand le trading s'arrête
- Automatiquement en cas d'interruption (Ctrl+C)
- Automatiquement en cas d'erreur

#### 3. **Sauvegarde Manuel**
```python
trader = XAUUSDHybridTrader()
# ... entrainement ...
trader.save_models()  # Sauvegarde immédiate
```

## 🚀 Utilisation

### Démarrage Basique avec Auto-Load

```bash
python main.py
```

**Comportement:**
1. ✓ Cherche les modèles dans `./models/`
2. ✓ Si trouvés: les charge et continue l'entraînement
3. ✓ Si non: démarre l'entraînement depuis zéro
4. ✓ Sauvegarde automatiquement à l'arrêt

### Entraînement Long avec Checkpoints

```python
from main import XAUUSDHybridTrader

# Créer trader
trader = XAUUSDHybridTrader(use_live_data=False)

# Charger modèles précédents (s'ils existent)
trader.load_models()

# Lancer entraînement
trader.live_trading_loop(max_steps=5000, training_mode=True)

# Sauvegarder (optionnel - fait automatiquement)
trader.save_models()
```

### Recommencer l'Entraînement

```python
# Créer nouveau trader (poids initialisés aléatoirement)
trader = XAUUSDHybridTrader()

# Ne PAS charger - commencer from scratch
# trainer.load_models()  # <- Commenter cette ligne

trader.live_trading_loop(max_steps=1000, training_mode=True)
```

## 📊 Configuration des Checkpoints

Modifier dans `config.py`:

```python
CHECKPOINT_INTERVAL = 50  # N * 10 steps entre checkpoints
# 50 = checkpoint tous les 500 steps
# 100 = checkpoint tous les 1000 steps
```

## 🔍 Inspection des Modèles Sauvegardés

```python
from agent import DQNAgent, PPOAgent
import pickle

# Charger les poids bruts
with open('./models/dqn_model.pkl', 'rb') as f:
    dqn_params = pickle.load(f)
    print("DQN weights:")
    for k, v in dqn_params.items():
        print(f"  {k}: {v.shape}")

with open('./models/ppo_model.pkl', 'rb') as f:
    ppo_params = pickle.load(f)
    print("PPO weights:")
    for k, v in ppo_params.items():
        print(f"  {k}: {v.shape}")
```

## ✅ Vérifier que ça Marche

### Test 1: Sauvegarde Basique

```python
from main import XAUUSDHybridTrader

trader = XAUUSDHybridTrader()
trader.save_models()  # Doit créer ./models/dqn_model.pkl et ppo_model.pkl
```

### Test 2: Chargement

```python
from main import XAUUSDHybridTrader

trader = XAUUSDHybridTrader()
success = trader.load_models()  # Doit retourner True
assert success, "Chargement des modèles a échoué!"
```

### Test 3: Entraînement Continu

```bash
# Session 1: Entraînement initial
python main.py  # Run 20 steps, puis Ctrl+C

# Session 2: Continuer l'entraînement
python main.py  # Charge les modèles de Session 1, continue
```

## 🎓 Workflow Recommandé

### Phase 1: Formation Initiale (24-48h)
```python
from main import XAUUSDHybridTrader

trader = XAUUSDHybridTrader(use_live_data=False)
# Entraînement rapide sur données synthétiques
trader.live_trading_loop(max_steps=10000, training_mode=True)
# Sauvegarde automatique
```

### Phase 2: Affinage (7j)
```python
from main import XAUUSDHybridTrader

trader = XAUUSDHybridTrader(use_live_data=False)
trader.load_models()  # Charger Phase 1
# Entraînement supplémentaire
trader.live_trading_loop(max_steps=50000, training_mode=True)
# Sauvegarde automatique
```

### Phase 3: Backtesting
```python
from main import XAUUSDHybridTrader

trader = XAUUSDHybridTrader()
trader.load_models()  # Charger Phase 2
# Tester sur données de backtest
# ...
```

### Phase 4: Live Trading
```python
from main import XAUUSDHybridTrader

trader = XAUUSDHybridTrader(use_live_data=True, dry_run=True)
trader.load_models()  # Charger Phase 3
trader.live_trading_loop(max_steps=1000, training_mode=True)
# Trading live + entraînement continu
```

## ⚠️ Points Important

### ✓ À Faire
- ✓ Sauvegarder régulièrement (automatique)
- ✓ Charger modèles avant continuation
- ✓ Utiliser dry_run=True avant live trading
- ✓ Monitorer les checkpoints

### ✗ À Éviter
- ✗ Supprimer le répertoire `./models/` pendant l'entraînement
- ✗ Modifier les fichiers .pkl manuellement
- ✗ Charger des modèles avec architectures différentes
- ✗ Passer de training_mode=True à False sans sauvegarder

## 🐛 Troubleshooting

### Problème: "No saved models found"
```python
# Solution 1: Vérifier que ./models/ existe
import os
os.makedirs('./models', exist_ok=True)

# Solution 2: Sauvegarder d'abord
trader.save_models()
```

### Problème: Models pas à jour après Ctrl+C
```python
# Solution: Utiliser try/finally
try:
    trader.live_trading_loop(max_steps=1000)
finally:
    trader.save_models()  # Garantit sauvegarde
```

### Problème: Erreur lors du chargement
```python
# Vérifier les fichiers existent
import os
print(os.path.exists('./models/dqn_model.pkl'))
print(os.path.exists('./models/ppo_model.pkl'))
```

## 📈 Métriques de Sauvegarde

Pour tracker l'entraînement:

```python
trader = XAUUSDHybridTrader()
trader.load_models()

# Ajouter logging personnalisé
import datetime
with open('training_log.txt', 'a') as f:
    f.write(f"{datetime.datetime.now()} - Checkpoint sauvegardé\n")
    f.write(f"DQN Epsilon: {trader.agent.dqn_agent.epsilon}\n")
    f.write(f"PPO Buffers: {len(trader.agent.ppo_agent.reward_buffer)}\n")
    f.write("---\n")
```

## 🎯 Prochaines Étapes

- [ ] Implémenter versioning des modèles (model_v1, model_v2, etc.)
- [ ] Ajouter monitoring du training loss
- [ ] Implémenter checkpoint conditionnels (meilleure reward)
- [ ] Ajouter support TensorFlow/PyTorch pour comparaison
