"""
Exemples Avancés - Hybrid PPO+DQN Trading System
"""

import numpy as np
from main import XAUUSDHybridTrader
from ensemble_strategies import EnsembleStrategy, EnsembleDecisionMaker
from risk_manager import TradingSession, PortfolioRiskManager, DynamicRiskAdjustment
from feature_extractor import FeatureExtractor, TradingState
import config


# ============= EXEMPLE 1: Comparaison des Stratégies d'Ensemble =============
def example_ensemble_strategies():
    """Comparer les performances de différentes stratégies d'ensemble"""
    print("\n" + "="*80)
    print("EXEMPLE 1: Comparaison des Stratégies d'Ensemble")
    print("="*80 + "\n")
    
    strategies = [
        EnsembleStrategy.VOTING,
        EnsembleStrategy.WEIGHTED_VOTING,
        EnsembleStrategy.AVERAGING,
        EnsembleStrategy.MAJORITY_VOTING,
    ]
    
    trader = XAUUSDHybridTrader(use_live_data=False)
    prices, volumes = trader.generate_synthetic_data(num_steps=2000)
    
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy.value}...")
        
        # Reset trader
        trader = XAUUSDHybridTrader(use_live_data=False)
        trader.ensemble.strategy = strategy
        
        # Mini-train
        trader.trading_loop(prices, volumes, episodes=5, learning_mode=True)
        
        # Store results
        metrics = trader.ensemble.get_performance_metrics()
        results[strategy.value] = {
            'total_reward': trader.total_reward,
            'avg_reward': np.mean(trader.episode_rewards),
            'metrics': metrics
        }
        
        print(f"  Total Reward: {trader.total_reward:.2f}")
        print(f"  Avg/Episode: {np.mean(trader.episode_rewards):.2f}")
    
    # Compare
    print("\n" + "-"*80)
    print("COMPARISON RESULTS:")
    print("-"*80)
    
    best_strategy = max(results.items(), key=lambda x: x[1]['total_reward'])
    print(f"\nBest Strategy: {best_strategy[0]} (Reward: {best_strategy[1]['total_reward']:.2f})")
    
    for strategy, result in results.items():
        print(f"\n{strategy}:")
        print(f"  Total Reward: {result['total_reward']:.2f}")
        print(f"  Avg Reward: {result['avg_reward']:.2f}")


# ============= EXEMPLE 2: Optimisation d'Hyperparamètres =============
def example_hyperparameter_optimization():
    """Chercher les meilleurs hyperparameters"""
    print("\n" + "="*80)
    print("EXEMPLE 2: Grid Search Hyperparameters")
    print("="*80 + "\n")
    
    # Générer données une fois
    temp_trader = XAUUSDHybridTrader()
    prices, volumes = temp_trader.generate_synthetic_data(num_steps=2000)
    
    # Hyperparams à tester
    configs = {
        'learning_rates': [0.0001, 0.0003, 0.001],
        'batch_sizes': [16, 32, 64],
    }
    
    best_config = None
    best_reward = -float('inf')
    
    for lr in configs['learning_rates']:
        for batch_size in configs['batch_sizes']:
            print(f"\nTesting LR={lr}, Batch={batch_size}...")
            
            # Créer trader avec config
            config.LR = lr
            config.BATCH_SIZE = batch_size
            
            trader = XAUUSDHybridTrader()
            trader.trading_loop(prices, volumes, episodes=3, learning_mode=True)
            
            total_reward = trader.total_reward
            print(f"  Result: {total_reward:.2f}")
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_config = {'lr': lr, 'batch_size': batch_size}
    
    print("\n" + "-"*80)
    print(f"BEST CONFIG: {best_config} (Reward: {best_reward:.2f})")
    print("-"*80)
    
    # Reset config
    config.LR = 0.0003
    config.BATCH_SIZE = 32


# ============= EXEMPLE 3: Risk Management Avancé =============
def example_advanced_risk_management():
    """Démonstration du système de risk management"""
    print("\n" + "="*80)
    print("EXEMPLE 3: Advanced Risk Management")
    print("="*80 + "\n")
    
    # Créer session avec risque manager
    session = TradingSession(capital=10000)
    
    print("Initial Capital: $10,000\n")
    
    # Simulation de 5 trades
    trades = [
        {'entry': 2050, 'stop': 2045, 'tp': 2060, 'exit': 2062, 'type': 'LONG'},
        {'entry': 2062, 'stop': 2067, 'tp': 2050, 'exit': 2053, 'type': 'SHORT'},
        {'entry': 2053, 'stop': 2048, 'tp': 2065, 'exit': 2064, 'type': 'LONG'},
        {'entry': 2064, 'stop': 2069, 'tp': 2050, 'exit': 2049, 'type': 'SHORT'},
        {'entry': 2049, 'stop': 2044, 'tp': 2060, 'exit': 2045, 'type': 'LONG'},
    ]
    
    for i, trade in enumerate(trades, 1):
        print(f"Trade {i}: {trade['type']}")
        
        # Ouvrir
        result = session.open_position(
            entry_price=trade['entry'],
            stop_loss_price=trade['stop'],
            take_profit_price=trade['tp'],
            direction=trade['type']
        )
        
        if result['status'] == 'opened':
            print(f"  Entry: {trade['entry']}")
            print(f"  Position Size: {result['position']['position_size']:.3f} lots")
            
            # Fermer
            close_result = session.close_position(0, exit_price=trade['exit'])
            print(f"  Exit: {trade['exit']}")
            print(f"  PnL: ${close_result['pnl']:.2f} ({close_result['return_pct']:.2f}%)")
        else:
            print(f"  ✗ Rejected: {result['reason']}")
        
        print()
    
    # Summary
    summary = session.get_session_summary()
    print("-"*80)
    print("SESSION SUMMARY:")
    print("-"*80)
    print(f"Final Capital: ${summary['current_capital']:.2f}")
    print(f"Total Return: {summary['return_pct']:.2f}%")
    print(f"Win Rate: {summary['win_rate']:.1%}")
    print(f"Profit Factor: {summary['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {summary['max_drawdown']:.2f}%")


# ============= EXEMPLE 4: Analyse d'une Décision Ensemble =============
def example_ensemble_decision_analysis():
    """Analyser comment l'ensemble prend une décision"""
    print("\n" + "="*80)
    print("EXEMPLE 4: Ensemble Decision Analysis")
    print("="*80 + "\n")
    
    trader = XAUUSDHybridTrader(use_live_data=False)
    
    # Créer un état
    prices = [2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054,
              2055, 2055, 2056, 2057, 2058, 2059, 2060, 2059, 2058, 2057]
    volumes = [2000, 2100, 2200, 2050, 1950, 2300, 2100, 2000, 2150, 2200,
               2050, 1900, 2250, 2000, 2100, 2200, 2150, 2050, 1950, 2100]
    
    state = trader.get_state(prices, volumes)
    
    # Get predictions
    dqn_out = trader.get_dqn_output(state)
    ppo_out = trader.get_ppo_output(state)
    
    action_names = ['BUY', 'SELL', 'HOLD']
    
    print("DQN Analysis:")
    print(f"  Predicted Action: {action_names[dqn_out['action']]}")
    print(f"  Q-Values: BUY={dqn_out['q_values'][0]:.4f}, SELL={dqn_out['q_values'][1]:.4f}, HOLD={dqn_out['q_values'][2]:.4f}")
    print(f"  Confidence: {dqn_out['confidence']:.1%}")
    
    print("\nPPO Analysis:")
    print(f"  Predicted Action: {action_names[ppo_out['action']]}")
    probs = ppo_out['probs']
    print(f"  Probabilities: BUY={probs[0]:.1%}, SELL={probs[1]:.1%}, HOLD={probs[2]:.1%}")
    print(f"  Value Estimate: {ppo_out['value']:.4f}")
    print(f"  Confidence: {ppo_out['confidence']:.1%}")
    
    # Ensemble decision
    ensemble_decision = trader.ensemble.combine_predictions(dqn_out, ppo_out)
    
    print("\nEnsemble Decision:")
    print(f"  Final Action: {action_names[ensemble_decision['action']]}")
    print(f"  Confidence: {ensemble_decision['confidence']:.1%}")
    print(f"  Rationale: {ensemble_decision['rationale']}")
    print(f"  Breakdown:")
    print(f"    DQN: {ensemble_decision['breakdown']}")


# ============= EXEMPLE 5: Feature Importance Analysis =============
def example_feature_analysis():
    """Analyser quels features sont importants"""
    print("\n" + "="*80)
    print("EXEMPLE 5: Feature Extraction & Analysis")
    print("="*80 + "\n")
    
    extractor = FeatureExtractor(lookback_window=20)
    
    # Créer données avec différents patterns
    # Pattern 1: Uptrend
    print("Pattern 1: Uptrend")
    uptrend_prices = np.linspace(2000, 2100, 20)
    features = extractor.extract_features(uptrend_prices.tolist())
    print(f"  Features shape: {features.shape}")
    print(f"  Feature values: {features[0][:5]}...")  # First 5 features
    
    # Pattern 2: Downtrend
    print("\nPattern 2: Downtrend")
    downtrend_prices = np.linspace(2100, 2000, 20)
    features = extractor.extract_features(downtrend_prices.tolist())
    print(f"  Features shape: {features.shape}")
    print(f"  Feature values: {features[0][:5]}...")
    
    # Pattern 3: Volatility spike
    print("\nPattern 3: High Volatility")
    volatile_prices = 2050 + np.random.normal(0, 5, 20)
    features = extractor.extract_features(volatile_prices.tolist())
    print(f"  Features shape: {features.shape}")
    print(f"  Feature values: {features[0][:5]}...")
    
    # Pattern 4: Mean reversion
    print("\nPattern 4: Mean Reversion")
    prices = [2050] * 10 + [2045, 2040, 2035, 2030, 2035, 2040, 2045, 2050, 2055, 2050]
    features = extractor.extract_features(prices)
    print(f"  Features shape: {features.shape}")
    print(f"  Feature values: {features[0][:5]}...")


# ============= EXEMPLE 6: Backtesting avec Différentes Conditions =============
def example_backtest_scenarios():
    """Backtester dans différentes conditions de marché"""
    print("\n" + "="*80)
    print("EXEMPLE 6: Backtesting Market Scenarios")
    print("="*80 + "\n")
    
    scenarios = {
        'Trending Up': lambda: np.linspace(2000, 2100, 1000),
        'Trending Down': lambda: np.linspace(2100, 2000, 1000),
        'Ranging': lambda: 2050 + 25 * np.sin(np.linspace(0, 20*np.pi, 1000)),
        'High Volatility': lambda: 2050 + np.random.normal(0, 3, 1000),
        'Low Volatility': lambda: 2050 + np.random.normal(0, 0.5, 1000),
    }
    
    for scenario_name, price_gen in scenarios.items():
        print(f"\nBacktesting: {scenario_name}")
        print("-" * 40)
        
        prices = price_gen()
        volumes = np.full(len(prices), 2000)
        
        trader = XAUUSDHybridTrader(use_live_data=False)
        trader.backtest(prices, volumes)


# ============= EXEMPLE 7: Dynamic Risk Adjustment =============
def example_dynamic_risk():
    """Démonstration d'ajustement dynamique du risque"""
    print("\n" + "="*80)
    print("EXEMPLE 7: Dynamic Risk Adjustment")
    print("="*80 + "\n")
    
    dra = DynamicRiskAdjustment(base_risk=0.02)
    
    # Simulation: Performance s'améliore
    print("Performance Progression:\n")
    
    scenarios = [
        {'win_rate': 0.45, 'sharpe': 0.5, 'drawdown': 0.15, 'label': 'Poor'},
        {'win_rate': 0.48, 'sharpe': 1.0, 'drawdown': 0.12, 'label': 'Fair'},
        {'win_rate': 0.52, 'sharpe': 1.5, 'drawdown': 0.10, 'label': 'Good'},
        {'win_rate': 0.55, 'sharpe': 2.0, 'drawdown': 0.08, 'label': 'Excellent'},
    ]
    
    for scenario in scenarios:
        risk = dra.adjust_risk(
            win_rate=scenario['win_rate'],
            sharpe_ratio=scenario['sharpe'],
            drawdown=scenario['drawdown']
        )
        
        print(f"{scenario['label']} Performance:")
        print(f"  Win Rate: {scenario['win_rate']:.1%}")
        print(f"  Sharpe: {scenario['sharpe']:.1f}")
        print(f"  Drawdown: {scenario['drawdown']:.1%}")
        print(f"  → Adjusted Risk: {risk:.2%} (Base: {dra.base_risk:.2%})")
        print()


# ============= RUNNER =============
def run_all_examples():
    """Exécuter tous les exemples"""
    
    examples = [
        ("Ensemble Strategies", example_ensemble_strategies),
        # ("Hyperparameter Optimization", example_hyperparameter_optimization),  # Comment out - slow
        ("Advanced Risk Management", example_advanced_risk_management),
        ("Ensemble Decision Analysis", example_ensemble_decision_analysis),
        ("Feature Analysis", example_feature_analysis),
        # ("Backtest Scenarios", example_backtest_scenarios),  # Comment out - slow
        ("Dynamic Risk Adjustment", example_dynamic_risk),
    ]
    
    print("\n" + "="*80)
    print("HYBRID PPO+DQN TRADING SYSTEM - ADVANCED EXAMPLES")
    print("="*80)
    
    for i, (name, func) in enumerate(examples, 1):
        try:
            print(f"\n[{i}/{len(examples)}] Running: {name}")
            func()
        except Exception as e:
            print(f"Error running {name}: {e}")
    
    print("\n" + "="*80)
    print("✓ All examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_all_examples()
