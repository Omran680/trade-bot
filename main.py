"""
XAUUSDHybridTrader - Main Entry Point
Point d'entrée principal du bot de trading hybride DQN+PPO
"""

import argparse
import logging
from XAUUSDHybridTrader import XAUUSDHybridTrader

# Configuration du logging (affichage console)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Point d'entrée principal du trading bot"""
    
    # Arguments de ligne de commande
    parser = argparse.ArgumentParser(
        description='XAUUSDHybridTrader - Bot de trading RL hybride DQN+PPO'
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Nombre maximum d'étapes (défaut: 50)"
    )
    parser.add_argument(
        "--forever",
        action="store_true",
        help="Exécuter indéfiniment jusqu'à Ctrl+C"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Mode simulation (défaut: activé, pour sécurité)"
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Désactiver l'entraînement (mode inférence seulement)"
    )
    
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("🚀 Démarrage du XAUUSDHybridTrader")
    logger.info("=" * 80)
    
    # Afficher configuration
    logger.info(f"Mode: {'DRY-RUN' if args.dry_run else 'LIVE'} (ATTENTION: Live = VRAI ARGENT)")
    logger.info(f"Entraînement: {'OUI' if not args.no_train else 'NON'}")
    logger.info(f"Étapes: {'Infini' if args.forever else args.max_steps}")
    logger.info("=" * 80)
    
    try:
        # Créer le bot
        logger.info("📦 Initialisation du bot...")
        bot = XAUUSDHybridTrader(
            use_live_data=True,
            dry_run= False
            #args.dry_run
        )
        logger.info("✅ Bot initialisé")

        # Activer streaming si disponible
        if bot.trader:
            try:
                logger.info("📡 Activation du streaming des prix...")
                bot.trader.enable_streaming(bot.epic)
                logger.info("✅ Streaming activé")
            except Exception as e:
                logger.warning(f"⚠️ Streaming non disponible: {e}")

        # Charger les modèles précédents s'ils existent
        logger.info("📚 Chargement des modèles...")
        bot.load_models()

        # Lancer la boucle de trading
        logger.info("-" * 80)
        if args.forever:
            logger.info("▶️ Lancement: Mode INFINI (Ctrl+C pour arrêter)")
            bot.live_trading_loop(
                max_steps=None,
                training_mode=not args.no_train
            )
        else:
            logger.info(f"▶️ Lancement: {args.max_steps} étapes")
            bot.live_trading_loop(
                max_steps=args.max_steps,
                training_mode=not args.no_train
            )

        # Sauvegarder les modèles après succès
        logger.info("-" * 80)
        logger.info("💾 Sauvegarde des modèles...")
        bot.save_models()
        logger.info("✅ Modèles sauvegardés")
        
        # Afficher résumé final
        if bot.metrics.trades:
            logger.info("\n" + "=" * 80)
            logger.info("📊 RÉSUMÉ FINAL DE LA SESSION")
            logger.info("=" * 80)
            logger.info(f"Total trades: {len(bot.metrics.trades)}")
            logger.info(f"Win rate: {bot.metrics.win_rate():.2f}%")
            logger.info(f"Total P&L: ${bot.metrics.total_pnl():.2f}")
            logger.info(f"Average P&L per trade: ${bot.metrics.avg_pnl_per_trade():.2f}")
            logger.info(f"Max drawdown: {bot.metrics.max_drawdown():.2f}")
            logger.info("=" * 80 + "\n")

    except KeyboardInterrupt:
        # Arrêt par l'utilisateur
        logger.info("\n" + "=" * 80)
        logger.info("🛑 ARRÊT PAR L'UTILISATEUR (Ctrl+C)")
        logger.info("=" * 80)
        
        # Sauvegarder d'urgence
        try:
            logger.info("💾 Sauvegarde d'urgence des modèles...")
            bot.save_models()
            logger.info("✅ Modèles sauvegardés")
        except:
            logger.error("❌ Erreur lors de la sauvegarde!")
        
        # Afficher résumé avant arrêt
        if bot.metrics.trades:
            logger.info(f"\n📊 Trades avant arrêt: {len(bot.metrics.trades)}")
            logger.info(f"💰 P&L: ${bot.metrics.total_pnl():.2f}\n")
        
        logger.info("✅ Arrêt sécurisé")

    except Exception as e:
        # Erreur non gérée
        logger.error("\n" + "=" * 80)
        logger.error("❌ ERREUR CRITIQUE")
        logger.error("=" * 80)
        logger.error(f"Erreur: {e}", exc_info=True)
        
        # Sauvegarder quand même
        try:
            logger.info("💾 Tentative de sauvegarde des modèles...")
            bot.save_models()
            logger.info("✅ Modèles sauvegardés")
        except:
            logger.error("❌ Impossible de sauvegarder!")
        
        logger.error("=" * 80 + "\n")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())