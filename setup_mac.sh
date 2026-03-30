#!/bin/bash
# ============================================================
#  setup_mac.sh — Installation complète sur Mac (Apple Silicon & Intel)
#  Usage : chmod +x setup_mac.sh && ./setup_mac.sh
# ============================================================

set -e
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "🥇 Gold Scalper Bot — Installation Mac"
echo "======================================="

# 1. Python 3.11+
if ! command -v python3 &>/dev/null; then
    echo "${YELLOW}Python non trouvé. Installation via Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install python@3.11
fi
echo "${GREEN}✓ Python : $(python3 --version)${NC}"

# 2. Environnement virtuel
python3 -m venv .venv
source .venv/bin/activate
echo "${GREEN}✓ Venv activé${NC}"

# 3. Dépendances
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "${GREEN}✓ Dépendances installées${NC}"

# 4. Dossiers de données
mkdir -p logs data
echo "${GREEN}✓ Dossiers créés${NC}"

# 5. Rappel configuration
echo ""
echo "======================================="
echo "  ✅ Installation terminée !"
echo "======================================="
echo ""
echo "  1. Édite config.py et ajoute tes clés Alpaca :"
echo "     ALPACA_API_KEY    = 'PKxxxx'"
echo "     ALPACA_API_SECRET = 'xxxxx'"
echo ""
echo "  2. Lance le bot :"
echo "     source .venv/bin/activate && python main.py"
echo ""
echo "  3. Tester un module seul :"
echo "     python -m modules.load_balancer"
echo "     python -m strategies.strategies"
echo "     python -m modules.session_filter"
echo "     python -m modules.risk_manager"
echo ""
