# modules/session_filter.py
from datetime import datetime, timezone
import config
from core.logger import get_logger

log = get_logger()

def is_tradeable():
    """Verifie si la session de trading est active"""
    now = datetime.now(timezone.utc)
    current_hour = now.hour
    
    if config.SESSION_START <= current_hour < config.SESSION_END:
        return True
    return False

def session_info():
    """Retourne les informations de session"""
    now = datetime.now(timezone.utc)
    sessions = []
    
    # Detecter les sessions
    if 1 <= now.hour < 9:
        sessions.append("Asia")
    if 8 <= now.hour < 17:
        sessions.append("London")
    if 13 <= now.hour < 22:
        sessions.append("New York")
    
    if sessions:
        return f"Sessions actives : {', '.join(sessions)}"
    return f"Hors session - Heure actuelle: {now.strftime('%H:%M')} UTC"