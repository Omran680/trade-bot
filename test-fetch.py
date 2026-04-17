import os
import json
import traceback
from dotenv import load_dotenv
from trading_ig import IGService

load_dotenv()

USERNAME = os.getenv("IG_USERNAME")
PASSWORD = os.getenv("IG_PASSWORD")
API_KEY = os.getenv("IG_API_KEY")
EPIC = "CS.D.IN_GOLD.MFI.IP"

print(f"Loaded env: USERNAME={'set' if USERNAME else 'MISSING'}, API_KEY={'set' if API_KEY else 'MISSING'}")

try:
    if not USERNAME or not PASSWORD or not API_KEY:
        raise RuntimeError("Missing IG credentials in environment")

    ig = IGService(USERNAME, PASSWORD, API_KEY, acc_type="DEMO")
    ig.create_session()
    data = ig.fetch_market_by_epic(EPIC)
    print("RAW RESPONSE:", json.dumps(data, indent=2))
except Exception as e:
    print("EXCEPTION:", repr(e))
    traceback.print_exc()