from trading_ig import IGService
import os
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv("IG_USERNAME")
PASSWORD = os.getenv("IG_PASSWORD")
API_KEY = os.getenv("IG_API_KEY")


class Trader:
    def __init__(self):
        self.ig = IGService(USERNAME, PASSWORD, API_KEY, acc_type="DEMO")
        self.ig.create_session()

    def get_price(self, epic):
        data = self.ig.fetch_market_by_epic(epic)
        return float(data["snapshot"]["bid"])

    def get_account_balance(self):
        """Get account balance"""
        return self.ig.fetch_accounts()

    def get_positions(self):
        """Get current positions"""
        return self.ig.fetch_open_positions()

    def close_position(self, deal_id):
        """Close a position"""
        return self.ig.close_open_position(deal_id=deal_id)