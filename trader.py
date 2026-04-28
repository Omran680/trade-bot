from trading_ig import IGService
import os
import time
import traceback
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

USERNAME = os.getenv("IG_USERNAME")
PASSWORD = os.getenv("IG_PASSWORD")
API_KEY = os.getenv("IG_API_KEY")


class Trader:
    """
    IG Markets API wrapper for trading XAU/USD.
    
    FIX: Added get_volume() method that was missing.
    """
    
    def __init__(self):
        self.ig = IGService(USERNAME, PASSWORD, API_KEY, acc_type="DEMO")
        self.ig.create_session()
        
        # Streaming attributes
        self._stream = None
        self._last_price = None
        self._last_volume = None
        self._use_stream = False
        logger.info("Trader initialized")

    def enable_streaming(self, epic=None):
        """
        Enable market streaming if available.
        
        This sets a flag and starts a background listener when possible.
        """
        try:
            # Lazy import of streaming client (optional dependency)
            from trading_ig import StreamingClient

            def _on_price(data):
                """Handle streaming price updates."""
                try:
                    if data and 'snapshot' in data:
                        snapshot = data['snapshot']
                        if 'bid' in snapshot:
                            self._last_price = float(snapshot['bid'])
                        if 'openInterest' in snapshot:
                            self._last_volume = float(snapshot['openInterest'])
                except Exception as e:
                    logger.debug(f"Error processing price update: {e}")

            # Create and start streaming client
            self._stream = StreamingClient(USERNAME, PASSWORD, API_KEY, acc_type='DEMO')
            self._stream.connect()
            
            # Subscribe to epic updates if provided
            try:
                if epic is not None:
                    # Try different method names for compatibility
                    try:
                        self._stream.subscribe_epic(epic)
                    except Exception:
                        try:
                            self._stream.subscribe(epic)
                        except Exception:
                            pass
            except Exception as e:
                logger.debug(f"Could not subscribe to epic: {e}")

            # Hook handler if available
            try:
                self._stream.on_price = _on_price
            except Exception:
                pass

            self._use_stream = True
            logger.info('Market streaming enabled')
            
        except Exception as e:
            logger.warning(f'Streaming not available: {e}')
            self._use_stream = False

    def get_price(self, epic):
        """
        Fetch current price with retries and fallback.
        
        Returns:
            Current bid price as float
            
        Raises:
            Exception: If price cannot be fetched after retries
        """
        # If streaming is enabled and has a recent price, use it
        if self._use_stream and self._last_price is not None:
            return self._last_price

        last_exc = None
        for attempt in range(1, 4):
            try:
                data = self.ig.fetch_market_by_epic(epic)
                
                # Defensive: ensure snapshot and bid exist
                if not data or "snapshot" not in data:
                    raise ValueError(f"Invalid market data structure: missing snapshot")
                
                snapshot = data["snapshot"]
                if "bid" not in snapshot:
                    raise ValueError(f"Invalid snapshot structure: missing bid")
                
                price = float(snapshot["bid"])
                self._last_price = price
                return price
                
            except Exception as e:
                last_exc = e
                logger.warning(f"Error fetching price (attempt {attempt}/3): {repr(e)}")
                
                # If rate limit, backoff longer
                if hasattr(e, '__class__') and 'Exceed' in e.__class__.__name__:
                    backoff = 10 * attempt
                else:
                    backoff = 1 * attempt
                
                if attempt < 3:
                    time.sleep(backoff)

        # After retries, re-raise
        logger.error(f"Failed to fetch price after 3 attempts: {last_exc}")
        raise last_exc

    def get_volume(self, epic):
        """
        Fetch current trading volume.
        
        FIX: This method was missing and caused crashes.
        
        Args:
            epic: Market epic code
        
        Returns:
            Volume as float, or 0.0 if unavailable
        """
        # If streaming is enabled and has recent volume, use it
        if self._use_stream and self._last_volume is not None:
            return max(self._last_volume, 0.0)

        try:
            data = self.ig.fetch_market_by_epic(epic)
            
            # Try different possible field names for volume/open interest
            if not data or "snapshot" not in data:
                logger.debug(f"No snapshot data for {epic}")
                return 0.0
            
            snapshot = data["snapshot"]
            
            # Try common field names
            volume = None
            for field in ['openInterest', 'dealSize', 'volume', 'Volume']:
                if field in snapshot:
                    volume = float(snapshot.get(field, 0))
                    if volume > 0:
                        self._last_volume = volume
                        return volume
            
            logger.debug(f"Volume field not found in snapshot for {epic}")
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error fetching volume: {repr(e)}")
            return 0.0

    def get_account_balance(self):
        """Get account balance and account information."""
        try:
            return self.ig.fetch_accounts()
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            raise

    def get_positions(self):
        """
        Get current open positions with safe fallback.
        
        Returns:
            List of positions or empty list on error
        """
        try:
            positions = self.ig.fetch_open_positions()
            return positions if positions is not None else []
        except Exception as e:
            logger.error(f"Error fetching positions: {repr(e)}")
            return []

    def close_trade(self, deal_id):
        """
        Close a specific position.
        
        Args:
            deal_id: Position deal ID
        
        Returns:
            Result from IG API
        """
        try:
            logger.info(f"Closing position {deal_id}")
            result = self.ig.close_open_position(deal_id=deal_id)
            logger.info(f"Position closed: {result}")
            return result
        except Exception as e:
            logger.error(f"Error closing position {deal_id}: {e}")
            raise

    def open_trade(self, epic, direction, size, sl=None, tp=None):
        """
        Open a new trade (position).
        
        Args:
            epic: Market epic code
            direction: 'BUY' or 'SELL'
            size: Position size in lots
            sl: Stop loss distance (absolute points)
            tp: Take profit distance (absolute points)
        
        Returns:
            dict: Result from IGService.create_open_position
            
        Raises:
            Exception: If trade creation fails
        """
        try:
            logger.info(f"Opening {direction} trade: {epic} size={size} sl={sl} tp={tp}")
            
            result = self.ig.create_open_position(
                currency_code="USD",
                direction=direction,
                epic=epic,
                expiry="-",  # OTC market, no expiry
                force_open=True,
                guaranteed_stop=False,
                level=None,
                limit_distance=tp,
                limit_level=None,
                order_type="MARKET",
                quote_id=None,
                size=size,
                stop_distance=sl,
                stop_level=None,
                trailing_stop=False,
                trailing_stop_increment=None,
                time_in_force="FILL_OR_KILL",
            )
            
            logger.info(f"Trade opened successfully: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error opening trade: {repr(e)}")
            traceback.print_exc()
            raise

    def get_market_info(self, epic):
        """
        Get detailed market information.
        
        Args:
            epic: Market epic code
        
        Returns:
            dict: Market snapshot data
        """
        try:
            data = self.ig.fetch_market_by_epic(epic)
            if data and "snapshot" in data:
                return data["snapshot"]
            return {}
        except Exception as e:
            logger.error(f"Error fetching market info: {e}")
            return {}

    def is_market_open(self, epic):
        """
        Check if market is currently open for trading.
        
        Args:
            epic: Market epic code
        
        Returns:
            bool: True if market is open
        """
        try:
            info = self.get_market_info(epic)
            # Check if market is tradeable
            return info.get('marketStatus') in ['TRADEABLE', 'EDITS_ONLY']
        except Exception as e:
            logger.warning(f"Could not determine market status: {e}")
            return False