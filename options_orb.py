import pandas as pd
import numpy as np
from datetime import datetime, time
import time as sleep_time
import logging
from typing import Dict, List, Optional, Tuple
try:
    from smartapi import SmartConnect  # pip package: smartapi_python
except Exception:  # ImportError, ModuleNotFoundError
    # Fallback to local package in this repo
    import os, sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from SmartApi import SmartConnect

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptionsStrategyTrader:
    def __init__(self, api_key: str, client_code: str, password: str, totp_secret: str, capital: float = 100000):
        """
        Initialize the Options Strategy Trader with Angel One SmartAPI
        
        Args:
            api_key: Your Angel One API key
            client_code: Your Angel One client code
            password: Your Angel One password
            totp_secret: Your TOTP secret for 2FA
            capital: Available trading capital
        """
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp_secret = totp_secret
        self.capital = capital
        self.risk_per_trade = 0.025  # 2.5% risk per trade
        self.max_daily_trades = 3
        self.daily_trades_count = 0
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        
        # Initialize SmartConnect
        self.smart_api = SmartConnect(api_key=self.api_key)
        self.is_logged_in = False
        
        # Market timing
        self.market_open = time(9, 15)
        self.observation_start = time(9, 15)
        self.observation_end = time(9, 30)
        self.entry_time = time(9, 32)
        self.exit_time = time(15, 25)
        
        # Trading data
        self.observation_high = 0
        self.observation_low = 0
        self.current_positions = []
        self.daily_pnl = 0
        
        # NIFTY token for Angel One
        self.nifty_token = "99926000"  # NIFTY 50 token
        self.india_vix_token = "99926009"  # India VIX token
        
    def login_to_angel_one(self) -> bool:
        """Login to Angel One using SmartAPI"""
        try:
            import pyotp
            
            # Generate TOTP
            totp = pyotp.TOTP(self.totp_secret).now()
            
            # Login
            data = self.smart_api.generateSession(self.client_code, self.password, totp)
            
            if data['status']:
                self.is_logged_in = True
                logger.info("Successfully logged into Angel One")
                return True
            else:
                logger.error(f"Login failed: {data.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
    
    def get_market_data(self, symbol: str = "NIFTY", token: str = None) -> Dict:
        """Fetch live market data from Angel One"""
        if not self.is_logged_in:
            if not self.login_to_angel_one():
                return {}
        
        try:
            # Use NIFTY token if not provided
            if not token:
                token = self.nifty_token
            
            # Get LTP data (this includes OHLC data in Angel One API)
            ltp_data = self.smart_api.ltpData("NSE", symbol, token)
            
            if ltp_data['status']:
                data = ltp_data['data']
                
                return {
                    'ltp': data.get('ltp', 0),
                    'open': data.get('open', 0),
                    'high': data.get('high', 0),
                    'low': data.get('low', 0),
                    'volume': data.get('volume', 0),
                    'prev_close': data.get('close', 0)
                }
            else:
                logger.error(f"LTP data fetch failed for {symbol}: {ltp_data.get('message', 'Unknown error')}")
                return {}
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}
    
    def get_vix_data(self) -> float:
        """Get India VIX value from Angel One"""
        try:
            vix_data = self.smart_api.ltpData("NSE", "INDIA VIX", self.india_vix_token)
            
            if vix_data['status']:
                return vix_data['data'].get('ltp', 0)
            else:
                logger.error("Failed to fetch VIX data")
                return 0
                
        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            return 0
    
    def calculate_gap_percentage(self, current_price: float, prev_close: float) -> float:
        """Calculate opening gap percentage"""
        if prev_close == 0:
            return 0
        return abs((current_price - prev_close) / prev_close) * 100
    
    def check_volume_condition(self, current_volume: float, avg_volume: float) -> bool:
        """Check if current volume > 120% of 10-day average"""
        if avg_volume == 0:
            return False
        return current_volume > (avg_volume * 1.2)
    
    def check_filters(self, market_data: Dict, avg_volume: float) -> Tuple[int, List[str]]:
        """
        Check all filters and return score
        Returns: (score, passed_filters)
        """
        passed_filters = []
        score = 0
        
        # Filter 1: Gap Check (>0.2%)
        gap_pct = self.calculate_gap_percentage(
            market_data.get('open', 0), 
            market_data.get('prev_close', 0)
        )
        if gap_pct > 0.2:
            score += 1
            passed_filters.append("Gap > 0.2%")
        
        # Filter 2: VIX > 18
        vix = self.get_vix_data()
        if vix > 11:
            score += 1
            passed_filters.append("VIX > 11")
        
        # Filter 3: Volume > 120% of average
        if self.check_volume_condition(market_data.get('volume', 0), avg_volume):
            score += 1
            passed_filters.append("Volume > 120%")
        
        return score, passed_filters
    
    def calculate_breakout_levels(self) -> Tuple[float, float]:
        """Calculate bullish and bearish breakout levels"""
        bullish_level = self.observation_high * 1.0025  # +0.25%
        bearish_level = self.observation_low * 0.9975   # -0.25%
        return bullish_level, bearish_level
    
    def get_option_chain(self, symbol: str = "NIFTY") -> List[Dict]:
        """Fetch option chain data from Angel One"""
        if not self.is_logged_in:
            if not self.login_to_angel_one():
                return []
        
        try:
            # Get current expiry Thursday
            current_date = datetime.now()
            days_until_thursday = (3 - current_date.weekday()) % 7
            if days_until_thursday == 0 and current_date.weekday() == 3:  # If today is Thursday
                days_until_thursday = 7  # Get next Thursday
            
            expiry_date = current_date + pd.Timedelta(days=days_until_thursday)
            expiry_str = expiry_date.strftime('%d%b%Y').upper()
            
            # Get option chain using Angel One's search API
            search_data = self.smart_api.searchScrip("NFO", f"{symbol}{expiry_str}")
            
            if search_data['status']:
                options = []
                for scrip in search_data['data']:
                    symbol_name = scrip.get('symbol', '')
                    if 'CE' in symbol_name or 'PE' in symbol_name:
                        # Get LTP for this option
                        token = scrip.get('symboltoken', '')
                        ltp_data = self.smart_api.ltpData("NFO", symbol_name, token)
                        
                        if ltp_data['status']:
                            premium = ltp_data['data'].get('ltp', 0)
                            
                            # Extract strike price
                            try:
                                if 'CE' in symbol_name:
                                    strike = float(symbol_name.split('CE')[0].split(expiry_str)[1])
                                    option_type = 'CE'
                                else:
                                    strike = float(symbol_name.split('PE')[0].split(expiry_str)[1])
                                    option_type = 'PE'
                                
                                options.append({
                                    'symbol': symbol_name,
                                    'token': token,
                                    'strike': strike,
                                    'premium': premium,
                                    'option_type': option_type,
                                    'expiry': expiry_str
                                })
                            except:
                                continue
                
                return options
            else:
                logger.error("Failed to fetch option chain")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching option chain: {e}")
            return []
    
    def select_option(self, option_type: str, current_price: float) -> Optional[Dict]:
        """
        Select appropriate option based on criteria
        
        Args:
            option_type: 'CE' for calls, 'PE' for puts
            current_price: Current underlying price
        """
        options = self.get_option_chain()
        
        suitable_options = []
        for option in options:
            # Check premium range (200-300)
            premium = option.get('premium', 0)
            if 200 <= premium <= 300:
                
                # Check if ATM or 1 OTM
                strike = option.get('strike', 0)
                if option_type == 'CE':
                    if strike >= current_price and strike <= current_price + 100:
                        suitable_options.append(option)
                elif option_type == 'PE':
                    if strike <= current_price and strike >= current_price - 100:
                        suitable_options.append(option)
        
        # Select the best option (closest to ATM with good liquidity)
        if suitable_options:
            # Sort by liquidity (bid-ask spread)
            suitable_options.sort(key=lambda x: x.get('bid_ask_spread', 999))
            return suitable_options[0]
        
        return None
    
    def calculate_position_size(self, premium: float) -> int:
        """Calculate number of lots based on risk management"""
        # Start with one lot only for conservative approach
        return 1
    
    def place_order(self, symbol: str, token: str, quantity: int, price: float, transaction_type: str = "BUY") -> Dict:
        """
        Place order using Angel One SmartAPI
        
        Args:
            symbol: Option symbol
            token: Option token
            quantity: Number of lots (will be multiplied by lot size)
            price: Limit price
            transaction_type: BUY or SELL
        """
        if not self.is_logged_in:
            if not self.login_to_angel_one():
                return {}
        
        try:
            # NIFTY lot size is 75
            lot_size = 75
            total_quantity = quantity * lot_size
            
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": symbol,
                "symboltoken": token,
                "transactiontype": transaction_type,
                "exchange": "NFO",
                "ordertype": "LIMIT",
                "producttype": "INTRADAY",
                "duration": "DAY",
                "price": str(price),
                "squareoff": "0",
                "stoploss": "0",
                "quantity": str(total_quantity)
            }
            
            order_result = self.smart_api.placeOrder(order_params)
            
            if order_result['status']:
                logger.info(f"Order placed successfully: {symbol} {transaction_type} {quantity} lots at ₹{price}")
                return {
                    'status': True,
                    'order_id': order_result['data']['orderid'],
                    'message': order_result['message']
                }
            else:
                logger.error(f"Order placement failed: {order_result.get('message', 'Unknown error')}")
                return {'status': False, 'message': order_result.get('message', 'Order failed')}
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {'status': False, 'message': str(e)}
    
    def monitor_positions(self):
        """Monitor current positions and manage stops/targets using Angel One API"""
        for position in self.current_positions:
            try:
                # Get current option price
                ltp_data = self.smart_api.ltpData("NFO", position['symbol'], position['token'])
                
                if not ltp_data['status']:
                    logger.error(f"Failed to get LTP for {position['symbol']}")
                    continue
                
                current_price = ltp_data['data'].get('ltp', 0)
                entry_price = position['entry_price']
                quantity = position['quantity']
                
                # Calculate P&L
                pnl = (current_price - entry_price) * quantity * 75  # 75 is NIFTY lot size
                pnl_pct = (current_price - entry_price) / entry_price * 100
                
                logger.info(f"Position {position['symbol']}: Entry=₹{entry_price}, Current=₹{current_price}, P&L=₹{pnl:.2f} ({pnl_pct:.1f}%)")
                
                # Check stop loss (10% loss or profit lock)
                stop_loss_price = entry_price * 0.90  # Default 10% stop loss
                
                # 10% profit lock - move stop loss to lock 10% profit once 15% profit achieved
                if current_price >= entry_price * 1.15 and not position.get('profit_locked', False):
                    position['profit_locked'] = True
                    position['stop_loss_price'] = entry_price * 1.10  # Lock 10% profit
                    logger.info(f"15% profit achieved for {position['symbol']} - stop loss moved to lock 10% profit at ₹{entry_price * 1.10}")
                
                # Use updated stop loss price if profit is locked
                if position.get('profit_locked', False):
                    stop_loss_price = position.get('stop_loss_price', entry_price)
                
                if current_price <= stop_loss_price:
                    if position.get('profit_locked', False):
                        logger.info(f"10% profit lock stop loss triggered for {position['symbol']} at ₹{current_price}")
                    else:
                        logger.info(f"10% stop loss triggered for {position['symbol']} at ₹{current_price}")
                    
                    remaining_quantity = position.get('quantity', quantity)
                    order_result = self.place_order(
                        position['symbol'], 
                        position['token'],
                        remaining_quantity, 
                        current_price, 
                        "SELL"
                    )
                    if order_result.get('status'):
                        self.current_positions.remove(position)
                        final_pnl = (current_price - entry_price) * remaining_quantity * 75
                        self.daily_pnl += final_pnl
                    
                # Handle 50% profit booking based on position size
                elif current_price >= entry_price * 1.50 and not position.get('partial_booked', False):
                    if quantity == 1:
                        # For 1 lot: Close full position at 50% profit
                        logger.info(f"50% profit target reached for {position['symbol']} (1 lot) - closing full position")
                        order_result = self.place_order(
                            position['symbol'],
                            position['token'], 
                            quantity, 
                            current_price, 
                            "SELL"
                        )
                        if order_result.get('status'):
                            self.current_positions.remove(position)
                            full_pnl = (current_price - entry_price) * quantity * 75
                            self.daily_pnl += full_pnl
                            logger.info(f"50% profit achieved - full position closed, P&L: ₹{full_pnl:.2f}")
                    else:
                        # For 2+ lots: Book 50% of position at 50% profit
                        half_quantity = quantity // 2
                        logger.info(f"50% profit target reached for {position['symbol']} ({quantity} lots) - booking {half_quantity} lots")
                        order_result = self.place_order(
                            position['symbol'],
                            position['token'], 
                            half_quantity, 
                            current_price, 
                            "SELL"
                        )
                        if order_result.get('status'):
                            position['quantity'] = quantity - half_quantity
                            position['partial_booked'] = True
                            partial_pnl = (current_price - entry_price) * half_quantity * 75
                            self.daily_pnl += partial_pnl
                            logger.info(f"Partial booking: {half_quantity} lots sold, P&L: ₹{partial_pnl:.2f}, Remaining: {position['quantity']} lots")
                
                # Check profit target (100% gain) - sell remaining position (only for 2+ lots)
                elif current_price >= entry_price * 2.0 and quantity > 1:
                    logger.info(f"100% profit target reached for {position['symbol']} - closing remaining position")
                    remaining_quantity = position['quantity']
                    order_result = self.place_order(
                        position['symbol'],
                        position['token'], 
                        remaining_quantity, 
                        current_price, 
                        "SELL"
                    )
                    if order_result.get('status'):
                        self.current_positions.remove(position)
                        remaining_pnl = (current_price - entry_price) * remaining_quantity * 75
                        self.daily_pnl += remaining_pnl
                        logger.info(f"100% target achieved - remaining {remaining_quantity} lots closed, P&L: ₹{remaining_pnl:.2f}")
                    
                # Trailing stop logic: for every 10 points profit, trail by 5 points
                elif current_price > entry_price:
                    profit_points = current_price - entry_price
                    ten_point_moves = int(profit_points / 10)
                    
                    if ten_point_moves > 0:
                        # Calculate trailing stop: entry price + (moves * 5 points trail)
                        trailing_stop_price = entry_price + (ten_point_moves * 5)
                        
                        # Update trailing stop if it's higher than current
                        if 'trailing_stop' not in position or trailing_stop_price > position.get('trailing_stop', 0):
                            position['trailing_stop'] = trailing_stop_price
                            logger.info(f"Updated trailing stop for {position['symbol']}: ₹{trailing_stop_price} (after {ten_point_moves} x 10pt moves)")
                        
                        # Check if trailing stop is hit
                        if current_price <= position.get('trailing_stop', 0):
                            logger.info(f"Trailing stop hit for {position['symbol']} at ₹{current_price}")
                            remaining_quantity = position['quantity']
                            order_result = self.place_order(
                                position['symbol'],
                                position['token'],
                                remaining_quantity,
                                current_price,
                                "SELL"
                            )
                            if order_result.get('status'):
                                self.current_positions.remove(position)
                                remaining_pnl = (current_price - entry_price) * remaining_quantity * 75
                                self.daily_pnl += remaining_pnl
                
            except Exception as e:
                logger.error(f"Error monitoring position {position.get('symbol', 'Unknown')}: {e}")
    
    def get_historical_data(self, days: int = 10) -> float:
        """Get average volume for filter calculation"""
        try:
            # Get historical data for volume calculation
            from_date = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d %H:%M')
            to_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            
            hist_data = self.smart_api.getCandleData({
                "exchange": "NSE",
                "symboltoken": self.nifty_token,
                "interval": "ONE_DAY",
                "fromdate": from_date,
                "todate": to_date
            })
            
            if hist_data['status'] and hist_data['data']:
                volumes = [float(candle[5]) for candle in hist_data['data']]  # Volume is at index 5
                return sum(volumes) / len(volumes) if volumes else 1000000
            else:
                return 1000000  # Default fallback
                
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return 1000000  # Default fallback
    
    def run_strategy(self):
        """Main strategy execution loop for Angel One"""
        logger.info("Starting Options Momentum Strategy with Angel One SmartAPI")
        
        # Login to Angel One
        if not self.login_to_angel_one():
            logger.error("Failed to login to Angel One. Exiting.")
            return
        
        # Get 10-day average volume
        avg_volume = self.get_historical_data(10)
        logger.info(f"10-day average volume: {avg_volume:,.0f}")
        
        while True:
            try:
                current_time = datetime.now().time()
                
                # Pre-market checks
                if current_time < self.market_open:
                    logger.info("Market not open yet. Waiting...")
                    sleep_time.sleep(60)
                    continue
                
                # Exit all positions before market close
                if current_time >= self.exit_time:
                    logger.info("Market closing soon. Exiting all positions...")
                    for position in self.current_positions:
                        self.place_order(
                            position['symbol'], 
                            position['token'],
                            position['quantity'], 
                            0, 
                            "SELL"
                        )
                    break
                
                # Observation phase (9:15 - 9:30)
                if self.observation_start <= current_time <= self.observation_end:
                    market_data = self.get_market_data("NIFTY 50", self.nifty_token)
                    if market_data:
                        self.observation_high = max(self.observation_high, market_data['high'])
                        self.observation_low = min(self.observation_low or 999999, market_data['low'])
                        logger.info(f"Observation - High: {self.observation_high}, Low: {self.observation_low}")
                
                # Trading phase (9:32 onwards)
                elif current_time >= self.entry_time:
                    # Check daily limits
                    if self.daily_trades_count >= self.max_daily_trades:
                        logger.info("Daily trade limit reached")
                        sleep_time.sleep(180)  # Check every 3 minutes
                        continue
                    
                    if abs(self.daily_pnl) >= self.capital * self.daily_loss_limit:
                        logger.info("Daily loss limit reached")
                        break
                    
                    # Get current market data
                    market_data = self.get_market_data("NIFTY 50", self.nifty_token)
                    if not market_data:
                        sleep_time.sleep(180)
                        continue
                    
                    current_price = market_data['ltp']
                    
                    # Check filters
                    filter_score, passed_filters = self.check_filters(market_data, avg_volume)
                    logger.info(f"Filter Score: {filter_score}/3, Passed: {passed_filters}")
                    
                    # Need at least 2/3 filters and no current positions
                    if filter_score >= 2 and len(self.current_positions) == 0:
                        bullish_level, bearish_level = self.calculate_breakout_levels()
                        
                        # Check for breakout
                        if current_price > bullish_level:
                            # Bullish breakout - Buy Call
                            logger.info(f"Bullish breakout detected: {current_price} > {bullish_level}")
                            option = self.select_option('CE', current_price)
                            
                            if option:
                                lots = self.calculate_position_size(option['premium'])
                                order_result = self.place_order(
                                    option['symbol'], 
                                    option['token'],
                                    lots, 
                                    option['premium']
                                )
                                
                                if order_result.get('status'):
                                    position = {
                                        'symbol': option['symbol'],
                                        'token': option['token'],
                                        'entry_price': option['premium'],
                                        'quantity': lots,
                                        'type': 'CE',
                                        'entry_time': current_time,
                                        'order_id': order_result.get('order_id')
                                    }
                                    self.current_positions.append(position)
                                    self.daily_trades_count += 1
                                    logger.info(f"Entered bullish position: {lots} lots of {option['symbol']}")
                        
                        elif current_price < bearish_level:
                            # Bearish breakout - Buy Put
                            logger.info(f"Bearish breakout detected: {current_price} < {bearish_level}")
                            option = self.select_option('PE', current_price)
                            
                            if option:
                                lots = self.calculate_position_size(option['premium'])
                                order_result = self.place_order(
                                    option['symbol'],
                                    option['token'], 
                                    lots, 
                                    option['premium']
                                )
                                
                                if order_result.get('status'):
                                    position = {
                                        'symbol': option['symbol'],
                                        'token': option['token'],
                                        'entry_price': option['premium'],
                                        'quantity': lots,
                                        'type': 'PE',
                                        'entry_time': current_time,
                                        'order_id': order_result.get('order_id')
                                    }
                                    self.current_positions.append(position)
                                    self.daily_trades_count += 1
                                    logger.info(f"Entered bearish position: {lots} lots of {option['symbol']}")
                
                # Monitor existing positions every 3 minutes
                if self.current_positions:
                    self.monitor_positions()
                
                # Wait 3 minutes before next check
                sleep_time.sleep(180)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                sleep_time.sleep(60)

# Usage Example for Angel One
if __name__ == "__main__":
    # Angel One API credentials
    API_KEY = "FkpWlaE9"    
    CLIENT_CODE = "SPHOA1034"  
    PASSWORD = "0509"
    TOTP_SECRET = "7CWRXGUI2AB364N43NGHQNNHJY"  # From Angel One app QR code
    CAPITAL = 100000  # ₹1 lakh
    
    print("Angel One Options Momentum Strategy")
    print("=" * 50)
    print("⚠️  IMPORTANT WARNINGS:")
    print("1. This is educational code - Test in paper trading first!")
    print("2. Options trading involves substantial risk of loss")
    print("3. Only trade with money you can afford to lose")
    print("4. Ensure you have sufficient margin (recommended ₹75K+)")
    print("5. Install required packages: pip install smartapi pyotp pandas")
    print("=" * 50)
    
    # Confirm before running
    confirm = input("\nHave you tested this in paper trading? (yes/no): ").lower()
    if confirm != 'yes':
        print("Please test in paper trading first. Exiting...")
        exit()
    
    # Initialize and run strategy
    try:
        trader = OptionsStrategyTrader(API_KEY, CLIENT_CODE, PASSWORD, TOTP_SECRET, CAPITAL)
        trader.run_strategy()
    except KeyboardInterrupt:
        print("\nStrategy stopped by user.")
    except Exception as e:
        print(f"Strategy error: {e}")
        
    print("Strategy execution completed.")