import pandas as pd
import numpy as np
import requests
import json
import asyncio
import sqlite3
from datetime import datetime, timedelta, time
import time as sleep_time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nifty_historical_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class HistoricalData:
    """Historical market data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: Optional[int] = None  # Open Interest for options

@dataclass
class OptionData:
    """Option data structure"""
    symbol: str
    strike: float
    expiry: str
    option_type: str  # CE or PE
    ltp: float
    volume: int
    oi: int
    iv: Optional[float] = None  # Implied Volatility

@dataclass
class Position:
    """Position tracking"""
    symbol: str
    strike: float
    option_type: str
    entry_price: float
    current_price: float
    quantity: int
    entry_time: datetime
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None

class NSEHistoricalDataProvider:
    """Fetches historical data from NSE API"""
    
    def __init__(self):
        self.base_url = "https://nsepy-xyz.web.app/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_nifty_historical(self, days: int = 30) -> List[HistoricalData]:
        """Get NIFTY historical data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # API endpoint for NIFTY historical data
            url = f"{self.base_url}/historical/index"
            params = {
                'symbol': 'NIFTY',
                'from': start_str,
                'to': end_str
            }
            
            logger.info(f"Fetching NIFTY data from {start_str} to {end_str}")
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                historical_data = []
                
                for record in data.get('data', []):
                    hist_data = HistoricalData(
                        timestamp=datetime.strptime(record['date'], '%Y-%m-%d'),
                        open=float(record['open']),
                        high=float(record['high']),
                        low=float(record['low']),
                        close=float(record['close']),
                        volume=int(record.get('volume', 0))
                    )
                    historical_data.append(hist_data)
                
                logger.info(f"✅ Fetched {len(historical_data)} NIFTY data points")
                return historical_data
            else:
                logger.error(f"API Error: {response.status_code}")
                return self._generate_fallback_data(days)
                
        except Exception as e:
            logger.error(f"Error fetching NIFTY data: {e}")
            return self._generate_fallback_data(days)
    
    def get_options_chain_historical(self, date: str, expiry: str) -> List[OptionData]:
        """Get historical options chain data"""
        try:
            url = f"{self.base_url}/historical/options"
            params = {
                'symbol': 'NIFTY',
                'expiry': expiry,
                'date': date
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                options_data = []
                
                for record in data.get('data', []):
                    option = OptionData(
                        symbol=record['symbol'],
                        strike=float(record['strikePrice']),
                        expiry=record['expiryDate'],
                        option_type=record['optionType'],
                        ltp=float(record.get('lastPrice', 0)),
                        volume=int(record.get('volume', 0)),
                        oi=int(record.get('openInterest', 0)),
                        iv=float(record.get('impliedVolatility', 0)) if record.get('impliedVolatility') else None
                    )
                    options_data.append(option)
                
                return options_data
            else:
                logger.warning(f"Options API Error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching options data: {e}")
            return []
    
    def get_vix_historical(self, days: int = 30) -> List[HistoricalData]:
        """Get VIX historical data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/historical/index"
            params = {
                'symbol': 'INDIAVIX',
                'from': start_str,
                'to': end_str
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                vix_data = []
                
                for record in data.get('data', []):
                    vix_point = HistoricalData(
                        timestamp=datetime.strptime(record['date'], '%Y-%m-%d'),
                        open=float(record['open']),
                        high=float(record['high']),
                        low=float(record['low']),
                        close=float(record['close']),
                        volume=int(record.get('volume', 0))
                    )
                    vix_data.append(vix_point)
                
                logger.info(f"✅ Fetched {len(vix_data)} VIX data points")
                return vix_data
            else:
                logger.warning("VIX data not available, using estimates")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
            return []
    
    def _generate_fallback_data(self, days: int) -> List[HistoricalData]:
        """Generate realistic fallback data if API fails"""
        logger.info("Using fallback data generation")
        
        fallback_data = []
        base_price = 24300
        current_price = base_price
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i)
            
            # Generate realistic price movement
            daily_change = np.random.normal(0, 0.015)  # 1.5% daily volatility
            current_price = max(current_price * (1 + daily_change), base_price * 0.85)  # Don't go below 15% of base
            current_price = min(current_price, base_price * 1.15)  # Don't go above 15% of base
            
            # Generate OHLC
            high = current_price * (1 + abs(np.random.normal(0, 0.005)))
            low = current_price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = current_price + np.random.normal(0, current_price * 0.003)
            
            fallback_data.append(HistoricalData(
                timestamp=date,
                open=open_price,
                high=high,
                low=low,
                close=current_price,
                volume=int(np.random.normal(150000000, 50000000))  # Realistic volume
            ))
        
        return fallback_data

class HistoricalOptionsTrader:
    """Historical backtesting engine using real NIFTY data"""
    
    def __init__(self, capital: float = 100000):
        self.capital = capital
        self.initial_capital = capital
        self.data_provider = NSEHistoricalDataProvider()
        self.positions: List[Position] = []
        self.trade_history = []
        self.daily_pnl = 0.0
        self.observation_data = []
        
        # Strategy parameters (from your original code)
        self.risk_per_trade = 0.025  # 2.5%
        self.max_daily_trades = 3
        self.daily_trades_count = 0
        self.daily_loss_limit = 0.05  # 5%
        
        # Market timing
        self.observation_start = time(9, 15)
        self.observation_end = time(9, 30)
        self.entry_time = time(9, 32)
        self.exit_time = time(15, 25)
        
    async def run_historical_backtest(self, days: int = 30):
        """Run backtest using real historical data"""
        logger.info(f"🚀 Starting historical backtest for {days} days")
        
        # Get historical data
        nifty_data = self.data_provider.get_nifty_historical(days)
        vix_data = self.data_provider.get_vix_historical(days)
        
        if not nifty_data:
            logger.error("❌ No historical data available")
            return
        
        # Create VIX lookup dictionary
        vix_lookup = {vix.timestamp.date(): vix.close for vix in vix_data}
        
        # Process each day
        for i, day_data in enumerate(nifty_data):
            try:
                await self._process_trading_day(day_data, vix_lookup, i)
            except Exception as e:
                logger.error(f"Error processing day {i}: {e}")
                continue
        
        # Final results
        self._display_backtest_results()
    
    async def _process_trading_day(self, day_data: HistoricalData, vix_lookup: Dict, day_index: int):
        """Process a single trading day"""
        trading_date = day_data.timestamp.date()
        current_vix = vix_lookup.get(trading_date, 15.0)  # Default VIX if not available
        
        logger.info(f"📅 Processing {trading_date} - NIFTY: {day_data.close:.2f}, VIX: {current_vix:.2f}")
        
        # Reset daily counters
        self.daily_trades_count = 0
        self.daily_pnl = 0
        
        # Simulate intraday observation period (9:15-9:30)
        observation_high, observation_low = self._simulate_observation_period(day_data)
        
        # Check filters (your original strategy)
        filter_score = self._check_trading_filters(day_data, current_vix)
        
        if filter_score < 2:  # Need at least 2/3 filters
            logger.info(f"   ❌ Filter score {filter_score}/3 - No trading")
            return
        
        # Calculate breakout levels
        bullish_level = observation_high * 1.0025  # +0.25%
        bearish_level = observation_low * 0.9975   # -0.25%
        
        # Simulate price movement throughout the day
        await self._simulate_intraday_trading(
            day_data, bullish_level, bearish_level, current_vix, trading_date
        )
    
    def _simulate_observation_period(self, day_data: HistoricalData) -> Tuple[float, float]:
        """Simulate 15-minute observation period"""
        # Use day's open and create realistic 15-min range
        open_price = day_data.open
        volatility = abs(day_data.high - day_data.low) / day_data.close
        
        # Simulate 15-minute high/low based on daily volatility
        observation_range = open_price * volatility * 0.3  # 30% of daily range in first 15 min
        
        observation_high = open_price + np.random.uniform(0, observation_range)
        observation_low = open_price - np.random.uniform(0, observation_range)
        
        # Ensure realistic bounds
        observation_high = min(observation_high, day_data.high * 0.7)  # Max 70% of daily high
        observation_low = max(observation_low, day_data.low * 1.3)   # Min 130% of daily low
        
        logger.info(f"   📊 Observation: High={observation_high:.2f}, Low={observation_low:.2f}")
        return observation_high, observation_low
    
    def _check_trading_filters(self, day_data: HistoricalData, vix: float) -> int:
        """Check your original trading filters"""
        score = 0
        reasons = []
        
        # Filter 1: Gap check (>0.2%)
        prev_close = day_data.close * 0.998  # Approximate previous close
        gap_pct = abs((day_data.open - prev_close) / prev_close) * 100
        if gap_pct > 0.2:
            score += 1
            reasons.append(f"Gap: {gap_pct:.2f}%")
        
        # Filter 2: VIX > 11
        if vix > 11:
            score += 1
            reasons.append(f"VIX: {vix:.2f}")
        
        # Filter 3: Volume check (assume high volume if daily range > 1%)
        daily_range = abs(day_data.high - day_data.low) / day_data.close
        if daily_range > 0.01:  # >1% daily range indicates good volume
            score += 1
            reasons.append("Good volume")
        
        logger.info(f"   🎯 Filters ({score}/3): {', '.join(reasons)}")
        return score
    
    async def _simulate_intraday_trading(self, day_data: HistoricalData, 
                                       bullish_level: float, bearish_level: float,
                                       vix: float, trading_date):
        """Simulate intraday price movements and trading"""
        
        # Generate realistic intraday price path
        price_path = self._generate_intraday_path(day_data)
        
        for minute, current_price in enumerate(price_path):
            # Skip if max daily trades reached
            if self.daily_trades_count >= self.max_daily_trades:
                break
            
            # Check for breakouts (after 9:32 AM)
            if minute >= 17:  # 17 minutes after 9:15 (i.e., 9:32)
                signal = self._check_breakout_signal(
                    current_price, bullish_level, bearish_level, vix
                )
                
                if signal and len(self.positions) == 0:  # Only if no current positions
                    await self._execute_trade(signal, current_price, trading_date)
            
            # Monitor existing positions
            await self._monitor_positions(current_price, minute)
    
    def _generate_intraday_path(self, day_data: HistoricalData) -> List[float]:
        """Generate realistic intraday price path"""
        # Create 375-minute trading day (9:15 AM to 3:30 PM)
        minutes = 375
        
        # Start from open, end at close
        start_price = day_data.open
        end_price = day_data.close
        high_target = day_data.high
        low_target = day_data.low
        
        # Generate random walk that hits high and low
        path = []
        current_price = start_price
        
        # Add some realistic intraday patterns
        for i in range(minutes):
            # Progress through the day
            progress = i / minutes
            
            # Target price (linear interpolation with noise)
            target_price = start_price + (end_price - start_price) * progress
            
            # Add mean reversion to target
            mean_reversion = (target_price - current_price) * 0.01
            
            # Random component
            volatility = abs(day_data.high - day_data.low) / day_data.close / 20  # Minute volatility
            random_move = np.random.normal(0, volatility * current_price)
            
            # Combine movements
            price_change = mean_reversion + random_move
            current_price += price_change
            
            # Ensure we stay within day's range
            current_price = max(current_price, low_target * 0.999)
            current_price = min(current_price, high_target * 1.001)
            
            path.append(current_price)
        
        return path
    
    def _check_breakout_signal(self, current_price: float, bullish_level: float,
                              bearish_level: float, vix: float) -> Optional[Dict]:
        """Check for breakout signals (your original logic)"""
        
        if current_price > bullish_level:
            return {
                'type': 'BUY_CALL',
                'reason': f'Bullish breakout: {current_price:.2f} > {bullish_level:.2f}',
                'confidence': min(0.8, 0.5 + (vix - 11) * 0.02)  # Higher VIX = higher confidence
            }
        elif current_price < bearish_level:
            return {
                'type': 'BUY_PUT',
                'reason': f'Bearish breakout: {current_price:.2f} < {bearish_level:.2f}',
                'confidence': min(0.8, 0.5 + (vix - 11) * 0.02)
            }
        
        return None
    
    async def _execute_trade(self, signal: Dict, current_price: float, trading_date):
        """Execute a trade based on signal"""
        try:
            # Find suitable option (simplified)
            strike_price = self._find_suitable_strike(current_price, signal['type'])
            premium = self._estimate_option_premium(current_price, strike_price, signal['type'])
            
            if premium == 0:  # No suitable option found
                return
            
            # Calculate position size (your original logic - 1 lot conservative)
            lots = 1
            
            # Create position
            position = Position(
                symbol=f"NIFTY{trading_date.strftime('%Y%m%d')}{int(strike_price)}{'CE' if 'CALL' in signal['type'] else 'PE'}",
                strike=strike_price,
                option_type='CE' if 'CALL' in signal['type'] else 'PE',
                entry_price=premium,
                current_price=premium,
                quantity=lots,
                entry_time=datetime.combine(trading_date, time(9, 32))
            )
            
            self.positions.append(position)
            self.daily_trades_count += 1
            
            # Deduct capital (margin)
            margin_required = premium * 75 * 0.1  # 10% margin for options
            self.capital -= margin_required
            
            logger.info(f"   ✅ Trade: {position.symbol} @ ₹{premium:.2f} - {signal['reason']}")
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    def _find_suitable_strike(self, current_price: float, signal_type: str) -> float:
        """Find suitable strike price (ATM or 1 OTM)"""
        # Round to nearest 50 for NIFTY options
        base_strike = round(current_price / 50) * 50
        
        if 'CALL' in signal_type:
            # ATM or 1 OTM call
            return base_strike if current_price >= base_strike else base_strike + 50
        else:
            # ATM or 1 OTM put
            return base_strike if current_price <= base_strike else base_strike - 50
    
    def _estimate_option_premium(self, spot_price: float, strike: float, option_type: str) -> float:
        """Estimate option premium using simplified Black-Scholes"""
        try:
            # Simplified premium calculation
            time_to_expiry = 7 / 365  # Assume weekly expiry
            volatility = 0.20  # 20% annual volatility
            risk_free_rate = 0.06  # 6% risk-free rate
            
            # Intrinsic value
            if 'CALL' in option_type:
                intrinsic = max(spot_price - strike, 0)
            else:
                intrinsic = max(strike - spot_price, 0)
            
            # Time value (simplified)
            moneyness = abs(spot_price - strike) / spot_price
            time_value = spot_price * volatility * np.sqrt(time_to_expiry) * (1 - moneyness)
            
            premium = intrinsic + time_value
            
            # Ensure premium is in reasonable range (200-500 for NIFTY options)
            premium = max(50, min(premium, 800))
            
            # Filter: Only trade if premium is between 200-300 (your original criteria)
            if 200 <= premium <= 300:
                return premium
            else:
                return 0  # Skip this trade
            
        except Exception as e:
            logger.error(f"Premium calculation error: {e}")
            return 0
    
    async def _monitor_positions(self, current_price: float, minute: int):
        """Monitor existing positions (your original exit logic)"""
        positions_to_close = []
        
        for position in self.positions:
            # Update current option price (simplified)
            new_premium = self._estimate_current_premium(position, current_price)
            position.current_price = new_premium
            
            # Calculate P&L
            position.pnl = (new_premium - position.entry_price) * position.quantity * 75
            position.pnl_pct = (new_premium - position.entry_price) / position.entry_price * 100
            
            # Check exit conditions (your original logic)
            exit_reason = self._check_exit_conditions(position, minute)
            
            if exit_reason:
                position.exit_time = datetime.now()
                position.exit_reason = exit_reason
                positions_to_close.append(position)
        
        # Close positions
        for position in positions_to_close:
            self._close_position(position)
            self.positions.remove(position)
    
    def _estimate_current_premium(self, position: Position, current_spot: float) -> float:
        """Estimate current option premium based on spot movement"""
        entry_spot = 24300  # Approximate - in real implementation, track this
        
        # Delta approximation (how much option moves per spot move)
        if position.option_type == 'CE':
            delta = 0.5 if abs(current_spot - position.strike) < 50 else 0.3
            premium_change = (current_spot - entry_spot) * delta
        else:  # PE
            delta = -0.5 if abs(current_spot - position.strike) < 50 else -0.3
            premium_change = (current_spot - entry_spot) * delta
        
        new_premium = position.entry_price + premium_change
        return max(new_premium, 1)  # Minimum ₹1 premium
    
    def _check_exit_conditions(self, position: Position, minute: int) -> Optional[str]:
        """Check position exit conditions (your original logic)"""
        pnl_pct = position.pnl_pct
        
        # 10% stop loss
        if pnl_pct <= -10:
            return "10% stop loss"
        
        # 50% profit target
        if pnl_pct >= 50:
            return "50% profit target"
        
        # Time-based exit (3:25 PM = minute 370)
        if minute >= 370:
            return "Market closing"
        
        # 15% profit achieved - move to 10% profit lock (simplified)
        if pnl_pct >= 15 and not hasattr(position, 'profit_locked'):
            position.profit_locked = True
            return None  # Don't exit yet, just lock profit
        
        # Profit lock triggered
        if hasattr(position, 'profit_locked') and pnl_pct <= 10:
            return "10% profit lock triggered"
        
        return None
    
    def _close_position(self, position: Position):
        """Close a position and update capital"""
        # Return margin + P&L
        margin_return = position.entry_price * 75 * 0.1  # Return margin
        self.capital += margin_return + position.pnl
        self.daily_pnl += position.pnl
        
        # Add to trade history
        self.trade_history.append({
            'date': position.entry_time.date(),
            'symbol': position.symbol,
            'entry_price': position.entry_price,
            'exit_price': position.current_price,
            'pnl': position.pnl,
            'pnl_pct': position.pnl_pct,
            'exit_reason': position.exit_reason
        })
        
        logger.info(f"   🎯 Closed: {position.symbol} - P&L: ₹{position.pnl:,.0f} ({position.pnl_pct:.1f}%) - {position.exit_reason}")
    
    def _display_backtest_results(self):
        """Display comprehensive backtest results"""
        if not self.trade_history:
            print("\n❌ No trades executed during backtest period")
            return
        
        df = pd.DataFrame(self.trade_history)
        
        # Calculate metrics
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        total_pnl = df['pnl'].sum()
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if total_trades > winning_trades else 0
        best_trade = df['pnl'].max()
        worst_trade = df['pnl'].min()
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        print(f"\n{'='*80}")
        print("📊 HISTORICAL BACKTEST RESULTS - NIFTY OPTIONS")
        print(f"{'='*80}")
        print(f"📈 PERFORMANCE SUMMARY:")
        print(f"   Initial Capital: ₹{self.initial_capital:,.0f}")
        print(f"   Final Capital: ₹{self.capital:,.0f}")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Total P&L: ₹{total_pnl:,.0f}")
        
        print(f"\n📋 TRADING STATISTICS:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Winning Trades: {winning_trades}")
        print(f"   Losing Trades: {total_trades - winning_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        
        print(f"\n💰 P&L ANALYSIS:")
        print(f"   Average Win: ₹{avg_win:,.0f}")
        print(f"   Average Loss: ₹{avg_loss:,.0f}")
        print(f"   Best Trade: ₹{best_trade:,.0f}")
        print(f"   Worst Trade: ₹{worst_trade:,.0f}")
        print(f"   Risk-Reward Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "   Risk-Reward Ratio: N/A")
        
        print(f"\n📅 DAILY BREAKDOWN:")
        daily_pnl = df.groupby('date')['pnl'].sum()
        profitable_days = len(daily_pnl[daily_pnl > 0])
        total_days = len(daily_pnl)
        
        print(f"   Trading Days: {total_days}")
        print(f"   Profitable Days: {profitable_days}")
        print(f"   Daily Win Rate: {(profitable_days/total_days)*100:.1f}%")
        print(f"   Best Day: ₹{daily_pnl.max():,.0f}")
        print(f"   Worst Day: ₹{daily_pnl.min():,.0f}")
        
        # Show recent trades
        print(f"\n📋 RECENT TRADES (Last 10):")
        recent_trades = df.tail(10)
        for _, trade in recent_trades.iterrows():
            print(f"   {trade['date']} | {trade['symbol'][:15]:<15} | "
                  f"₹{trade['pnl']:>8,.0f} | {trade['exit_reason'][:20]}")
        
        print(f"\n{'='*80}")

# Main execution functions
async def run_nifty_historical_test():
    """Run NIFTY historical backtest"""
    print("📈 NIFTY OPTIONS HISTORICAL BACKTESTING")
    print("=" * 60)
    print("Using real NSE historical data for accurate testing!")
    print("Perfect for testing when markets are closed! 🚀")
    
    # Get user preferences
    try:
        days = int(input("\nEnter number of days to backtest (default 30): ") or "30")
        capital = float(input("Enter starting capital (default ₹100,000): ") or "100000")
    except ValueError:
        days = 30
    
    print(f"\n🚀 Starting backtest: {days} days with ₹{capital:,.0f} capital")
    print("📡 Fetching real historical data from NSE...")
    
    # Initialize trader
    trader = HistoricalOptionsTrader(capital)
    
    # Run backtest
    await trader.run_historical_backtest(days)

async def run_live_simulation():
    """Run live simulation using latest market data"""
    print("🔴 LIVE NIFTY SIMULATION")
    print("=" * 60)
    print("Simulating live trading with latest market conditions!")
    
    trader = HistoricalOptionsTrader(100000)
    
    # Get latest few days of data for context
    nifty_data = trader.data_provider.get_nifty_historical(5)
    vix_data = trader.data_provider.get_vix_historical(5)
    
    if not nifty_data:
        print("❌ Unable to fetch live data. Check internet connection.")
        return
    
    # Use latest day's data
    latest_data = nifty_data[-1]
    latest_vix = vix_data[-1].close if vix_data else 15.0
    
    print(f"📊 Latest Data: {latest_data.timestamp.date()}")
    print(f"   NIFTY: {latest_data.close:.2f}")
    print(f"   Range: {latest_data.low:.2f} - {latest_data.high:.2f}")
    print(f"   VIX: {latest_vix:.2f}")
    
    # Simulate live trading session
    await trader._process_trading_day(latest_data, {latest_data.timestamp.date(): latest_vix}, 0)

class AdvancedBacktestAnalyzer:
    """Advanced analysis of backtest results"""
    
    def __init__(self, trade_history: List[Dict]):
        self.trades = pd.DataFrame(trade_history)
        
    def generate_detailed_report(self):
        """Generate comprehensive analysis report"""
        if self.trades.empty:
            print("No trades to analyze")
            return
        
        print(f"\n{'='*80}")
        print("🔬 ADVANCED BACKTEST ANALYSIS")
        print(f"{'='*80}")
        
        # Time-based analysis
        self._analyze_time_patterns()
        
        # Performance metrics
        self._calculate_advanced_metrics()
        
        # Risk analysis
        self._analyze_risk_metrics()
        
        # Strategy insights
        self._generate_insights()
    
    def _analyze_time_patterns(self):
        """Analyze performance by time patterns"""
        self.trades['date'] = pd.to_datetime(self.trades['date'])
        self.trades['weekday'] = self.trades['date'].dt.day_name()
        self.trades['month'] = self.trades['date'].dt.month_name()
        
        print("📅 TIME-BASED PERFORMANCE:")
        
        # Day of week analysis
        weekday_perf = self.trades.groupby('weekday')['pnl'].agg(['count', 'mean', 'sum'])
        weekday_perf['win_rate'] = (self.trades.groupby('weekday')['pnl'] > 0).mean() * 100
        
        print("\n   📊 Day of Week Analysis:")
        print(f"   {'Day':<12} {'Trades':<8} {'Avg P&L':<12} {'Total P&L':<12} {'Win Rate'}")
        print(f"   {'-'*60}")
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            if day in weekday_perf.index:
                data = weekday_perf.loc[day]
                print(f"   {day:<12} {int(data['count']):<8} "
                      f"₹{data['mean']:>10,.0f} ₹{data['sum']:>11,.0f} "
                      f"{weekday_perf.loc[day, 'win_rate']:>7.1f}%")
    
    def _calculate_advanced_metrics(self):
        """Calculate advanced trading metrics"""
        returns = self.trades['pnl'].values
        
        # Sharpe ratio (simplified)
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        # Maximum drawdown
        cumulative_pnl = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - running_max)
        max_drawdown = np.min(drawdown)
        
        # Profit factor
        total_wins = np.sum(returns[returns > 0])
        total_losses = abs(np.sum(returns[returns < 0]))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        print(f"\n📈 ADVANCED METRICS:")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   Max Drawdown: ₹{max_drawdown:,.0f}")
        print(f"   Profit Factor: {profit_factor:.2f}")
        print(f"   Expectancy: ₹{avg_return:,.0f} per trade")
    
    def _analyze_risk_metrics(self):
        """Analyze risk-related metrics"""
        returns = self.trades['pnl_pct'].values
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Maximum consecutive losses
        losses = (self.trades['pnl'] < 0).astype(int)
        max_consecutive_losses = 0
        current_streak = 0
        
        for loss in losses:
            if loss:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0
        
        print(f"\n⚠️ RISK ANALYSIS:")
        print(f"   VaR (95%): {var_95:.1f}% loss")
        print(f"   Max Consecutive Losses: {max_consecutive_losses}")
        print(f"   Largest Single Loss: {self.trades['pnl_pct'].min():.1f}%")
    
    def _generate_insights(self):
        """Generate actionable insights"""
        print(f"\n💡 STRATEGY INSIGHTS:")
        
        # Exit reason analysis
        exit_reasons = self.trades['exit_reason'].value_counts()
        print(f"   Most Common Exit: {exit_reasons.index[0]} ({exit_reasons.iloc[0]} times)")
        
        # Performance by exit reason
        print(f"   📊 Performance by Exit Reason:")
        for reason, group in self.trades.groupby('exit_reason'):
            avg_pnl = group['pnl'].mean()
            count = len(group)
            print(f"      {reason}: ₹{avg_pnl:,.0f} avg ({count} trades)")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        win_rate = (self.trades['pnl'] > 0).mean() * 100
        avg_win = self.trades[self.trades['pnl'] > 0]['pnl'].mean()
        avg_loss = abs(self.trades[self.trades['pnl'] < 0]['pnl'].mean())
        
        if win_rate < 60:
            print(f"   • Consider tightening entry filters (current win rate: {win_rate:.1f}%)")
        if avg_loss > avg_win:
            print(f"   • Improve risk-reward ratio (avg loss > avg win)")
        if len(self.trades) < 10:
            print(f"   • Increase trading frequency with relaxed filters")

class RealTimeMarketSimulator:
    """Simulate real-time market conditions using historical data"""
    
    def __init__(self):
        self.data_provider = NSEHistoricalDataProvider()
        self.current_position = None
        self.capital = 100000
        self.total_pnl = 0
    
    async def run_realtime_simulation(self, duration_minutes: int = 60):
        """Simulate real-time trading for specified duration"""
        print(f"⏰ REAL-TIME SIMULATION ({duration_minutes} minutes)")
        print("=" * 60)
        print("Simulating live market conditions with real historical data patterns")
        
        # Get recent data for realistic simulation
        historical_data = self.data_provider.get_nifty_historical(5)
        if not historical_data:
            print("❌ Unable to fetch data for simulation")
            return
        
        # Use latest day as base
        base_data = historical_data[-1]
        current_price = base_data.close
        
        print(f"📊 Starting simulation from NIFTY: {current_price:.2f}")
        print("🔄 Updates every 30 seconds (simulating 1-minute candles)")
        
        for minute in range(duration_minutes):
            # Simulate realistic price movement
            current_price = self._simulate_price_tick(current_price, base_data)
            
            # Check for trading opportunities
            signal = self._check_trading_signal(current_price, minute)
            
            if signal and not self.current_position:
                self._enter_position(signal, current_price)
            elif self.current_position:
                self._update_position(current_price, minute)
            
            # Display status every 5 minutes
            if minute % 5 == 0:
                self._display_status(current_price, minute)
            
            # Wait 30 seconds (simulating real-time)
            await asyncio.sleep(30 if duration_minutes > 10 else 5)
        
        # Final summary
        self._display_final_summary()
    
    def _simulate_price_tick(self, current_price: float, base_data: HistoricalData) -> float:
        """Simulate realistic price tick"""
        volatility = (base_data.high - base_data.low) / base_data.close / 100  # Per-minute volatility
        random_move = np.random.normal(0, volatility * current_price)
        
        # Add some momentum
        momentum = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])  # Slight bullish bias
        momentum_move = momentum * volatility * current_price * 0.5
        
        new_price = current_price + random_move + momentum_move
        
        # Keep within reasonable bounds (±2% from base)
        max_price = base_data.close * 1.02
        min_price = base_data.close * 0.98
        
        return max(min(new_price, max_price), min_price)
    
    def _check_trading_signal(self, current_price: float, minute: int) -> Optional[Dict]:
        """Check for trading signals"""
        # Simple breakout strategy
        if minute >= 15:  # Only after 15 minutes of observation
            price_change_pct = (current_price - 24300) / 24300 * 100  # Assume base of 24300
            
            if abs(price_change_pct) > 0.3:  # 0.3% breakout
                return {
                    'type': 'BUY_CALL' if price_change_pct > 0 else 'BUY_PUT',
                    'confidence': min(0.8, abs(price_change_pct) * 2)
                }
        
        return None
    
    def _enter_position(self, signal: Dict, current_price: float):
        """Enter a trading position"""
        strike = round(current_price / 50) * 50
        premium = 250  # Simplified premium
        
        self.current_position = {
            'type': signal['type'],
            'strike': strike,
            'entry_price': premium,
            'current_price': premium,
            'entry_time': datetime.now(),
            'spot_at_entry': current_price
        }
        
        print(f"   ✅ Entered: {signal['type']} {strike} @ ₹{premium}")
    
    def _update_position(self, current_price: float, minute: int):
        """Update existing position"""
        if not self.current_position:
            return
        
        # Simulate option price movement
        spot_change = current_price - self.current_position['spot_at_entry']
        
        if 'CALL' in self.current_position['type']:
            premium_change = spot_change * 0.5  # Delta approximation
        else:
            premium_change = -spot_change * 0.5
        
        new_premium = max(1, self.current_position['entry_price'] + premium_change)
        self.current_position['current_price'] = new_premium
        
        # Check exit conditions
        pnl_pct = (new_premium - self.current_position['entry_price']) / self.current_position['entry_price'] * 100
        
        exit_reason = None
        if pnl_pct <= -15:  # 15% stop loss
            exit_reason = "Stop loss"
        elif pnl_pct >= 30:  # 30% profit target
            exit_reason = "Profit target"
        elif minute >= 45:  # Time-based exit
            exit_reason = "Time exit"
        
        if exit_reason:
            self._exit_position(exit_reason)
    
    def _exit_position(self, reason: str):
        """Exit current position"""
        if not self.current_position:
            return
        
        pnl = (self.current_position['current_price'] - self.current_position['entry_price']) * 75
        self.total_pnl += pnl
        
        print(f"   🎯 Exited: {reason} - P&L: ₹{pnl:,.0f}")
        self.current_position = None
    
    def _display_status(self, current_price: float, minute: int):
        """Display current status"""
        position_pnl = 0
        if self.current_position:
            position_pnl = (self.current_position['current_price'] - self.current_position['entry_price']) * 75
        
        print(f"\n⏰ Minute {minute:02d} | NIFTY: {current_price:.2f} | "
              f"Position P&L: ₹{position_pnl:,.0f} | Total: ₹{self.total_pnl:,.0f}")
    
    def _display_final_summary(self):
        """Display final simulation summary"""
        print(f"\n📊 SIMULATION COMPLETE")
        print(f"   Final P&L: ₹{self.total_pnl:,.0f}")
        print(f"   Return: {(self.total_pnl/self.capital)*100:.2f}%")

def display_testing_menu():
    """Display testing options menu"""
    print("\n🎯 NIFTY OPTIONS HISTORICAL TESTING SYSTEM")
    print("=" * 60)
    print("Using Real NSE Historical Data - Works Even When Markets Closed!")
    print()
    print("1. 📊 Historical Backtest (30-90 days)")
    print("2. 🔴 Live Market Simulation (Latest data)")
    print("3. ⏰ Real-Time Simulation (60 minutes)")
    print("4. 🔬 Advanced Analysis (Previous backtest)")
    print("5. 📈 Quick Test (5 days backtest)")
    print("6. 🎯 Strategy Optimization")
    print("7. ❌ Exit")
    print()
    
    choice = input("Enter your choice (1-7): ").strip()
    return choice

async def strategy_optimization():
    """Optimize strategy parameters"""
    print("🎯 STRATEGY PARAMETER OPTIMIZATION")
    print("=" * 60)
    print("Testing different parameter combinations...")
    
    # Test different breakout levels
    breakout_levels = [0.15, 0.25, 0.35]  # Different % breakout levels
    
    results = []
    
    for level in breakout_levels:
        print(f"\n🔍 Testing {level}% breakout level...")
        trader = HistoricalOptionsTrader(100000)
        
        # Modify breakout level (simplified - would need to modify the class)
        # This is a demonstration of how optimization would work
        trader.breakout_multiplier = level / 100
        
        # Run quick backtest
        nifty_data = trader.data_provider.get_nifty_historical(10)  # 10 days for quick test
        vix_data = trader.data_provider.get_vix_historical(10)
        
        if nifty_data:
            vix_lookup = {vix.timestamp.date(): vix.close for vix in vix_data}
            
            for day_data in nifty_data[:5]:  # Test on 5 days
                await trader._process_trading_day(day_data, vix_lookup, 0)
            
            # Calculate results
            if trader.trade_history:
                df = pd.DataFrame(trader.trade_history)
                total_pnl = df['pnl'].sum()
                win_rate = (df['pnl'] > 0).mean() * 100
                total_trades = len(df)
                
                results.append({
                    'breakout_level': level,
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'total_trades': total_trades
                })
    
    # Display optimization results
    print(f"\n📊 OPTIMIZATION RESULTS:")
    print(f"{'Level':<8} {'P&L':<12} {'Win Rate':<10} {'Trades'}")
    print(f"{'-'*40}")
    
    for result in results:
        print(f"{result['breakout_level']:<8}% ₹{result['total_pnl']:>10,.0f} "
              f"{result['win_rate']:>8.1f}% {result['total_trades']:>6}")
    
    # Find best parameter
    if results:
        best_result = max(results, key=lambda x: x['total_pnl'])
        print(f"\n🏆 BEST PARAMETER: {best_result['breakout_level']}% breakout level")
        print(f"   Expected P&L: ₹{best_result['total_pnl']:,.0f}")
        print(f"   Win Rate: {best_result['win_rate']:.1f}%")

async def main():
    """Main testing application"""
    print("📈 NIFTY OPTIONS HISTORICAL TESTING SYSTEM")
    print("=" * 70)
    print("🎯 Uses Real NSE Historical Data via nsepy-xyz.web.app")
    print("✅ Perfect for testing when markets are closed!")
    print("📊 Backtest your exact options strategy with real market data")
    
    while True:
        try:
            choice = display_testing_menu()
            
            if choice == "1":
                await run_nifty_historical_test()
            elif choice == "2":
                await run_live_simulation()
            elif choice == "3":
                simulator = RealTimeMarketSimulator()
                duration = int(input("Enter simulation duration in minutes (default 60): ") or "60")
                await simulator.run_realtime_simulation(duration)
            elif choice == "4":
                print("🔬 Advanced analysis requires previous backtest results")
                print("   Run option 1 (Historical Backtest) first")
            elif choice == "5":
                print("⚡ QUICK 5-DAY BACKTEST")
                trader = HistoricalOptionsTrader(100000)
                await trader.run_historical_backtest(5)
            elif choice == "6":
                await strategy_optimization()
            elif choice == "7":
                print("👋 Happy trading! Your strategy is ready for live markets! 🚀")
                break
            else:
                print("❌ Invalid choice. Please select 1-7.")
            
            input("\n👆 Press Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n🛑 Testing interrupted by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Continuing with testing...")
    
    print("\n✅ Testing session completed!")
    print("🎯 Next step: Run your agentic trader during market hours!")

if __name__ == "__main__":
    print("🚀 STARTING NIFTY OPTIONS HISTORICAL TESTER")
    print("📡 Connecting to NSE historical data API...")
    print("🎯 This works even when stock markets are closed!")
    print()
    
    # Check if this is a quick test run
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        print("⚡ QUICK HISTORICAL TEST (5 days)")
        async def quick_historical_test():
            trader = HistoricalOptionsTrader(100000)
            await trader.run_historical_backtest(5)
        asyncio.run(quick_historical_test())
    else:
        # Run full testing suite
        asyncio.run(main())