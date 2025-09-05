import pandas as pd
import numpy as np
import sqlite3
import json
import asyncio
import random
from datetime import datetime, time, timedelta
import time as sleep_time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
    import joblib
    HAS_SKLEARN = True
except ImportError:
    print("⚠️ ML libraries not found. Install: pip install scikit-learn")
    HAS_SKLEARN = False

# TA-Lib imports
try:
    import talib
    HAS_TALIB = True
except ImportError:
    print("⚠️ TA-Lib not found. Using basic technical analysis.")
    HAS_TALIB = False

# Angel One imports (optional for testing)
try:
    from smartapi import SmartConnect
    import pyotp
    HAS_ANGEL_ONE = True
except ImportError:
    print("⚠️ Angel One libraries not found. Running in simulation mode.")
    HAS_ANGEL_ONE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agentic_trader_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Universal market data structure"""
    symbol: str
    timestamp: datetime
    ltp: float
    open: float
    high: float
    low: float
    volume: int
    prev_close: float
    volatility: float  # VIX for NIFTY, implied volatility for others
    gap_pct: float

@dataclass
class Position:
    """Universal position structure"""
    symbol: str
    instrument_type: str  # 'OPTION', 'FUTURE', 'STOCK'
    entry_price: float
    current_price: float
    quantity: int
    side: str  # 'LONG', 'SHORT'
    entry_time: datetime
    pnl: float = 0.0
    pnl_pct: float = 0.0
    trailing_stop: Optional[float] = None
    profit_locked: bool = False
    partial_booked: bool = False
    stop_loss_price: Optional[float] = None

@dataclass
class TradingSignal:
    """Universal trading signal"""
    signal_type: str  # 'BUY', 'SELL', 'HOLD', 'EXIT'
    instrument: str   # 'CALL', 'PUT', 'FUTURE', 'STOCK'
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    reasons: List[str]
    timestamp: datetime

class DataSimulator:
    """Simulates realistic market data for testing"""
    
    def __init__(self, asset_type: str = "NIFTY"):
        self.asset_type = asset_type
        self.current_price = self.get_base_price()
        self.trend_direction = random.choice([-1, 1])
        self.trend_strength = random.uniform(0.1, 0.8)
        self.volatility = random.uniform(0.15, 0.35)
        self.time_counter = 0
        
    def get_base_price(self) -> float:
        """Get realistic base price for different assets"""
        base_prices = {
            "NIFTY": 24300 + random.uniform(-200, 200),
            "CRUDE": 6800 + random.uniform(-100, 100),
            "BANKNIFTY": 52000 + random.uniform(-500, 500),
            "GOLD": 72000 + random.uniform(-1000, 1000)
        }
        return base_prices.get(self.asset_type, 24300)
    
    def generate_realistic_data(self) -> MarketData:
        """Generate realistic market data with trends and volatility"""
        self.time_counter += 1
        
        # Create realistic price movement
        random_component = np.random.normal(0, self.volatility * 0.01)
        trend_component = self.trend_direction * self.trend_strength * 0.001
        
        # Mean reversion component
        mean_reversion = (self.get_base_price() - self.current_price) * 0.0001
        
        # Combine all components
        price_change = (trend_component + random_component + mean_reversion) * self.current_price
        
        # Update price
        self.current_price += price_change
        
        # Occasionally change trend
        if random.random() < 0.05:  # 5% chance to change trend
            self.trend_direction *= -1
            self.trend_strength = random.uniform(0.1, 0.8)
        
        # Generate OHLC data
        high = self.current_price * (1 + abs(random_component) * 0.5)
        low = self.current_price * (1 - abs(random_component) * 0.5)
        open_price = self.current_price + random.uniform(-price_change, price_change)
        
        # Generate volume (higher during volatile periods)
        base_volume = 1000000 if self.asset_type == "NIFTY" else 100000
        volume_multiplier = 1 + abs(random_component) * 10
        volume = int(base_volume * volume_multiplier)
        
        # Calculate gap percentage
        prev_close = self.current_price - price_change
        gap_pct = abs((open_price - prev_close) / prev_close) * 100
        
        # Generate volatility measure
        volatility = 12 + abs(random_component) * 20  # VIX-like measure
        
        return MarketData(
            symbol=self.asset_type,
            timestamp=datetime.now(),
            ltp=self.current_price,
            open=open_price,
            high=high,
            low=low,
            volume=volume,
            prev_close=prev_close,
            volatility=volatility,
            gap_pct=gap_pct
        )

class CrudeOilDataProvider:
    """Provides live/simulated Crude Oil data"""
    
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        if simulation_mode:
            self.simulator = DataSimulator("CRUDE")
        
    def get_live_data(self) -> MarketData:
        """Get live crude oil data"""
        if self.simulation_mode:
            return self.simulator.generate_realistic_data()
        else:
            # In real mode, you could integrate with commodity data providers
            # like Alpha Vantage, Quandl, or other commodity APIs
            return self._fetch_real_crude_data()
    
    def _fetch_real_crude_data(self) -> MarketData:
        """Fetch real crude oil data from external API"""
        # Placeholder for real data integration
        # You could integrate with:
        # 1. Alpha Vantage API for commodities
        # 2. MCX data providers
        # 3. International crude oil APIs
        
        # For now, return simulated data
        return self.simulator.generate_realistic_data()

class PaperTradingEngine:
    """Paper trading engine for risk-free testing"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: List[Position] = []
        self.trade_history = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        
    def place_order(self, signal: TradingSignal, quantity: int) -> Dict[str, Any]:
        """Simulate order placement"""
        try:
            # Calculate position size based on available capital
            position_value = signal.entry_price * quantity
            
            # For options, multiply by lot size
            if signal.instrument in ['CALL', 'PUT']:
                if signal.signal_type.startswith('NIFTY'):
                    position_value *= 75  # NIFTY lot size
                elif signal.signal_type.startswith('BANKNIFTY'):
                    position_value *= 25  # BANKNIFTY lot size
            
            # Check if we have enough capital
            if position_value > self.current_capital * 0.8:  # Use max 80% of capital
                return {
                    'status': False,
                    'message': f'Insufficient capital. Need ₹{position_value:,.0f}, have ₹{self.current_capital:,.0f}'
                }
            
            # Create position
            position = Position(
                symbol=f"{signal.signal_type}_{self.trade_count}",
                instrument_type=signal.instrument,
                entry_price=signal.entry_price,
                current_price=signal.entry_price,
                quantity=quantity,
                side='LONG' if signal.signal_type.startswith('BUY') else 'SHORT',
                entry_time=datetime.now()
            )
            
            self.positions.append(position)
            self.trade_count += 1
            
            # Reduce available capital
            self.current_capital -= position_value * 0.1  # Assume 10% margin requirement
            
            logger.info(f"📋 Paper Trade Executed: {position.symbol} - {quantity} units at ₹{signal.entry_price}")
            
            return {
                'status': True,
                'message': f'Order executed: {position.symbol}',
                'position': position
            }
            
        except Exception as e:
            logger.error(f"Paper trading error: {e}")
            return {'status': False, 'message': str(e)}
    
    def update_positions(self, current_price: float):
        """Update all position P&L"""
        total_pnl = 0
        positions_to_remove = []
        
        for position in self.positions:
            # Update current price (simplified)
            price_multiplier = random.uniform(0.95, 1.05)  # ±5% random movement
            position.current_price = position.entry_price * price_multiplier
            
            # Calculate P&L
            if position.side == 'LONG':
                position.pnl = (position.current_price - position.entry_price) * position.quantity
            else:
                position.pnl = (position.entry_price - position.current_price) * position.quantity
            
            position.pnl_pct = (position.pnl / (position.entry_price * position.quantity)) * 100
            total_pnl += position.pnl
            
            # Check exit conditions
            if self._should_exit_position(position):
                self._close_position(position)
                positions_to_remove.append(position)
        
        # Remove closed positions
        for position in positions_to_remove:
            self.positions.remove(position)
        
        self.total_pnl = total_pnl
        return total_pnl
    
    def _should_exit_position(self, position: Position) -> bool:
        """Check if position should be exited"""
        # 20% stop loss
        if position.pnl_pct <= -20:
            return True
        
        # 50% profit target
        if position.pnl_pct >= 50:
            return True
        
        # Time-based exit (hold for max 2 hours in simulation)
        if (datetime.now() - position.entry_time).seconds > 7200:
            return True
        
        return False
    
    def _close_position(self, position: Position):
        """Close a position"""
        self.trade_history.append({
            'symbol': position.symbol,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'entry_price': position.entry_price,
            'exit_price': position.current_price,
            'quantity': position.quantity,
            'pnl': position.pnl,
            'pnl_pct': position.pnl_pct
        })
        
        # Return capital
        position_value = position.entry_price * position.quantity * 0.1  # Return margin
        self.current_capital += position_value + position.pnl
        
        logger.info(f"🎯 Position Closed: {position.symbol} - P&L: ₹{position.pnl:,.0f} ({position.pnl_pct:.1f}%)")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get trading performance summary"""
        if not self.trade_history:
            return {"message": "No trades completed yet"}
        
        trades_df = pd.DataFrame(self.trade_history)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        total_trades = len(trades_df)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades) * 100 if total_trades > 0 else 0,
            'total_pnl': trades_df['pnl'].sum(),
            'avg_pnl': trades_df['pnl'].mean(),
            'best_trade': trades_df['pnl'].max(),
            'worst_trade': trades_df['pnl'].min(),
            'current_capital': self.current_capital,
            'total_return': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        }

class UniversalTradingAgent:
    """Universal agent that works with any asset"""
    
    def __init__(self, asset_type: str = "NIFTY", test_mode: bool = True):
        self.asset_type = asset_type
        self.test_mode = test_mode
        
        # Initialize data provider based on asset
        if asset_type == "CRUDE":
            self.data_provider = CrudeOilDataProvider(simulation_mode=test_mode)
        else:
            self.data_provider = DataSimulator(asset_type)
        
        # Initialize paper trading engine
        self.paper_engine = PaperTradingEngine(100000)
        
        # Trading parameters
        self.observation_period = []
        self.daily_trades = 0
        self.max_daily_trades = 5 if asset_type == "CRUDE" else 3
        
        # ML components
        if HAS_SKLEARN:
            self.ml_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.scaler = StandardScaler()
            self.is_trained = False
            self.training_data = []
    
    def collect_market_data(self) -> MarketData:
        """Collect market data for any asset"""
        if self.asset_type == "CRUDE":
            return self.data_provider.get_live_data()
        else:
            return self.data_provider.generate_realistic_data()
    
    def analyze_market(self, market_data: MarketData) -> Dict[str, Any]:
        """Universal market analysis"""
        analysis = {
            'trend': 'NEUTRAL',
            'momentum': 0.0,
            'volatility_score': market_data.volatility / 20.0,  # Normalized
            'volume_score': 1.0,  # Placeholder
            'technical_score': 0.0,
            'pattern': 'NONE'
        }
        
        # Store data for ML training
        if HAS_SKLEARN:
            self.training_data.append(market_data)
            if len(self.training_data) > 50 and not self.is_trained:
                self._train_ml_model()
        
        # Basic trend analysis
        if len(self.observation_period) > 5:
            recent_prices = [data.ltp for data in self.observation_period[-5:]]
            trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            
            if trend_slope > 0:
                analysis['trend'] = 'BULLISH'
                analysis['momentum'] = min(trend_slope / market_data.ltp, 0.05)
            else:
                analysis['trend'] = 'BEARISH'
                analysis['momentum'] = max(trend_slope / market_data.ltp, -0.05)
        
        # Volatility analysis
        if market_data.volatility > 18:
            analysis['pattern'] = 'HIGH_VOLATILITY'
        elif market_data.gap_pct > 0.5:
            analysis['pattern'] = 'GAP_MOVE'
        
        # Technical score (simplified)
        analysis['technical_score'] = (
            (1 if analysis['trend'] == 'BULLISH' else -1 if analysis['trend'] == 'BEARISH' else 0) * 0.3 +
            analysis['momentum'] * 10 +
            (1 if analysis['pattern'] in ['HIGH_VOLATILITY', 'GAP_MOVE'] else 0) * 0.2
        )
        
        return analysis
    
    def _train_ml_model(self):
        """Train ML model with collected data"""
        try:
            if len(self.training_data) < 20:
                return
            
            # Prepare features and targets
            features = []
            targets = []
            
            for i in range(10, len(self.training_data) - 1):
                # Features: current market state
                current_data = self.training_data[i]
                feature_vector = [
                    current_data.ltp,
                    current_data.volume,
                    current_data.volatility,
                    current_data.gap_pct,
                    current_data.timestamp.hour,
                    current_data.timestamp.weekday()
                ]
                
                # Target: next period's price change
                next_price = self.training_data[i + 1].ltp
                price_change = (next_price - current_data.ltp) / current_data.ltp
                
                features.append(feature_vector)
                targets.append(price_change)
            
            if len(features) > 10:
                # Train model
                X = np.array(features)
                y = np.array(targets)
                
                X_scaled = self.scaler.fit_transform(X)
                self.ml_model.fit(X_scaled, y)
                self.is_trained = True
                
                logger.info(f"🧠 ML Model trained with {len(features)} samples for {self.asset_type}")
        
        except Exception as e:
            logger.error(f"ML training error: {e}")
    
    def generate_signal(self, market_data: MarketData, analysis: Dict[str, Any]) -> TradingSignal:
        """Generate trading signal for any asset"""
        signal_type = "HOLD"
        instrument = "FUTURE"  # Default for commodities
        confidence = 0.0
        reasons = []
        
        # Asset-specific signal generation
        if self.asset_type == "CRUDE":
            signal_type, instrument, confidence, reasons = self._generate_crude_signal(market_data, analysis)
        elif self.asset_type in ["NIFTY", "BANKNIFTY"]:
            signal_type, instrument, confidence, reasons = self._generate_options_signal(market_data, analysis)
        
        return TradingSignal(
            signal_type=signal_type,
            instrument=instrument,
            confidence=confidence,
            entry_price=market_data.ltp,
            target_price=market_data.ltp * (1.02 if 'BUY' in signal_type else 0.98),
            stop_loss=market_data.ltp * (0.98 if 'BUY' in signal_type else 1.02),
            reasons=reasons,
            timestamp=datetime.now()
        )
    
    def _generate_crude_signal(self, market_data: MarketData, analysis: Dict[str, Any]) -> Tuple[str, str, float, List[str]]:
        """Generate crude oil specific signals"""
        signal_type = "HOLD"
        instrument = "FUTURE"
        confidence = 0.0
        reasons = []
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return signal_type, instrument, confidence, ["Daily trade limit reached"]
        
        # Crude oil specific conditions
        technical_score = analysis.get('technical_score', 0)
        volatility_score = analysis.get('volatility_score', 0)
        pattern = analysis.get('pattern', 'NONE')
        
        # Strong bullish signal
        if technical_score > 0.5 and volatility_score > 0.6:
            signal_type = "BUY_FUTURE"
            confidence = min(0.8, technical_score + volatility_score)
            reasons = [
                f"Technical score: {technical_score:.2f}",
                f"High volatility: {market_data.volatility:.1f}",
                f"Pattern: {pattern}"
            ]
        
        # Strong bearish signal
        elif technical_score < -0.5 and volatility_score > 0.6:
            signal_type = "SELL_FUTURE"
            confidence = min(0.8, abs(technical_score) + volatility_score)
            reasons = [
                f"Technical score: {technical_score:.2f}",
                f"High volatility: {market_data.volatility:.1f}",
                f"Pattern: {pattern}"
            ]
        
        # Gap trading
        elif market_data.gap_pct > 1.0:  # 1% gap for crude
            if analysis['trend'] == 'BULLISH':
                signal_type = "BUY_FUTURE"
                confidence = 0.6
                reasons = [f"Gap up: {market_data.gap_pct:.2f}%", "Bullish trend"]
            elif analysis['trend'] == 'BEARISH':
                signal_type = "SELL_FUTURE"
                confidence = 0.6
                reasons = [f"Gap down: {market_data.gap_pct:.2f}%", "Bearish trend"]
        
        return signal_type, instrument, confidence, reasons
    
    def _generate_options_signal(self, market_data: MarketData, analysis: Dict[str, Any]) -> Tuple[str, str, float, List[str]]:
        """Generate options specific signals (for NIFTY/BANKNIFTY)"""
        signal_type = "HOLD"
        instrument = "CALL"
        confidence = 0.0
        reasons = []
        
        # Check if we have observation data for breakout strategy
        if len(self.observation_period) < 10:
            return signal_type, instrument, confidence, ["Insufficient observation data"]
        
        # Calculate breakout levels
        observation_high = max([data.high for data in self.observation_period])
        observation_low = min([data.low for data in self.observation_period])
        
        bullish_breakout = observation_high * 1.0025  # +0.25%
        bearish_breakout = observation_low * 0.9975   # -0.25%
        
        current_price = market_data.ltp
        
        # Bullish breakout
        if current_price > bullish_breakout:
            signal_type = "BUY_CALL"
            instrument = "CALL"
            confidence = 0.7
            reasons = [
                f"Bullish breakout: {current_price} > {bullish_breakout}",
                f"Volatility: {market_data.volatility:.1f}",
                f"Gap: {market_data.gap_pct:.2f}%"
            ]
        
        # Bearish breakout
        elif current_price < bearish_breakout:
            signal_type = "BUY_PUT"
            instrument = "PUT"
            confidence = 0.7
            reasons = [
                f"Bearish breakout: {current_price} < {bearish_breakout}",
                f"Volatility: {market_data.volatility:.1f}",
                f"Gap: {market_data.gap_pct:.2f}%"
            ]
        
        return signal_type, instrument, confidence, reasons
    
    async def run_test_session(self, duration_minutes: int = 60):
        """Run a complete test session"""
        logger.info(f"🚀 Starting {duration_minutes}-minute test session for {self.asset_type}")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        cycle_count = 0
        
        try:
            while datetime.now() < end_time:
                cycle_count += 1
                
                # Collect market data
                market_data = self.collect_market_data()
                self.observation_period.append(market_data)
                
                # Keep only last 50 observations
                if len(self.observation_period) > 50:
                    self.observation_period.pop(0)
                
                # Analyze market
                analysis = self.analyze_market(market_data)
                
                # Generate signal
                signal = self.generate_signal(market_data, analysis)
                
                # Execute signal in paper trading
                if signal.signal_type != "HOLD" and signal.confidence > 0.5:
                    position_size = self._calculate_position_size(signal, market_data)
                    result = self.paper_engine.place_order(signal, position_size)
                    
                    if result['status']:
                        self.daily_trades += 1
                
                # Update existing positions
                self.paper_engine.update_positions(market_data.ltp)
                
                # Display status every 10 cycles
                if cycle_count % 10 == 0:
                    self._display_test_status(market_data, signal, analysis, cycle_count)
                
                # Wait before next cycle (3 minutes in real trading, 10 seconds in testing)
                await asyncio.sleep(10)
        
        except KeyboardInterrupt:
            logger.info("Test session stopped by user")
        
        finally:
            # Final report
            self._display_final_report(duration_minutes, cycle_count)
    
    def _calculate_position_size(self, signal: TradingSignal, market_data: MarketData) -> int:
        """Calculate appropriate position size"""
        # For crude: 1-3 lots based on confidence
        if self.asset_type == "CRUDE":
            return max(1, min(3, int(signal.confidence * 4)))
        
        # For options: 1-2 lots based on confidence
        else:
            return max(1, min(2, int(signal.confidence * 3)))
    
    def _display_test_status(self, market_data: MarketData, signal: TradingSignal, analysis: Dict[str, Any], cycle: int):
        """Display current test status"""
        performance = self.paper_engine.get_performance_summary()
        
        print(f"\n{'='*80}")
        print(f"🤖 TEST CYCLE {cycle} - {self.asset_type} TRADER - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*80}")
        print(f"Price: {market_data.ltp:.2f} | Volatility: {market_data.volatility:.1f} | Gap: {market_data.gap_pct:.2f}%")
        print(f"Signal: {signal.signal_type} | Confidence: {signal.confidence:.2f}")
        print(f"Reasons: {', '.join(signal.reasons)}")
        print(f"Active Positions: {len(self.paper_engine.positions)}")
        
        if isinstance(performance, dict) and 'total_pnl' in performance:
            print(f"Paper P&L: ₹{performance['total_pnl']:,.0f} | Total Trades: {performance['total_trades']}")
            print(f"Win Rate: {performance['win_rate']:.1f}% | Capital: ₹{performance['current_capital']:,.0f}")
        
        print(f"{'='*80}")
    
    def _display_final_report(self, duration: int, cycles: int):
        """Display final test report"""
        performance = self.paper_engine.get_performance_summary()
        
        print(f"\n{'='*80}")
        print(f"📊 FINAL TEST REPORT - {self.asset_type}")
        print(f"{'='*80}")
        print(f"Test Duration: {duration} minutes ({cycles} cycles)")
        print(f"Asset Tested: {self.asset_type}")
        print(f"Mode: {'Paper Trading' if self.test_mode else 'Live Trading'}")
        
        if isinstance(performance, dict) and 'total_trades' in performance:
            print(f"\n📈 PERFORMANCE METRICS:")
            print(f"Total Trades: {performance['total_trades']}")
            print(f"Winning Trades: {performance['winning_trades']}")
            print(f"Win Rate: {performance['win_rate']:.1f}%")
            print(f"Total P&L: ₹{performance['total_pnl']:,.0f}")
            print(f"Average P&L per Trade: ₹{performance['avg_pnl']:,.0f}")
            print(f"Best Trade: ₹{performance['best_trade']:,.0f}")
            print(f"Worst Trade: ₹{performance['worst_trade']:,.0f}")
            print(f"Final Capital: ₹{performance['current_capital']:,.0f}")
            print(f"Total Return: {performance['total_return']:.2f}%")
        else:
            print("📊 No trades executed during test period")
        
        print(f"\n🎯 NEXT STEPS:")
        if self.asset_type == "CRUDE":
            print("1. Crude oil trading available 24/5 (Mon-Fri)")
            print("2. Adjust parameters based on test results")
            print("3. Consider live testing with small position sizes")
        else:
            print("1. Test during market hours (9:15 AM - 3:30 PM)")
            print("2. Compare with NIFTY options strategy")
            print("3. Optimize entry/exit rules based on results")
        
        print(f"{'='*80}")

# Main execution functions
async def test_crude_oil():
    """Test system with Crude Oil"""
    print("🛢️ CRUDE OIL TESTING MODE")
    print("=" * 50)
    print("Testing agentic system with simulated crude oil data...")
    print("This works even when equity markets are closed!")
    
    trader = UniversalTradingAgent("CRUDE", test_mode=True)
    await trader.run_test_session(duration_minutes=30)  # 30-minute test

async def test_nifty_options():
    """Test system with NIFTY Options (simulated)"""
    print("📈 NIFTY OPTIONS TESTING MODE")
    print("=" * 50)
    print("Testing agentic system with simulated NIFTY options data...")
    print("Perfect for testing your original strategy!")
    
    trader = UniversalTradingAgent("NIFTY", test_mode=True)
    await trader.run_test_session(duration_minutes=45)  # 45-minute test

async def test_banknifty():
    """Test system with Bank NIFTY"""
    print("🏦 BANK NIFTY TESTING MODE")
    print("=" * 50)
    print("Testing agentic system with Bank NIFTY futures/options...")
    
    trader = UniversalTradingAgent("BANKNIFTY", test_mode=True)
    await trader.run_test_session(duration_minutes=30)

async def test_gold():
    """Test system with Gold futures"""
    print("🥇 GOLD TESTING MODE")
    print("=" * 50)
    print("Testing agentic system with simulated gold futures...")
    
    trader = UniversalTradingAgent("GOLD", test_mode=True)
    await trader.run_test_session(duration_minutes=30)

class MultiAssetTester:
    """Run comprehensive testing across multiple assets"""
    
    def __init__(self):
        self.test_results = {}
    
    async def run_comprehensive_test(self):
        """Test system across multiple assets"""
        print("🌐 COMPREHENSIVE MULTI-ASSET TESTING")
        print("=" * 60)
        print("Testing agentic system across different asset classes...")
        
        assets = ["CRUDE", "NIFTY", "BANKNIFTY", "GOLD"]
        
        for asset in assets:
            print(f"\n🔍 Testing {asset}...")
            trader = UniversalTradingAgent(asset, test_mode=True)
            
            # Run shorter test for each asset
            start_time = datetime.now()
            await trader.run_test_session(duration_minutes=15)
            end_time = datetime.now()
            
            # Store results
            performance = trader.paper_engine.get_performance_summary()
            self.test_results[asset] = {
                'performance': performance,
                'test_duration': (end_time - start_time).total_seconds() / 60,
                'total_cycles': len(trader.observation_period)
            }
        
        # Display comparative results
        self._display_comparative_results()
    
    def _display_comparative_results(self):
        """Display results across all tested assets"""
        print(f"\n{'='*80}")
        print("📊 COMPARATIVE ASSET TESTING RESULTS")
        print(f"{'='*80}")
        
        print(f"{'Asset':<12} {'Trades':<8} {'Win Rate':<10} {'P&L':<15} {'Return':<10}")
        print(f"{'-'*80}")
        
        for asset, results in self.test_results.items():
            perf = results['performance']
            if isinstance(perf, dict) and 'total_trades' in perf:
                print(f"{asset:<12} {perf['total_trades']:<8} {perf['win_rate']:<10.1f}% "
                      f"₹{perf['total_pnl']:<14,.0f} {perf['total_return']:<10.2f}%")
            else:
                print(f"{asset:<12} {'0':<8} {'N/A':<10} {'₹0':<15} {'0.00%':<10}")
        
        print(f"\n🎯 INSIGHTS:")
        best_performer = max(self.test_results.keys(), 
                           key=lambda x: self.test_results[x]['performance'].get('total_return', 0) 
                           if isinstance(self.test_results[x]['performance'], dict) else 0)
        
        print(f"• Best Performer: {best_performer}")
        print(f"• Crude Oil: Available 24/5 for continuous testing")
        print(f"• NIFTY/BANKNIFTY: Best during market hours (9:15 AM - 3:30 PM)")
        print(f"• Gold: Good for diversification and volatility trading")

class BacktestingEngine:
    """Advanced backtesting with historical data"""
    
    def __init__(self, asset_type: str = "NIFTY"):
        self.asset_type = asset_type
        self.historical_data = []
        self.generate_historical_data()
    
    def generate_historical_data(self, days: int = 30):
        """Generate realistic historical data for backtesting"""
        simulator = DataSimulator(self.asset_type)
        
        # Generate data for past 30 days (assuming 6 hours trading per day, 72 data points per day)
        for day in range(days):
            for hour in range(72):  # 6 hours * 12 (5-minute intervals)
                data = simulator.generate_realistic_data()
                
                # Adjust timestamp to past
                data.timestamp = datetime.now() - timedelta(days=days-day, minutes=hour*5)
                self.historical_data.append(data)
    
    async def run_backtest(self):
        """Run complete backtest on historical data"""
        print(f"📊 BACKTESTING {self.asset_type} - {len(self.historical_data)} data points")
        print("=" * 60)
        
        trader = UniversalTradingAgent(self.asset_type, test_mode=True)
        
        for i, market_data in enumerate(self.historical_data):
            # Add to observation period
            trader.observation_period.append(market_data)
            
            if len(trader.observation_period) > 50:
                trader.observation_period.pop(0)
            
            # Only trade after sufficient observation
            if len(trader.observation_period) >= 10:
                analysis = trader.analyze_market(market_data)
                signal = trader.generate_signal(market_data, analysis)
                
                # Execute profitable signals
                if signal.signal_type != "HOLD" and signal.confidence > 0.6:
                    position_size = trader._calculate_position_size(signal, market_data)
                    trader.paper_engine.place_order(signal, position_size)
                
                # Update positions
                trader.paper_engine.update_positions(market_data.ltp)
            
            # Show progress every 100 data points
            if i % 100 == 0:
                progress = (i / len(self.historical_data)) * 100
                print(f"Backtest Progress: {progress:.1f}% - {i}/{len(self.historical_data)} data points")
        
        # Final results
        performance = trader.paper_engine.get_performance_summary()
        self._display_backtest_results(performance)
    
    def _display_backtest_results(self, performance):
        """Display backtest results"""
        print(f"\n{'='*60}")
        print(f"📈 BACKTEST RESULTS - {self.asset_type}")
        print(f"{'='*60}")
        
        if isinstance(performance, dict) and 'total_trades' in performance:
            print(f"Historical Period: 30 days")
            print(f"Total Signals: {performance['total_trades']}")
            print(f"Win Rate: {performance['win_rate']:.1f}%")
            print(f"Total Return: {performance['total_return']:.2f}%")
            print(f"Best Trade: ₹{performance['best_trade']:,.0f}")
            print(f"Worst Trade: ₹{performance['worst_trade']:,.0f}")
            print(f"Average P&L: ₹{performance['avg_pnl']:,.0f}")
            
            # Performance metrics
            if performance['total_trades'] > 0:
                sharpe_ratio = performance['avg_pnl'] / abs(performance['worst_trade']) if performance['worst_trade'] != 0 else 0
                print(f"Risk-Reward Ratio: {sharpe_ratio:.2f}")
        
        print(f"\n✅ Backtest completed successfully!")

def display_testing_menu():
    """Display testing options menu"""
    print("\n🎯 AGENTIC AI TESTING SYSTEM")
    print("=" * 50)
    print("Choose your testing mode:")
    print()
    print("1. 🛢️  Crude Oil Testing (Works 24/5 - Even when markets closed!)")
    print("2. 📈 NIFTY Options Testing (Your original strategy)")
    print("3. 🏦 Bank NIFTY Testing (Higher volatility)")
    print("4. 🥇 Gold Testing (Safe haven asset)")
    print("5. 🌐 Multi-Asset Testing (Test all assets)")
    print("6. 📊 Backtesting (Historical data analysis)")
    print("7. 🎮 Interactive Demo (Step-by-step walkthrough)")
    print("8. ❌ Exit")
    print()
    
    choice = input("Enter your choice (1-8): ").strip()
    return choice

async def interactive_demo():
    """Interactive demo showing how the system works step-by-step"""
    print("\n🎮 INTERACTIVE DEMO - See How Each Agent Works")
    print("=" * 60)
    
    # Initialize crude oil trader for demo
    trader = UniversalTradingAgent("CRUDE", test_mode=True)
    
    print("This demo shows you exactly how each AI agent processes data...")
    input("\n👆 Press Enter to start...")
    
    for step in range(5):
        print(f"\n🔄 STEP {step + 1}: Agent Pipeline Execution")
        print("-" * 40)
        
        # Step 1: Data Collection
        print("1. 📊 Data Collection Agent working...")
        market_data = trader.collect_market_data()
        print(f"   ✅ Collected: Price=₹{market_data.ltp:.0f}, Vol={market_data.volatility:.1f}, Gap={market_data.gap_pct:.2f}%")
        
        # Step 2: Market Analysis
        print("2. 🧠 Analysis Agent thinking...")
        analysis = trader.analyze_market(market_data)
        print(f"   ✅ Analysis: Trend={analysis['trend']}, Score={analysis.get('technical_score', 0):.2f}")
        
        # Step 3: Signal Generation
        print("3. 🎯 Strategy Agent deciding...")
        signal = trader.generate_signal(market_data, analysis)
        print(f"   ✅ Signal: {signal.signal_type} (Confidence: {signal.confidence:.2f})")
        print(f"   📝 Reasoning: {', '.join(signal.reasons[:2])}")
        
        # Step 4: Risk Assessment
        print("4. 🛡️ Risk Manager checking...")
        print(f"   ✅ Position size: {trader._calculate_position_size(signal, market_data)} lots")
        print(f"   ✅ Daily trades: {trader.daily_trades}/{trader.max_daily_trades}")
        
        # Step 5: Execution
        print("5. ⚡ Execution Agent acting...")
        if signal.signal_type != "HOLD" and signal.confidence > 0.5:
            result = trader.paper_engine.place_order(signal, 1)
            if result['status']:
                trader.daily_trades += 1
                print(f"   ✅ Trade executed: {signal.signal_type}")
            else:
                print(f"   ❌ Trade skipped: {result['message']}")
        else:
            print(f"   ⏸️ No trade: Signal confidence too low or HOLD signal")
        
        # Update positions
        trader.paper_engine.update_positions(market_data.ltp)
        
        print(f"\n📊 Current Status:")
        print(f"   💰 Paper P&L: ₹{trader.paper_engine.total_pnl:,.0f}")
        print(f"   🎯 Active Positions: {len(trader.paper_engine.positions)}")
        
        if step < 4:  # Don't wait after last step
            input("👆 Press Enter for next step...")
    
    print(f"\n🎉 DEMO COMPLETE!")
    print("You've seen how all 5 AI agents work together every 3 minutes!")
    print("In real trading, this happens automatically while you relax! 😎")

async def main():
    """Main testing application"""
    print("🤖 AGENTIC AI OPTIONS TRADER - TESTING SUITE")
    print("=" * 60)
    print("Perfect for testing when markets are closed!")
    print("Works with multiple asset classes including Crude Oil 24/5")
    
    while True:
        try:
            choice = display_testing_menu()
            
            if choice == "1":
                await test_crude_oil()
            elif choice == "2":
                await test_nifty_options()
            elif choice == "3":
                await test_banknifty()
            elif choice == "4":
                await test_gold()
            elif choice == "5":
                tester = MultiAssetTester()
                await tester.run_comprehensive_test()
            elif choice == "6":
                asset = input("Enter asset for backtesting (CRUDE/NIFTY/BANKNIFTY/GOLD): ").upper()
                if asset in ["CRUDE", "NIFTY", "BANKNIFTY", "GOLD"]:
                    backtester = BacktestingEngine(asset)
                    await backtester.run_backtest()
                else:
                    print("❌ Invalid asset. Using NIFTY...")
                    backtester = BacktestingEngine("NIFTY")
                    await backtester.run_backtest()
            elif choice == "7":
                await interactive_demo()
            elif choice == "8":
                print("👋 Thanks for testing! Ready to go live? 🚀")
                break
            else:
                print("❌ Invalid choice. Please select 1-8.")
            
            input("\n👆 Press Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n🛑 Testing interrupted by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Continuing with testing...")
    
    print("\n✅ Testing session completed!")

if __name__ == "__main__":
    print("🚀 STARTING AGENTIC AI TESTING SYSTEM")
    print("📋 Note: This works even when stock markets are closed!")
    print("🛢️ Crude oil and commodities trade almost 24/5")
    print()
    
    # Check if this is a quick test run
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        print("⚡ QUICK CRUDE OIL TEST (5 minutes)")
        async def quick_test():
            trader = UniversalTradingAgent("CRUDE", test_mode=True)
            await trader.run_test_session(duration_minutes=5)
        asyncio.run(quick_test())
    else:
        # Run full testing suite
        asyncio.run(main())