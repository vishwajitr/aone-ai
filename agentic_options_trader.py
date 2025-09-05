import pandas as pd
import numpy as np
import sqlite3
import json
import asyncio
import threading
from datetime import datetime, time, timedelta
import time as sleep_time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

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
    
# ML and Analysis imports
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import joblib
except ImportError:
    print("⚠️ ML libraries not found. Install: pip install scikit-learn")

try:
    import talib
except ImportError:
    print("⚠️ TA-Lib not found. Install: pip install TA-Lib")
    talib = None

try:
    from smartapi import SmartConnect
    import pyotp
except ImportError:
    print("⚠️ SmartAPI not found. Install: pip install smartapi-python pyotp")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agentic_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Structured market data"""
    symbol: str
    timestamp: datetime
    ltp: float
    open: float
    high: float
    low: float
    volume: int
    prev_close: float
    vix: float
    gap_pct: float

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    token: str
    entry_price: float
    current_price: float
    quantity: int
    option_type: str
    entry_time: datetime
    order_id: str
    pnl: float = 0.0
    pnl_pct: float = 0.0
    trailing_stop: Optional[float] = None
    profit_locked: bool = False
    partial_booked: bool = False
    stop_loss_price: Optional[float] = None

@dataclass
class TradingSignal:
    """Trading signal structure"""
    signal_type: str  # 'BUY_CALL', 'BUY_PUT', 'HOLD', 'EXIT'
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    reasons: List[str]
    timestamp: datetime

class BaseAgent(ABC):
    """Base class for all trading agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Agent.{name}")
        self.is_active = True
    
    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data and return results"""
        pass
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(f"[{self.name}] {message}")
    
    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(f"[{self.name}] {message}")

class DataCollectionAgent(BaseAgent):
    """Collects and processes market data"""
    
    def __init__(self, smart_api: SmartConnect):
        super().__init__("DataCollector")
        self.smart_api = smart_api
        self.nifty_token = "99926000"
        self.vix_token = "99926009"
        self.data_history = []
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect comprehensive market data"""
        try:
            # Get NIFTY data
            nifty_data = self.get_market_data("NIFTY 50", self.nifty_token)
            if not nifty_data:
                return {"error": "Failed to fetch NIFTY data"}
            
            # Get VIX data
            vix = self.get_vix_data()
            
            # Calculate additional metrics
            gap_pct = self.calculate_gap_percentage(
                nifty_data.get('open', 0),
                nifty_data.get('prev_close', 0)
            )
            
            market_data = MarketData(
                symbol="NIFTY",
                timestamp=datetime.now(),
                ltp=nifty_data.get('ltp', 0),
                open=nifty_data.get('open', 0),
                high=nifty_data.get('high', 0),
                low=nifty_data.get('low', 0),
                volume=nifty_data.get('volume', 0),
                prev_close=nifty_data.get('prev_close', 0),
                vix=vix,
                gap_pct=gap_pct
            )
            
            # Store in history
            self.data_history.append(market_data)
            if len(self.data_history) > 1000:  # Keep last 1000 data points
                self.data_history.pop(0)
            
            return {
                "market_data": market_data,
                "status": "success",
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.log_error(f"Data collection failed: {e}")
            return {"error": str(e)}
    
    def get_market_data(self, symbol: str, token: str) -> Dict:
        """Fetch market data from Angel One"""
        try:
            ltp_data = self.smart_api.ltpData("NSE", symbol, token)
            if ltp_data['status']:
                return ltp_data['data']
            return {}
        except Exception as e:
            self.log_error(f"Error fetching market data: {e}")
            return {}
    
    def get_vix_data(self) -> float:
        """Get VIX data"""
        try:
            vix_data = self.smart_api.ltpData("NSE", "INDIA VIX", self.vix_token)
            if vix_data['status']:
                return vix_data['data'].get('ltp', 0)
            return 0
        except Exception as e:
            self.log_error(f"Error fetching VIX: {e}")
            return 0
    
    def calculate_gap_percentage(self, current: float, prev_close: float) -> float:
        """Calculate gap percentage"""
        if prev_close == 0:
            return 0
        return abs((current - prev_close) / prev_close) * 100
    
    def get_historical_data(self, days: int = 10) -> List[Dict]:
        """Get historical data for analysis"""
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M')
            to_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            
            hist_data = self.smart_api.getCandleData({
                "exchange": "NSE",
                "symboltoken": self.nifty_token,
                "interval": "ONE_DAY",
                "fromdate": from_date,
                "todate": to_date
            })
            
            if hist_data['status'] and hist_data['data']:
                return hist_data['data']
            return []
        except Exception as e:
            self.log_error(f"Error getting historical data: {e}")
            return []

class TechnicalAnalysisAgent(BaseAgent):
    """Performs technical analysis and generates insights"""
    
    def __init__(self):
        super().__init__("TechnicalAnalyst")
        self.indicators = {}
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform technical analysis"""
        try:
            market_data = data.get('market_data')
            if not market_data:
                return {"error": "No market data provided"}
            
            # Get historical data for calculations
            data_collector = data.get('data_collector')
            if data_collector:
                historical_data = data_collector.get_historical_data(20)
                if historical_data:
                    analysis = self.calculate_technical_indicators(historical_data, market_data)
                    return {
                        "technical_analysis": analysis,
                        "status": "success",
                        "timestamp": datetime.now()
                    }
            
            # Fallback basic analysis
            return {
                "technical_analysis": {
                    "trend": "NEUTRAL",
                    "momentum": 0.5,
                    "volatility": market_data.vix / 30.0,  # Normalized
                    "support": market_data.ltp * 0.98,
                    "resistance": market_data.ltp * 1.02
                },
                "status": "success"
            }
            
        except Exception as e:
            self.log_error(f"Technical analysis failed: {e}")
            return {"error": str(e)}
    
    def calculate_technical_indicators(self, historical_data: List, current_data: MarketData) -> Dict:
        """Calculate comprehensive technical indicators"""
        try:
            # Extract price data
            closes = np.array([float(candle[4]) for candle in historical_data])  # Close prices
            highs = np.array([float(candle[2]) for candle in historical_data])   # High prices
            lows = np.array([float(candle[3]) for candle in historical_data])    # Low prices
            volumes = np.array([float(candle[5]) for candle in historical_data]) # Volume
            
            if len(closes) < 14:  # Need minimum data for indicators
                return {"trend": "INSUFFICIENT_DATA", "momentum": 0.5}
            
            analysis = {}
            
            # Moving Averages
            if len(closes) >= 9:
                sma_9 = np.mean(closes[-9:])
                analysis['sma_9'] = sma_9
                analysis['price_vs_sma9'] = current_data.ltp / sma_9
            
            # RSI calculation
            if talib and len(closes) >= 14:
                rsi = talib.RSI(closes, timeperiod=14)[-1]
                analysis['rsi'] = rsi
                analysis['rsi_signal'] = 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL'
            
            # Bollinger Bands
            if talib and len(closes) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)
                analysis['bb_upper'] = bb_upper[-1]
                analysis['bb_lower'] = bb_lower[-1]
                analysis['bb_position'] = (current_data.ltp - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            # MACD
            if talib and len(closes) >= 26:
                macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
                analysis['macd'] = macd[-1]
                analysis['macd_signal'] = macd_signal[-1]
                analysis['macd_trend'] = 'BULLISH' if macd[-1] > macd_signal[-1] else 'BEARISH'
            
            # Volume analysis
            avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes.mean()
            analysis['volume_ratio'] = current_data.volume / avg_volume if avg_volume > 0 else 1
            
            # Trend determination
            if len(closes) >= 5:
                recent_trend = np.polyfit(range(5), closes[-5:], 1)[0]
                analysis['trend'] = 'BULLISH' if recent_trend > 0 else 'BEARISH'
                analysis['trend_strength'] = abs(recent_trend) / closes[-1]
            
            # Momentum calculation
            if len(closes) >= 5:
                momentum = (closes[-1] - closes[-5]) / closes[-5]
                analysis['momentum'] = momentum
            
            # Support and Resistance
            recent_lows = lows[-10:] if len(lows) >= 10 else lows
            recent_highs = highs[-10:] if len(highs) >= 10 else highs
            analysis['support'] = np.min(recent_lows)
            analysis['resistance'] = np.max(recent_highs)
            
            return analysis
            
        except Exception as e:
            self.log_error(f"Error calculating technical indicators: {e}")
            return {"trend": "ERROR", "momentum": 0.5}

class MLPredictionAgent(BaseAgent):
    """Machine learning predictions and pattern recognition"""
    
    def __init__(self):
        super().__init__("MLPredictor")
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ML predictions"""
        try:
            market_data = data.get('market_data')
            technical_analysis = data.get('technical_analysis', {})
            
            if not market_data:
                return {"error": "No market data provided"}
            
            # Prepare features
            features = self.prepare_features(market_data, technical_analysis)
            
            # Train model if not trained
            if not self.is_trained:
                data_collector = data.get('data_collector')
                if data_collector and len(data_collector.data_history) > 50:
                    await self.train_model(data_collector.data_history)
            
            # Generate predictions
            predictions = {}
            
            if self.is_trained and self.model:
                # Price prediction
                feature_array = np.array([features]).reshape(1, -1)
                feature_scaled = self.scaler.transform(feature_array)
                price_pred = self.model.predict(feature_scaled)[0]
                predictions['predicted_price'] = price_pred
                predictions['price_change_pct'] = (price_pred - market_data.ltp) / market_data.ltp * 100
            
            # Anomaly detection
            if len(features) > 0:
                anomaly_score = self.anomaly_detector.fit_predict([features])[0]
                predictions['is_anomaly'] = anomaly_score == -1
                predictions['market_regime'] = 'UNUSUAL' if anomaly_score == -1 else 'NORMAL'
            
            # Pattern recognition
            patterns = self.detect_patterns(market_data, technical_analysis)
            predictions['patterns'] = patterns
            
            return {
                "ml_predictions": predictions,
                "status": "success",
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.log_error(f"ML prediction failed: {e}")
            return {"error": str(e)}
    
    def prepare_features(self, market_data: MarketData, technical_analysis: Dict) -> List[float]:
        """Prepare features for ML model"""
        features = [
            market_data.ltp,
            market_data.volume,
            market_data.vix,
            market_data.gap_pct,
            technical_analysis.get('rsi', 50),
            technical_analysis.get('momentum', 0),
            technical_analysis.get('volume_ratio', 1),
            technical_analysis.get('bb_position', 0.5),
            datetime.now().hour,  # Time of day
            datetime.now().weekday()  # Day of week
        ]
        return [f for f in features if f is not None]
    
    async def train_model(self, historical_data: List[MarketData]):
        """Train the ML model"""
        try:
            if len(historical_data) < 50:
                return
            
            # Prepare training data
            X, y = [], []
            for i in range(10, len(historical_data)):
                # Features: current market state
                features = [
                    historical_data[i].ltp,
                    historical_data[i].volume,
                    historical_data[i].vix,
                    historical_data[i].gap_pct,
                    historical_data[i].timestamp.hour,
                    historical_data[i].timestamp.weekday()
                ]
                
                # Target: next period's price change
                if i + 1 < len(historical_data):
                    target = historical_data[i + 1].ltp
                    X.append(features)
                    y.append(target)
            
            if len(X) > 20:
                X = np.array(X)
                y = np.array(y)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train model
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.model.fit(X_scaled, y)
                self.is_trained = True
                
                self.log_info(f"Model trained with {len(X)} samples")
            
        except Exception as e:
            self.log_error(f"Model training failed: {e}")
    
    def detect_patterns(self, market_data: MarketData, technical_analysis: Dict) -> List[str]:
        """Detect market patterns"""
        patterns = []
        
        # High VIX pattern
        if market_data.vix > 18:
            patterns.append("HIGH_VIX")
        
        # Gap pattern
        if market_data.gap_pct > 0.5:
            patterns.append("SIGNIFICANT_GAP")
        
        # Volume spike
        volume_ratio = technical_analysis.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            patterns.append("VOLUME_SPIKE")
        
        # RSI extremes
        rsi = technical_analysis.get('rsi', 50)
        if rsi < 30:
            patterns.append("RSI_OVERSOLD")
        elif rsi > 70:
            patterns.append("RSI_OVERBOUGHT")
        
        # Trend patterns
        trend = technical_analysis.get('trend', 'NEUTRAL')
        momentum = technical_analysis.get('momentum', 0)
        if trend == 'BULLISH' and momentum > 0.02:
            patterns.append("STRONG_BULLISH_MOMENTUM")
        elif trend == 'BEARISH' and momentum < -0.02:
            patterns.append("STRONG_BEARISH_MOMENTUM")
        
        return patterns

class RiskManagementAgent(BaseAgent):
    """Advanced risk management and position sizing"""
    
    def __init__(self, capital: float):
        super().__init__("RiskManager")
        self.capital = capital
        self.max_risk_per_trade = 0.025  # 2.5%
        self.max_portfolio_risk = 0.05   # 5%
        self.max_daily_trades = 3
        self.daily_loss_limit = 0.05     # 5%
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess and manage risks"""
        try:
            market_data = data.get('market_data')
            current_positions = data.get('current_positions', [])
            daily_pnl = data.get('daily_pnl', 0)
            daily_trades = data.get('daily_trades', 0)
            
            risk_assessment = {
                "can_trade": True,
                "reasons": [],
                "position_size": 1,  # Number of lots
                "max_loss": 0,
                "portfolio_risk": 0
            }
            
            # Check daily limits
            if daily_trades >= self.max_daily_trades:
                risk_assessment["can_trade"] = False
                risk_assessment["reasons"].append("Daily trade limit reached")
            
            if abs(daily_pnl) >= self.capital * self.daily_loss_limit:
                risk_assessment["can_trade"] = False
                risk_assessment["reasons"].append("Daily loss limit reached")
            
            # Calculate current portfolio risk
            total_risk = sum([pos.entry_price * pos.quantity * 75 for pos in current_positions])
            portfolio_risk_pct = total_risk / self.capital
            risk_assessment["portfolio_risk"] = portfolio_risk_pct
            
            if portfolio_risk_pct >= self.max_portfolio_risk:
                risk_assessment["can_trade"] = False
                risk_assessment["reasons"].append("Maximum portfolio risk reached")
            
            # Calculate optimal position size
            if risk_assessment["can_trade"]:
                ml_predictions = data.get('ml_predictions', {})
                technical_analysis = data.get('technical_analysis', {})
                
                # Base position size
                risk_amount = self.capital * self.max_risk_per_trade
                
                # Adjust based on volatility (VIX)
                vix_multiplier = 1.0
                if market_data and market_data.vix > 20:
                    vix_multiplier = 0.7  # Reduce size in high volatility
                elif market_data and market_data.vix < 12:
                    vix_multiplier = 1.3  # Increase size in low volatility
                
                # Adjust based on confidence
                confidence_multiplier = 1.0
                patterns = ml_predictions.get('patterns', [])
                if len(patterns) >= 3:  # Strong signal
                    confidence_multiplier = 1.2
                elif len(patterns) <= 1:  # Weak signal
                    confidence_multiplier = 0.8
                
                # Calculate final position size
                adjusted_risk = risk_amount * vix_multiplier * confidence_multiplier
                risk_assessment["max_loss"] = adjusted_risk
                
                # For options, assume average premium of 250
                estimated_premium = 250
                max_lots = int(adjusted_risk / (estimated_premium * 75))  # 75 is lot size
                risk_assessment["position_size"] = max(1, min(3, max_lots))  # Between 1-3 lots
            
            return {
                "risk_assessment": risk_assessment,
                "status": "success",
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.log_error(f"Risk assessment failed: {e}")
            return {"error": str(e)}

class StrategyAgent(BaseAgent):
    """Main strategy decision maker"""
    
    def __init__(self):
        super().__init__("StrategyMaker")
        self.observation_high = 0
        self.observation_low = 0
        self.observation_phase = False
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make trading decisions based on all inputs"""
        try:
            market_data = data.get('market_data')
            technical_analysis = data.get('technical_analysis', {})
            ml_predictions = data.get('ml_predictions', {})
            risk_assessment = data.get('risk_assessment', {})
            
            if not market_data:
                return {"error": "No market data provided"}
            
            current_time = datetime.now().time()
            
            # Handle observation phase (9:15 - 9:30)
            if time(9, 15) <= current_time <= time(9, 30):
                self.observation_high = max(self.observation_high, market_data.high)
                self.observation_low = min(self.observation_low or 999999, market_data.low)
                self.observation_phase = True
                
                return {
                    "trading_signal": TradingSignal(
                        signal_type="OBSERVE",
                        confidence=0.0,
                        entry_price=0.0,
                        target_price=0.0,
                        stop_loss=0.0,
                        reasons=[f"Observation phase - High: {self.observation_high}, Low: {self.observation_low}"],
                        timestamp=datetime.now()
                    ),
                    "status": "success"
                }
            
            # Check if we can trade
            if not risk_assessment.get("can_trade", False):
                reasons = risk_assessment.get("reasons", ["Risk limits exceeded"])
                return {
                    "trading_signal": TradingSignal(
                        signal_type="HOLD",
                        confidence=0.0,
                        entry_price=0.0,
                        target_price=0.0,
                        stop_loss=0.0,
                        reasons=reasons,
                        timestamp=datetime.now()
                    ),
                    "status": "success"
                }
            
            # Generate trading signal
            signal = self.generate_trading_signal(
                market_data, technical_analysis, ml_predictions, risk_assessment
            )
            
            return {
                "trading_signal": signal,
                "status": "success",
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.log_error(f"Strategy decision failed: {e}")
            return {"error": str(e)}
    
    def generate_trading_signal(self, market_data: MarketData, technical_analysis: Dict, 
                               ml_predictions: Dict, risk_assessment: Dict) -> TradingSignal:
        """Generate comprehensive trading signal"""
        try:
            # Calculate breakout levels
            bullish_level = self.observation_high * 1.0025  # +0.25%
            bearish_level = self.observation_low * 0.9975   # -0.25%
            
            current_price = market_data.ltp
            confidence = 0.0
            reasons = []
            signal_type = "HOLD"
            
            # Check basic filters
            filter_score = 0
            
            # Filter 1: Gap check
            if market_data.gap_pct > 0.2:
                filter_score += 1
                reasons.append(f"Gap: {market_data.gap_pct:.2f}%")
                confidence += 0.2
            
            # Filter 2: VIX check
            if market_data.vix > 11:
                filter_score += 1
                reasons.append(f"VIX: {market_data.vix:.2f}")
                confidence += 0.2
            
            # Filter 3: Volume check
            volume_ratio = technical_analysis.get('volume_ratio', 1)
            if volume_ratio > 1.2:
                filter_score += 1
                reasons.append(f"Volume: {volume_ratio:.1f}x")
                confidence += 0.2
            
            # Need at least 2/3 filters
            if filter_score < 2:
                return TradingSignal(
                    signal_type="HOLD",
                    confidence=confidence,
                    entry_price=0.0,
                    target_price=0.0,
                    stop_loss=0.0,
                    reasons=[f"Filter score: {filter_score}/3"] + reasons,
                    timestamp=datetime.now()
                )
            
            # Check breakout conditions
            if current_price > bullish_level:
                signal_type = "BUY_CALL"
                confidence += 0.3
                reasons.append(f"Bullish breakout: {current_price} > {bullish_level}")
                
                # Add ML confirmation
                patterns = ml_predictions.get('patterns', [])
                if 'STRONG_BULLISH_MOMENTUM' in patterns:
                    confidence += 0.2
                    reasons.append("ML: Strong bullish momentum")
                
                # Add technical confirmation
                if technical_analysis.get('trend') == 'BULLISH':
                    confidence += 0.1
                    reasons.append("Technical: Bullish trend")
                
            elif current_price < bearish_level:
                signal_type = "BUY_PUT"
                confidence += 0.3
                reasons.append(f"Bearish breakout: {current_price} < {bearish_level}")
                
                # Add ML confirmation
                patterns = ml_predictions.get('patterns', [])
                if 'STRONG_BEARISH_MOMENTUM' in patterns:
                    confidence += 0.2
                    reasons.append("ML: Strong bearish momentum")
                
                # Add technical confirmation
                if technical_analysis.get('trend') == 'BEARISH':
                    confidence += 0.1
                    reasons.append("Technical: Bearish trend")
            
            # Estimate prices (simplified)
            estimated_premium = 250  # Base premium
            target_price = estimated_premium * 1.5  # 50% target
            stop_loss = estimated_premium * 0.9     # 10% stop
            
            return TradingSignal(
                signal_type=signal_type,
                confidence=min(1.0, confidence),
                entry_price=estimated_premium,
                target_price=target_price,
                stop_loss=stop_loss,
                reasons=reasons,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.log_error(f"Error generating signal: {e}")
            return TradingSignal(
                signal_type="HOLD",
                confidence=0.0,
                entry_price=0.0,
                target_price=0.0,
                stop_loss=0.0,
                reasons=[f"Error: {e}"],
                timestamp=datetime.now()
            )

class ExecutionAgent(BaseAgent):
    """Handles order execution and position management"""
    
    def __init__(self, smart_api: SmartConnect, capital: float):
        super().__init__("ExecutionAgent")
        self.smart_api = smart_api
        self.capital = capital
        self.current_positions: List[Position] = []
        self.daily_trades = 0
        self.daily_pnl = 0.0
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading decisions"""
        try:
            trading_signal = data.get('trading_signal')
            risk_assessment = data.get('risk_assessment', {})
            
            if not trading_signal:
                return {"error": "No trading signal provided"}
            
            result = {"executed": False, "message": "", "positions": self.current_positions}
            
            # Execute signal
            if trading_signal.signal_type in ["BUY_CALL", "BUY_PUT"] and len(self.current_positions) == 0:
                option_type = 'CE' if trading_signal.signal_type == "BUY_CALL" else 'PE'
                market_data = data.get('market_data')
                
                if market_data:
                    # Find suitable option
                    option = self.select_option(option_type, market_data.ltp)
                    if option:
                        position_size = risk_assessment.get('position_size', 1)
                        order_result = self.place_order(option, position_size, "BUY")
                        
                        if order_result.get('status'):
                            # Create position
                            position = Position(
                                symbol=option['symbol'],
                                token=option['token'],
                                entry_price=option['premium'],
                                current_price=option['premium'],
                                quantity=position_size,
                                option_type=option_type,
                                entry_time=datetime.now(),
                                order_id=order_result.get('order_id', '')
                            )
                            
                            self.current_positions.append(position)
                            self.daily_trades += 1
                            
                            result['executed'] = True
                            result['message'] = f"Opened {option_type} position: {position_size} lots"
                            result['position'] = position
                            
                            self.log_info(f"Position opened: {option['symbol']} - {position_size} lots")
                        else:
                            result['message'] = f"Order failed: {order_result.get('message', 'Unknown error')}"
                    else:
                        result['message'] = "No suitable option found"
            
            # Monitor existing positions
            if self.current_positions:
                await self.monitor_positions()
            
            return {
                "execution_result": result,
                "positions": [asdict(pos) for pos in self.current_positions],
                "daily_pnl": self.daily_pnl,
                "daily_trades": self.daily_trades,
                "status": "success",
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.log_error(f"Execution failed: {e}")
            return {"error": str(e)}
    
    async def monitor_positions(self):
        """Monitor and manage existing positions"""
        positions_to_remove = []
        
        for position in self.current_positions:
            try:
                # Get current option price
                ltp_data = self.smart_api.ltpData("NFO", position.symbol, position.token)
                
                if not ltp_data['status']:
                    continue
                
                current_price = ltp_data['data'].get('ltp', 0)
                position.current_price = current_price
                
                # Calculate P&L
                position.pnl = (current_price - position.entry_price) * position.quantity * 75
                position.pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
                
                # Check exit conditions
                exit_reason = self.check_exit_conditions(position)
                
                if exit_reason:
                    order_result = self.place_order(
                        {
                            'symbol': position.symbol,
                            'token': position.token,
                            'premium': current_price
                        },
                        position.quantity,
                        "SELL"
                    )
                    
                    if order_result.get('status'):
                        self.daily_pnl += position.pnl
                        positions_to_remove.append(position)
                        self.log_info(f"Position closed: {position.symbol} - {exit_reason} - P&L: ₹{position.pnl:.2f}")
                
            except Exception as e:
                self.log_error(f"Error monitoring position {position.symbol}: {e}")
        
        # Remove closed positions
        for position in positions_to_remove:
            self.current_positions.remove(position)
    
    def check_exit_conditions(self, position: Position) -> Optional[str]:
        """Check if position should be exited"""
        current_price = position.current_price
        entry_price = position.entry_price
        pnl_pct = position.pnl_pct
        
        # 10% stop loss
        if pnl_pct <= -10:
            return "Stop loss triggered"
        
        # Profit lock logic
        if pnl_pct >= 15 and not position.profit_locked:
            position.profit_locked = True
            position.stop_loss_price = entry_price * 1.10  # Lock 10% profit
            return None
        
        # Check profit-locked stop
        if position.profit_locked and position.stop_loss_price:
            if current_price <= position.stop_loss_price:
                return "Profit lock stop triggered"
        
        # 50% profit target
        if pnl_pct >= 50:
            if position.quantity == 1:
                return "50% profit target (full position)"
            else:
                # Book half position
                if not position.partial_booked:
                    half_quantity = position.quantity // 2
                    self.place_order(
                        {
                            'symbol': position.symbol,
                            'token': position.token,
                            'premium': current_price
                        },
                        half_quantity,
                        "SELL"
                    )
                    position.quantity -= half_quantity
                    position.partial_booked = True
                    partial_pnl = (current_price - entry_price) * half_quantity * 75
                    self.daily_pnl += partial_pnl
                    return None
        
        # 100% profit target (remaining position)
        if pnl_pct >= 100 and position.quantity > 0:
            return "100% profit target"
        
        # Trailing stop logic
        if current_price > entry_price:
            profit_points = current_price - entry_price
            ten_point_moves = int(profit_points / 10)
            
            if ten_point_moves > 0:
                trailing_stop = entry_price + (ten_point_moves * 5)
                if position.trailing_stop is None or trailing_stop > position.trailing_stop:
                    position.trailing_stop = trailing_stop
                
                if current_price <= position.trailing_stop:
                    return "Trailing stop triggered"
        
        # Time-based exit (near market close)
        current_time = datetime.now().time()
        if current_time >= time(15, 25):
            return "Market closing"
        
        return None
    
    def select_option(self, option_type: str, current_price: float) -> Optional[Dict]:
        """Select appropriate option"""
        try:
            options = self.get_option_chain()
            suitable_options = []
            
            for option in options:
                if option.get('option_type') != option_type:
                    continue
                
                premium = option.get('premium', 0)
                strike = option.get('strike', 0)
                
                # Check premium range (200-300)
                if not (200 <= premium <= 300):
                    continue
                
                # Check if ATM or 1 OTM
                if option_type == 'CE':
                    if current_price <= strike <= current_price + 100:
                        suitable_options.append(option)
                else:  # PE
                    if current_price - 100 <= strike <= current_price:
                        suitable_options.append(option)
            
            if suitable_options:
                # Sort by closeness to ATM
                suitable_options.sort(key=lambda x: abs(x['strike'] - current_price))
                return suitable_options[0]
            
            return None
            
        except Exception as e:
            self.log_error(f"Error selecting option: {e}")
            return None
    
    def get_option_chain(self) -> List[Dict]:
        """Get option chain from Angel One"""
        try:
            # Get current expiry Thursday
            current_date = datetime.now()
            days_until_thursday = (3 - current_date.weekday()) % 7
            if days_until_thursday == 0 and current_date.weekday() == 3:
                days_until_thursday = 7
            
            expiry_date = current_date + timedelta(days=days_until_thursday)
            expiry_str = expiry_date.strftime('%d%b%Y').upper()
            
            # Search for options
            search_data = self.smart_api.searchScrip("NFO", f"NIFTY{expiry_str}")
            
            if not search_data['status']:
                return []
            
            options = []
            for scrip in search_data['data'][:50]:  # Limit to first 50 results
                symbol_name = scrip.get('symbol', '')
                if 'CE' not in symbol_name and 'PE' not in symbol_name:
                    continue
                
                token = scrip.get('symboltoken', '')
                ltp_data = self.smart_api.ltpData("NFO", symbol_name, token)
                
                if ltp_data['status']:
                    premium = ltp_data['data'].get('ltp', 0)
                    
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
            
        except Exception as e:
            self.log_error(f"Error fetching option chain: {e}")
            return []
    
    def place_order(self, option: Dict, quantity: int, transaction_type: str) -> Dict:
        """Place order using Angel One API"""
        try:
            lot_size = 75  # NIFTY lot size
            total_quantity = quantity * lot_size
            
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": option['symbol'],
                "symboltoken": option['token'],
                "transactiontype": transaction_type,
                "exchange": "NFO",
                "ordertype": "MARKET",  # Use market orders for better execution
                "producttype": "INTRADAY",
                "duration": "DAY",
                "price": "0",  # Market order
                "squareoff": "0",
                "stoploss": "0",
                "quantity": str(total_quantity)
            }
            
            order_result = self.smart_api.placeOrder(order_params)
            
            if order_result['status']:
                return {
                    'status': True,
                    'order_id': order_result['data']['orderid'],
                    'message': order_result['message']
                }
            else:
                return {
                    'status': False,
                    'message': order_result.get('message', 'Order failed')
                }
                
        except Exception as e:
            self.log_error(f"Error placing order: {e}")
            return {'status': False, 'message': str(e)}

class DatabaseAgent(BaseAgent):
    """Handles data persistence and analytics"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        super().__init__("DatabaseAgent")
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Market data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    ltp REAL,
                    open REAL,
                    high REAL,
                    low REAL,
                    volume INTEGER,
                    vix REAL,
                    gap_pct REAL
                )
            ''')
            
            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    option_type TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity INTEGER,
                    pnl REAL,
                    pnl_pct REAL,
                    exit_reason TEXT,
                    signal_confidence REAL
                )
            ''')
            
            # Signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    signal_type TEXT,
                    confidence REAL,
                    reasons TEXT,
                    executed BOOLEAN
                )
            ''')
            
            conn.commit()
            conn.close()
            self.log_info("Database initialized successfully")
            
        except Exception as e:
            self.log_error(f"Database initialization failed: {e}")
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Store data in database"""
        try:
            # Store market data
            market_data = data.get('market_data')
            if market_data:
                self.store_market_data(market_data)
            
            # Store signals
            trading_signal = data.get('trading_signal')
            if trading_signal:
                self.store_signal(trading_signal, data.get('execution_result', {}).get('executed', False))
            
            # Store completed trades
            execution_result = data.get('execution_result', {})
            if execution_result.get('executed') and 'position' in execution_result:
                # This will be called when position is closed
                pass
            
            return {"status": "success", "message": "Data stored successfully"}
            
        except Exception as e:
            self.log_error(f"Database storage failed: {e}")
            return {"error": str(e)}
    
    def store_market_data(self, market_data: MarketData):
        """Store market data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_data (timestamp, symbol, ltp, open, high, low, volume, vix, gap_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                market_data.timestamp,
                market_data.symbol,
                market_data.ltp,
                market_data.open,
                market_data.high,
                market_data.low,
                market_data.volume,
                market_data.vix,
                market_data.gap_pct
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.log_error(f"Error storing market data: {e}")
    
    def store_signal(self, signal: TradingSignal, executed: bool):
        """Store trading signal"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO signals (timestamp, signal_type, confidence, reasons, executed)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                signal.timestamp,
                signal.signal_type,
                signal.confidence,
                json.dumps(signal.reasons),
                executed
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.log_error(f"Error storing signal: {e}")
    
    def store_trade(self, position: Position, exit_price: float, exit_reason: str, confidence: float):
        """Store completed trade"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (timestamp, symbol, option_type, entry_price, exit_price, 
                                  quantity, pnl, pnl_pct, exit_reason, signal_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.entry_time,
                position.symbol,
                position.option_type,
                position.entry_price,
                exit_price,
                position.quantity,
                position.pnl,
                position.pnl_pct,
                exit_reason,
                confidence
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.log_error(f"Error storing trade: {e}")
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total trades
            cursor.execute("SELECT COUNT(*) FROM trades")
            total_trades = cursor.fetchone()[0]
            
            # Win rate
            cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0")
            winning_trades = cursor.fetchone()[0]
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Total P&L
            cursor.execute("SELECT SUM(pnl) FROM trades")
            total_pnl = cursor.fetchone()[0] or 0
            
            # Average P&L per trade
            avg_pnl = (total_pnl / total_trades) if total_trades > 0 else 0
            
            # Best and worst trades
            cursor.execute("SELECT MAX(pnl), MIN(pnl) FROM trades")
            best_trade, worst_trade = cursor.fetchone()
            
            conn.close()
            
            return {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl_per_trade": avg_pnl,
                "best_trade": best_trade or 0,
                "worst_trade": worst_trade or 0
            }
            
        except Exception as e:
            self.log_error(f"Error getting analytics: {e}")
            return {}

class AgenticOptionsTrader:
    """Main agentic trading system orchestrator"""
    
    def __init__(self, api_key: str, client_code: str, password: str, totp_secret: str, capital: float = 100000):
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp_secret = totp_secret
        self.capital = capital
        
        # Initialize SmartConnect
        self.smart_api = SmartConnect(api_key=self.api_key)
        self.is_logged_in = False
        
        # Initialize agents
        self.agents = {}
        self.initialize_agents()
        
        # Control variables
        self.is_running = False
        self.market_open = time(9, 15)
        self.market_close = time(15, 30)
        
        self.log_info("Agentic Options Trader initialized")
    
    def initialize_agents(self):
        """Initialize all trading agents"""
        try:
            self.agents['data_collector'] = DataCollectionAgent(self.smart_api)
            self.agents['technical_analyst'] = TechnicalAnalysisAgent()
            self.agents['ml_predictor'] = MLPredictionAgent()
            self.agents['risk_manager'] = RiskManagementAgent(self.capital)
            self.agents['strategy_maker'] = StrategyAgent()
            self.agents['execution_agent'] = ExecutionAgent(self.smart_api, self.capital)
            self.agents['database_agent'] = DatabaseAgent()
            
            logger.info("All agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise
    
    def login_to_angel_one(self) -> bool:
        """Login to Angel One"""
        try:
            totp = pyotp.TOTP(self.totp_secret).now()
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
    
    def log_info(self, message: str):
        """Log info message"""
        logger.info(f"[AgenticTrader] {message}")
    
    def log_error(self, message: str):
        """Log error message"""
        logger.error(f"[AgenticTrader] {message}")
    
    async def run_agent_pipeline(self) -> Dict[str, Any]:
        """Run the complete agent pipeline"""
        try:
            results = {}
            
            # 1. Data Collection
            data_result = await self.agents['data_collector'].process({})
            if 'error' in data_result:
                return data_result
            
            results.update(data_result)
            
            # 2. Technical Analysis
            tech_result = await self.agents['technical_analyst'].process({
                'market_data': results.get('market_data'),
                'data_collector': self.agents['data_collector']
            })
            results.update(tech_result)
            
            # 3. ML Predictions
            ml_result = await self.agents['ml_predictor'].process({
                'market_data': results.get('market_data'),
                'technical_analysis': results.get('technical_analysis'),
                'data_collector': self.agents['data_collector']
            })
            results.update(ml_result)
            
            # 4. Risk Assessment
            risk_result = await self.agents['risk_manager'].process({
                'market_data': results.get('market_data'),
                'current_positions': self.agents['execution_agent'].current_positions,
                'daily_pnl': self.agents['execution_agent'].daily_pnl,
                'daily_trades': self.agents['execution_agent'].daily_trades,
                'ml_predictions': results.get('ml_predictions'),
                'technical_analysis': results.get('technical_analysis')
            })
            results.update(risk_result)
            
            # 5. Strategy Decision
            strategy_result = await self.agents['strategy_maker'].process({
                'market_data': results.get('market_data'),
                'technical_analysis': results.get('technical_analysis'),
                'ml_predictions': results.get('ml_predictions'),
                'risk_assessment': results.get('risk_assessment')
            })
            results.update(strategy_result)
            
            # 6. Execution
            execution_result = await self.agents['execution_agent'].process({
                'trading_signal': results.get('trading_signal'),
                'risk_assessment': results.get('risk_assessment'),
                'market_data': results.get('market_data')
            })
            results.update(execution_result)
            
            # 7. Data Storage
            await self.agents['database_agent'].process(results)
            
            return results
            
        except Exception as e:
            self.log_error(f"Agent pipeline failed: {e}")
            return {"error": str(e)}
    
    def display_status(self, results: Dict[str, Any]):
        """Display current status"""
        try:
            market_data = results.get('market_data')
            trading_signal = results.get('trading_signal')
            execution_result = results.get('execution_result', {})
            positions = results.get('positions', [])
            
            print("\n" + "="*80)
            print(f"AGENTIC OPTIONS TRADER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            if market_data:
                print(f"NIFTY: {market_data.ltp:.2f} | VIX: {market_data.vix:.2f} | Gap: {market_data.gap_pct:.2f}%")
            
            if trading_signal:
                print(f"Signal: {trading_signal.signal_type} | Confidence: {trading_signal.confidence:.2f}")
                print(f"Reasons: {', '.join(trading_signal.reasons)}")
            
            print(f"Daily P&L: ₹{results.get('daily_pnl', 0):.2f} | Trades: {results.get('daily_trades', 0)}")
            
            if positions:
                print(f"\nACTIVE POSITIONS ({len(positions)}):")
                for pos in positions:
                    print(f"  {pos['symbol']}: ₹{pos['pnl']:.2f} ({pos['pnl_pct']:.1f}%)")
            
            if execution_result.get('executed'):
                print(f"✅ EXECUTED: {execution_result['message']}")
            
            print("="*80)
            
        except Exception as e:
            self.log_error(f"Status display error: {e}")
    
    async def run_strategy(self):
        """Main strategy execution loop"""
        self.log_info("Starting Agentic Options Trading Strategy")
        
        # Login to Angel One
        if not self.login_to_angel_one():
            self.log_error("Failed to login to Angel One. Exiting.")
            return
        
        self.is_running = True
        
        try:
            while self.is_running:
                current_time = datetime.now().time()
                
                # Check market hours
                if current_time < self.market_open:
                    self.log_info("Market not open yet. Waiting...")
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue
                
                if current_time >= self.market_close:
                    self.log_info("Market closed. Stopping strategy.")
                    break
                
                # Run agent pipeline
                results = await self.run_agent_pipeline()
                
                if 'error' in results:
                    self.log_error(f"Pipeline error: {results['error']}")
                    await asyncio.sleep(180)  # Wait 3 minutes on error
                    continue
                
                # Display status
                self.display_status(results)
                
                # Wait before next iteration
                await asyncio.sleep(180)  # 3 minutes between cycles
                
        except KeyboardInterrupt:
            self.log_info("Strategy stopped by user")
        except Exception as e:
            self.log_error(f"Strategy error: {e}")
        finally:
            # Close all positions before exit
            await self.close_all_positions()
            self.is_running = False
    
    async def close_all_positions(self):
        """Close all open positions"""
        try:
            execution_agent = self.agents['execution_agent']
            for position in execution_agent.current_positions:
                order_result = execution_agent.place_order(
                    {
                        'symbol': position.symbol,
                        'token': position.token,
                        'premium': position.current_price
                    },
                    position.quantity,
                    "SELL"
                )
                
                if order_result.get('status'):
                    self.log_info(f"Position closed: {position.symbol}")
            
            execution_agent.current_positions.clear()
            
        except Exception as e:
            self.log_error(f"Error closing positions: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        try:
            return self.agents['database_agent'].get_performance_analytics()
        except Exception as e:
            self.log_error(f"Performance report error: {e}")
            return {}

# Usage Example
async def main():
    """Main execution function"""
    
    # Angel One API credentials
    API_KEY = "FkpWlaE9"    
    CLIENT_CODE = "SPHOA1034"  
    PASSWORD = "0509"
    TOTP_SECRET = "7CWRXGUI2AB364N43NGHQNNHJY"
    CAPITAL = 100000  # ₹1 lakh
    
    print("🤖 AGENTIC OPTIONS TRADING SYSTEM")
    print("=" * 50)
    print("⚠️  IMPORTANT WARNINGS:")
    print("1. This is educational code - Test in paper trading first!")
    print("2. Options trading involves substantial risk of loss")
    print("3. Only trade with money you can afford to lose")
    print("4. Ensure you have sufficient margin (recommended ₹75K+)")
    print("5. Install required packages:")
    print("   pip install smartapi-python pyotp pandas scikit-learn")
    print("   Optional (for advanced indicators): pip install TA-Lib")
    print("=" * 50)
    
    # Confirm before running
    confirm = input("\n🔍 Have you tested this in paper trading? (yes/no): ").lower()
    if confirm != 'yes':
        print("❌ Please test in paper trading first. Exiting...")
        return
    
    try:
        # Initialize agentic trader
        trader = AgenticOptionsTrader(API_KEY, CLIENT_CODE, PASSWORD, TOTP_SECRET, CAPITAL)
        
        # Run the strategy
        await trader.run_strategy()
        
        # Show final performance report
        print("\n📊 FINAL PERFORMANCE REPORT:")
        print("=" * 50)
        performance = trader.get_performance_report()
        for key, value in performance.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
    except KeyboardInterrupt:
        print("\n🛑 Strategy stopped by user.")
    except Exception as e:
        print(f"💥 Strategy error: {e}")
    
    print("\n✅ Agentic strategy execution completed.")

if __name__ == "__main__":
    # Run the agentic strategy
    asyncio.run(main())