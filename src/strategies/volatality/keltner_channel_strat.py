# trading_system/src/strategies/keltner_channel.py

#TODO: TEST THIS CODE

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.database.config import DatabaseConfig
import pandas as pd
import logging
from typing import Dict, Optional

class KeltnerChannelStrategy(BaseStrategy):
    """
    Implements Keltner Channel strategy with database integration.
    
    Hyperparameters:
        kc_lookback (int): EMA period for the middle line (default: 20)
        atr_lookback (int): Lookback period for ATR calculation (default: 10)
        multiplier (float): ATR multiplier for bands (default: 2.0)
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self.params.setdefault('kc_lookback', 20)
        self.params.setdefault('atr_lookback', 10)
        self.params.setdefault('multiplier', 2.0)
        
        # Ensure all hyperparameters are positive
        if any(v <= 0 for v in self.params.values()):
            raise ValueError("All hyperparameters must be positive")

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        """
        Generates trading signals using the Keltner Channel strategy.

        Returns a DataFrame with:
            - close: The closing price
            - signal: Trading signal (1=Buy, -1=Sell, 0=Neutral)
            - position: Position management (1=Long, 0=Out)
            - signal_strength: Volatility-normalized signal strength
            - kc_middle: The middle (EMA) channel
            - kc_upper: The upper channel
            - kc_lower: The lower channel
        """
        self.logger.info(f"Generating signals for {ticker}")
        
        kc_lb = self.params['kc_lookback']
        atr_lb = self.params['atr_lookback']
        
        # You may opt to use this conservative lookback or change to:
        # required_data = max(kc_lb, atr_lb) + 1
        # if you only need a minimal number of rows.
        required_data = kc_lb + atr_lb
        
        try:
            prices = self.get_historical_prices(ticker, lookback=required_data)
            if not self._validate_data(prices, required_data):
                return pd.DataFrame()
            
            # Extract high, low, and close prices
            high, low, close = prices['high'], prices['low'], prices['close']
            prev_close = close.shift(1)
            
            # Calculate the True Range (TR) using vectorized operations
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            
            # Compute the Average True Range (ATR) via EMA
            atr = tr.ewm(alpha=1/atr_lb, adjust=False).mean()
            
            # Calculate Keltner Channel components:
            # Middle line as EMA of close, upper and lower bands offset by ATR*multiplier.
            kc_middle = close.ewm(span=kc_lb, adjust=False).mean()
            kc_upper = kc_middle + self.params['multiplier'] * atr
            kc_lower = kc_middle - self.params['multiplier'] * atr
            
            # Generate trading signals:
            # Buy when the close is below the lower band but recovering from the previous close.
            # Sell when the close is above the upper band but falling from the previous close.
            buy_cond = (close < kc_lower) & (close > prev_close)
            sell_cond = (close > kc_upper) & (close < prev_close)
            signals = pd.Series(0, index=close.index)
            signals[buy_cond] = 1
            signals[sell_cond] = -1
            
            # Determine the trading position:
            # Forward-fill the signals to carry the last non-neutral signal,
            # then clip any negative signals (sell) to 0 to represent being out of the market.
            position = signals.replace(0, method='ffill').clip(lower=0)
            
            # Calculate signal strength (volatility-normalized distance from the channel boundary)
            strength = pd.Series(0.0, index=close.index)
            strength[buy_cond] = (close[buy_cond] - kc_lower[buy_cond]) / atr[buy_cond]
            strength[sell_cond] = (kc_upper[sell_cond] - close[sell_cond]) / atr[sell_cond]
            
            result = pd.DataFrame({
                'close': close,
                'signal': signals,
                'position': position,
                'signal_strength': strength,
                'kc_middle': kc_middle,
                'kc_upper': kc_upper,
                'kc_lower': kc_lower
            }).dropna().sort_index()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed generating signals for {ticker}: {str(e)}")
            # Re-raise the exception using 'from e' to maintain the original traceback.
            raise DataRetrievalError(f"Signal generation failed for {ticker}") from e