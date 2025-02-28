# trading_system/src/strategies/adx_strat.py

import numpy as np
import pandas as pd
from numba import njit
from typing import Optional, Dict
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager


class ADXStrategy(BaseStrategy):
    """
    ADX Strategy for algorithmic trading that calculates the Average Directional Index (ADX),
    plus and minus Directional Indicators (DI+ and DI–), and generates trading signals based on trend
    strength. This strategy integrates risk management with stop loss, take profit, slippage, and
    transaction cost adjustments. It is optimized for hyperparameter tuning and performance evaluation.
    
    Mathematical Explanation:
    ---------------------------
    Let:
      TR(i) = max( high(i) - low(i), |high(i) - close(i-1)|, |low(i) - close(i-1)| )
      plusDM(i) = (high(i) - high(i-1)) if (high(i) - high(i-1)) > (low(i-1) - low(i)) and > 0 else 0
      minusDM(i) = (low(i-1) - low(i)) if (low(i-1) - low(i)) > (high(i) - high(i-1)) and > 0 else 0

    Using Wilder's smoothing (with period window):
      ATR = smooth(TR, window)
      DI+ = 100 * smooth(plusDM, window) / (ATR + epsilon)
      DI– = 100 * smooth(minusDM, window) / (ATR + epsilon)
      DX = 100 * |DI+ - DI–| / (DI+ + DI– + epsilon)
      ADX = smooth(DX, window)
    
    A trading signal is generated when ADX crosses above 25:
      - Long Signal: ADX crosses above 25 and DI+ > DI–
      - Short Signal: ADX crosses above 25 and DI– > DI+
    
    Signal strength is defined as:
      signal_strength = (DI+ - DI–) * ADX / 100
    """

    def __init__(self, db_config, params: Optional[Dict] = None):
        """
        Initialize the ADX strategy with database configuration and strategy parameters.

        Args:
            db_config: Database configuration object.
            params (Dict, optional): Dictionary containing strategy hyperparameters.
                Expected keys include:
                    - 'stop_loss_pct': Stop loss percentage (default 0.05)
                    - 'take_profit_pct': Take profit percentage (default 0.10)
                    - 'slippage': Slippage percentage (default 0.001)
                    - 'transaction_cost': Transaction cost percentage (default 0.001)
                    - 'adx_lookback': Lookback period for ADX calculation (default 14)
        """
        super().__init__(db_config, params)
        self.default_lookback = 14  # Default ADX period if not provided
        self.risk_manager = RiskManager(
            stop_loss_pct=self.params.get('stop_loss_pct', 0.05),
            take_profit_pct=self.params.get('take_profit_pct', 0.10),
            slippage_pct=self.params.get('slippage', 0.001),
            transaction_cost_pct=self.params.get('transaction_cost', 0.001)
        )

    def generate_signals(self, ticker: str, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals and risk-managed trades for the specified ticker.
        
        Retrieves historical price data, calculates ADX and directional indicators using a 
        vectorized and Numba-accelerated function, generates trading signals based on ADX 
        crossover conditions, and applies risk management including stop loss and take profit.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date for backtesting (YYYY-MM-DD). If None, 
                the method uses a lookback period sufficient for stable indicator calculation.
            end_date (str, optional): End date for backtesting (YYYY-MM-DD).
            initial_position (int): Initial trading position (0 = no position, 1 = long, -1 = short).
            latest_only (bool): If True, returns only the latest signal for potential EOD decision.
        
        Returns:
            pd.DataFrame: DataFrame containing price data, generated signals, risk management metrics,
                          and additional performance calculations (e.g., strategy returns). When latest_only is True, 
                          only the most recent entry is returned.
        """
        # Retrieve price data with sufficient lookback periods for indicator calculation.
        df = self._get_price_data(ticker, start_date, end_date)
        
        # Calculate ADX and Directional Indicators using NumPy arrays.
        adx_lookback = self.params.get('adx_lookback', self.default_lookback)
        df['plus_di'], df['minus_di'], df['adx'] = self._calculate_adx(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            adx_lookback
        )
        
        # Generate raw trading signals based on ADX crossover conditions.
        signals = self._generate_vectorized_signals(df, adx_lookback)
        df = pd.concat([df, signals], axis=1)
        
        # Apply vectorized risk management (stop loss, take profit, slippage, and transaction cost).
        df = self.risk_manager.apply(df, initial_position)
        
        # Calculate returns from close price changes and strategy returns by applying the position lag.
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['position'].shift() * df['returns']
        
        return df.iloc[-1:] if latest_only else df

    def _get_price_data(self, ticker: str, 
                        start_date: Optional[str], 
                        end_date: Optional[str]) -> pd.DataFrame:
        """
        Retrieve historical price data for the given ticker and validate data sufficiency for ADX calculation.
        
        This method ensures that a sufficient number of data points is retrieved, either based on a specific
        date range or a default lookback needed for stable ADX indicator computation.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date (YYYY-MM-DD). If None, a lookback period is used.
            end_date (str, optional): End date (YYYY-MM-DD).
        
        Returns:
            pd.DataFrame: DataFrame containing historical price data with columns:
                          'open', 'high', 'low', 'close', 'volume'.
        
        Raises:
            DataRetrievalError: If insufficient data is available for analysis.
        """
        adx_lookback = self.params.get('adx_lookback', self.default_lookback)
        required_lookback = 2 * adx_lookback + 1  # Buffer for stable ADX calculation
        
        # Retrieve historical prices using the base class method.
        df = self.get_historical_prices(
            ticker=ticker,
            from_date=start_date,
            to_date=end_date,
            lookback=required_lookback if start_date is None else None
        )
        
        # Validate that there are enough records for computing the indicators.
        min_records = 2 * adx_lookback + 10  # Buffer for reliable indicator computation
        if not self._validate_data(df, min_records):
            raise DataRetrievalError("Insufficient data for ADX calculation")
            
        return df[['open', 'high', 'low', 'close', 'volume']]

    @staticmethod
    @njit
    def _calculate_adx(high: np.ndarray, low: np.ndarray, 
                       close: np.ndarray, lookback: int) -> tuple:
        """
        Calculate the Plus Directional Indicator (DI+), Minus Directional Indicator (DI–), and 
        Average Directional Index (ADX) using vectorized operations accelerated with Numba.
        
        Wilder's smoothing technique is used to compute the True Range (ATR), directional movements,
        and subsequently the directional indicators.
        
        Args:
            high (np.ndarray): Array of high prices.
            low (np.ndarray): Array of low prices.
            close (np.ndarray): Array of close prices.
            lookback (int): Lookback period for smoothing and indicator calculation.
        
        Returns:
            tuple: A tuple containing three arrays:
                - plus_di (np.ndarray): Array of DI+ values.
                - minus_di (np.ndarray): Array of DI– values.
                - adx (np.ndarray): Array of ADX values.
        """
        n = len(high)
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        # Calculate True Range (TR) and Directional Movements.
        for i in range(1, n):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff

        # Wilder's smoothing function.
        def smooth(series, window):
            smoothed = np.zeros(series.shape[0])
            sum_val = 0.0
            # Calculate initial average
            for j in range(window):
                sum_val += series[j]
            smoothed[window-1] = sum_val / window
            # Smooth the remaining data points.
            for j in range(window, series.shape[0]):
                smoothed[j] = (smoothed[j-1]*(window-1) + series[j]) / window
            return smoothed

        # Calculate Average True Range using smoothing.
        atr = smooth(tr, lookback)
        # Avoid division by zero by adding a small epsilon.
        epsilon = 1e-10
        atr_safe = atr + epsilon
        
        # Calculate smoothed directional movements.
        plus_di_smoothed = smooth(plus_dm, lookback)
        minus_di_smoothed = smooth(minus_dm, lookback)
        
        plus_di = 100 * (plus_di_smoothed / atr_safe)
        minus_di = 100 * (minus_di_smoothed / atr_safe)
        
        # Compute DX and ADX, ensuring denominators are safe from division by zero.
        diff_di = np.abs(plus_di - minus_di)
        sum_di = plus_di + minus_di
        # Replace zeros in the denominator with epsilon.
        sum_di_safe = np.where(sum_di == 0, epsilon, sum_di)
        
        dx = 100 * (diff_di / sum_di_safe)
        adx = smooth(dx, lookback)
        
        return plus_di, minus_di, adx

    def _generate_vectorized_signals(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """
        Generate trading signals based on ADX crossover conditions using vectorized operations.
        
        A signal to enter a trade is generated when ADX crosses above the 25 threshold:
          - Long Signal: When ADX crosses above 25 and DI+ > DI–
          - Short Signal: When ADX crosses above 25 and DI– > DI+
        
        Signal strength is calculated as the normalized difference between DI's weighted by ADX.
        
        Args:
            df (pd.DataFrame): DataFrame containing at least the columns 'adx', 'plus_di', and 'minus_di'.
            lookback (int): Lookback period for calculations (not directly used in the signal logic).
        
        Returns:
            pd.DataFrame: DataFrame with the following columns:
                - 'signal': Trading signal (1 for long, -1 for short, 0 for no action).
                - 'signal_strength': Numeric indicator of the strength of the signal.
        """
        signals = pd.DataFrame(index=df.index)
        adx = df['adx']
        pdi = df['plus_di']
        mdi = df['minus_di']
        
        # Identify the ADX cross: a transition from below 25 to at or above 25.
        adx_cross = (adx.shift(1) < 25) & (adx >= 25)
        # Generate long and short signals based on the DI comparison.
        long_signal = adx_cross & (pdi > mdi)
        short_signal = adx_cross & (mdi > pdi)
        
        # Calculate signal strength as a normalized, weighted difference.
        signal_strength = (pdi - mdi) * adx / 100
        
        signals['signal'] = 0
        signals.loc[long_signal, 'signal'] = 1
        signals.loc[short_signal, 'signal'] = -1
        signals['signal_strength'] = signal_strength
        
        return signals