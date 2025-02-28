# trading_system/src/strategies/keltner_channel.py

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager
from src.database.config import DatabaseConfig
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from datetime import timedelta

class KeltnerChannelStrategy(BaseStrategy):
    """
    Keltner Channel Strategy generates trading signals based on the Keltner Channel indicator.
    
    Mathematical Overview:
        - Computes the exponential moving average (EMA) of the closing price to form the channel middle (kc_middle):
              EMA(t) = α * P(t) + (1 - α) * EMA(t-1), where α = 2/(span+1)
        - Calculates the True Range (TR) as:
              TR = max(high - low, |high - prev_close|, |low - prev_close|)
        - Computes the Average True Range (ATR) (using an EMA over a period 'atr_span'):
              ATR = EMA(TR)
        - Determines channel boundaries:
              kc_upper = kc_middle + multiplier * ATR
              kc_lower = kc_middle - multiplier * ATR
        - Signal Generation:
            • Buy signal (+1) is generated when the close is below kc_lower and shows upward momentum (close > previous close).
            • Sell signal (-1) is generated when the close is above kc_upper and shows downward momentum (close < previous close).
        - Signal Strength:
            • For a buy signal: (kc_lower – close) / ATR (the further below the lower band, the stronger the oversold signal).
            • For a sell signal: (close – kc_upper) / ATR (the further above the upper band, the stronger the overbought signal).
    
    Risk Management:
        The strategy then applies risk management via the RiskManager class, which adjusts the entry prices for slippage 
        and transaction costs and applies stop loss and take profit rules. This produces a final backtesting DataFrame 
        that includes positions, realized returns, cumulative returns, and the exit type.
    
    Strategy Flow:
        1. Data Retrieval: Fetches historical price data with an optimal lookback period.
        2. Indicator Calculation: Computes EMA, ATR, and Keltner Channel boundaries using vectorized operations.
        3. Signal Generation: Determines trading signals and a normalized signal_strength.
        4. Risk Management: Adjusts trading signals by applying stop-loss, take-profit, slippage, and transaction cost rules.
        5. Output: Returns a comprehensive DataFrame that can be used for further performance analysis (e.g., Sharpe ratio, drawdowns)
           and for generating the latest EOD signal.
    
    Parameters:
        db_config (DatabaseConfig): Database configuration settings.
        params (dict, optional): Strategy-specific hyperparameters.
        
    Hyperparameters in `params`:
        - kc_span (int): EMA period for channel middle (default: 20)
        - atr_span (int): Period for ATR calculation (default: 10)
        - multiplier (float): Factor to scale the ATR for channel boundaries (default: 2.0)
        - long_only (bool): If True, generate long-only signals (default: True)
        - stop_loss (float): Stop loss percentage (default: 0.05)
        - take_profit (float): Take profit percentage (default: 0.10)
        - slippage (float): Slippage percentage (default: 0.001)
        - transaction_cost (float): Transaction cost percentage (default: 0.001)
    """
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the Keltner Channel Strategy with the given database configuration and strategy parameters.
        
        Args:
            db_config (DatabaseConfig): The database configuration object.
            params (dict, optional): A dictionary of strategy-specific hyperparameters.
        """
        super().__init__(db_config, params)
        self._validate_params()
        self.risk_manager = self._init_risk_manager()

    def _validate_params(self):
        """
        Validate strategy parameters and set default values as needed.
        
        Raises:
            ValueError: If any non-risk parameter is non-positive or if risk parameters are negative.
        """
        self.params.setdefault('kc_span', 20)
        self.params.setdefault('atr_span', 10)
        self.params.setdefault('multiplier', 2.0)
        self.params.setdefault('stop_loss', 0.05)
        self.params.setdefault('take_profit', 0.10)
        self.params.setdefault('slippage', 0.001)
        self.params.setdefault('transaction_cost', 0.001)
        self.params.setdefault('long_only', True)

        if any(v <= 0 for k, v in self.params.items() if k not in ['stop_loss', 'take_profit']):
            raise ValueError("All strategy parameters must be positive")
        if self.params['stop_loss'] < 0 or self.params['take_profit'] < 0:
            raise ValueError("Risk parameters cannot be negative")

    def _init_risk_manager(self):
        """
        Initialize the RiskManager instance configured with strategy-specific risk parameters.
        
        Returns:
            RiskManager: A risk manager instance for handling stop loss, take profit, slippage, and transaction costs.
        """
        return RiskManager(
            stop_loss_pct=self.params['stop_loss'],
            take_profit_pct=self.params['take_profit'],
            slippage_pct=self.params['slippage'],
            transaction_cost_pct=self.params['transaction_cost']
        )

    def generate_signals(self, ticker: str, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals with integrated risk management.
        
        The method retrieves historical prices, computes technical indicators and channel boundaries,
        generates raw buy/sell signals (with associated normalized signal strength), applies risk management
        adjustments to factor in slippage and transaction costs, and finally formats the data for backtesting
        or for returning the latest signal.
        
        Args:
            ticker (str): Stock ticker.
            start_date (str, optional): Start date (YYYY-MM-DD) for backtesting.
            end_date (str, optional): End date (YYYY-MM-DD) for backtesting.
            initial_position (int): The starting position (0 for no position, 1 for long, -1 for short).
            latest_only (bool): If True, only the latest signal (end-of-day) record is returned.
            
        Returns:
            pd.DataFrame: DataFrame containing price data, technical indicators, raw and risk-managed signals,
                          along with positions, returns, cumulative returns, and exit event indicators.
                          
        Raises:
            DataRetrievalError: If an error occurs during signal generation.
        """
        self.logger.info(f"Processing {ticker} with {self.params}")

        # Determine the extra lookback required for stable indicator calculation.
        max_lookback = max(self.params['kc_span'], self.params['atr_span'])
        required_bars = max_lookback * 2  # Buffer period
        
        try:
            prices = self._fetch_data(ticker, start_date, end_date, required_bars, latest_only)
            if prices.empty:
                return pd.DataFrame()

            # Compute the technical indicators and raw signals in a vectorized manner.
            signals = self._calculate_components(prices)
            
            # Apply risk management adjustments to the generated signals.
            risk_managed = self.risk_manager.apply(signals, initial_position)
            
            # Merge the original price data, technical components, and risk-managed trade metrics.
            full_df = self._format_output(prices, signals, risk_managed)
            
            # Filter the final result based on the provided date range or extract just the latest record.
            return self._filter_results(full_df, start_date, end_date, latest_only)

        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            raise DataRetrievalError(f"Failed processing {ticker}") from e

    def _fetch_data(self, ticker: str, start_date: str, end_date: str, 
                    lookback: int, latest_only: bool) -> pd.DataFrame:
        """
        Retrieve historical price data using an optimized lookback window for indicator stability.
        
        If 'latest_only' is True, a fixed number of recent records is fetched;
        Otherwise, the start date is extended backward by 'lookback' days to ensure stable indicator calculations.
        
        Args:
            ticker (str): Stock ticker.
            start_date (str): Backtesting start date (YYYY-MM-DD).
            end_date (str): Backtesting end date (YYYY-MM-DD).
            lookback (int): Number of extra bars to include for indicator calculation.
            latest_only (bool): If True, fetch only the latest records.
            
        Returns:
            pd.DataFrame: Historical price data obtained from the database.
        """
        if latest_only:
            return self.get_historical_prices(ticker, lookback=lookback)

        if start_date and end_date:
            adjusted_start = (pd.to_datetime(start_date) - timedelta(days=lookback)).strftime('%Y-%m-%d')
            return self.get_historical_prices(ticker, from_date=adjusted_start, to_date=end_date)
        
        return self.get_historical_prices(ticker, lookback=lookback)

    def _calculate_components(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators including True Range, ATR, and Keltner Channel boundaries.
        
        The True Range (TR) is calculated as:
            TR = max(high - low, |high - previous close|, |low - previous close|)
        An exponential moving average (EMA) of TR over 'atr_span' produces the ATR,
        and an EMA of the closing price over 'kc_span' gives the channel middle (kc_middle).
        The upper and lower channels are defined by adding/subtracting (multiplier * ATR) to/from kc_middle.
        Raw trading signals are then generated based on whether the price is below the lower channel (buy)
        or above the upper channel (sell). Signal strength is a normalized measure of the deviation.
        
        Args:
            prices (pd.DataFrame): Historical price data containing 'close', 'high', and 'low' columns.
            
        Returns:
            pd.DataFrame: DataFrame containing the computed technical indicators and raw signals.
        """
        close = prices['close']
        high = prices['high']
        low = prices['low']
        
        # Calculate True Range (TR)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        # Compute ATR as the exponential moving average of TR
        atr = tr.ewm(span=self.params['atr_span'], adjust=False).mean()
        
        # EMA of close forms the channel middle (kc_middle)
        kc_middle = close.ewm(span=self.params['kc_span'], adjust=False).mean()
        
        # Define channel boundaries using the multiplier and ATR
        kc_upper = kc_middle + self.params['multiplier'] * atr
        kc_lower = kc_middle - self.params['multiplier'] * atr
        
        # Generate signals: Buy (+1) if below lower band (and trending upward),
        # and Sell (-1) if above upper band (and trending downward).
        buy_cond = (close < kc_lower) & (close > close.shift())
        sell_cond = (close > kc_upper) & (close < close.shift())
        
        signals = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close,
            'signal': np.select([buy_cond, sell_cond], [1, -1], 0),
            # Calculate signal strength as a positive normalized distance from the channel boundary.
            'signal_strength': self._calculate_strength(close, kc_upper, kc_lower, atr),
            'kc_middle': kc_middle,
            'kc_upper': kc_upper,
            'kc_lower': kc_lower
        }, index=prices.index)

        if self.long_only:
            # Override sell signals (-1) with 0 (exit)
            signals.loc[sell_cond, 'signal'] = 0
        
        return signals.dropna()

    def _calculate_strength(self, close: pd.Series, upper: pd.Series,
                              lower: pd.Series, atr: pd.Series) -> pd.Series:
        """
        Calculate normalized signal strength as a measure of the distance from the channel boundary.
        
        For a buy signal (price below kc_lower), the strength is computed as:
              (kc_lower - close) / ATR
        For a sell signal (price above kc_upper), it is computed as:
              (close - kc_upper) / ATR
        The values are positive and indicate the intensity of the signal.
        
        Args:
            close (pd.Series): Series of closing prices.
            upper (pd.Series): Upper channel boundary (kc_upper).
            lower (pd.Series): Lower channel boundary (kc_lower).
            atr (pd.Series): Average True Range.
            
        Returns:
            pd.Series: Normalized signal strengths.
        """
        long_strength = (lower - close) / atr
        short_strength = (close - upper) / atr
        return np.where(close < lower, long_strength, np.where(close > upper, short_strength, 0))

    def _format_output(self, prices: pd.DataFrame, signals: pd.DataFrame,
                       risk_managed: pd.DataFrame) -> pd.DataFrame:
        """
        Consolidate and format output by merging raw price data, technical indicators,
        and risk-managed trading outcomes.
        
        Args:
            prices (pd.DataFrame): Original historical price data.
            signals (pd.DataFrame): Technical indicators and raw signals DataFrame.
            risk_managed (pd.DataFrame): DataFrame with positions, returns, and risk management data.
            
        Returns:
            pd.DataFrame: A complete DataFrame ready for backtesting or further analysis.
        """
        return pd.concat([
            prices[['open', 'high', 'low', 'close']],
            signals[['kc_middle', 'kc_upper', 'kc_lower', 'signal', 'signal_strength']],
            risk_managed[['position', 'return', 'cumulative_return', 'exit_type']]
        ], axis=1).ffill().dropna()

    def _filter_results(self, df: pd.DataFrame, start_date: str, end_date: str,
                        latest_only: bool) -> pd.DataFrame:
        """
        Apply final date filtering and formatting to the output DataFrame.
        
        Args:
            df (pd.DataFrame): The merged DataFrame with backtesting data.
            start_date (str): Start date (YYYY-MM-DD) for filtering.
            end_date (str): End date (YYYY-MM-DD) for filtering.
            latest_only (bool): If True, return only the last record (for end-of-day trading).
            
        Returns:
            pd.DataFrame: Filtered and chronologically sorted DataFrame.
        """
        if start_date and end_date:
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            df = df.loc[mask]
            
        if latest_only:
            return df.tail(1)
            
        return df.sort_index()

    def __repr__(self):
        """
        Return a string representation of the KeltnerChannelStrategy instance showcasing key parameters.
        
        Returns:
            str: A summary of the Keltner Channel Strategy configuration.
        """
        return (f"KeltnerChannelStrategy(ema={self.params['kc_span']}, "
                f"atr={self.params['atr_span']}, mult={self.params['multiplier']})")