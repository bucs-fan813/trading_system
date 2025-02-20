# trading_system/src/strategies/cci_oscillator_strat.py

import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.strategies.base_strat import BaseStrategy
from src.database.config import DatabaseConfig
from src.strategies.risk_management import RiskManager

class CCIStrategy(BaseStrategy):
    """
    Commodity Channel Index (CCI) Strategy with Integrated Risk Management
    
    This strategy calculates the CCI indicator based on the typical price, its moving average,
    and its mean absolute deviation. It generates trading signals when the CCI crosses preset
    threshold levels and then refines these signals using a RiskManager component that incorporates
    stop loss, take profit, slippage, and transaction cost adjustments.
    
    The final output provides a comprehensive DataFrame including price data, the computed CCI,
    raw and risk-managed signals, and performance metrics for full backtesting or a single (latest)
    signal to support end-of-day decision making.
    
    Attributes:
        db_config (DatabaseConfig): Database configuration settings.
        params (dict): Dictionary of strategy parameters.
        risk_manager (RiskManager): Instance of the risk management handler.
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the CCIStrategy with the provided database configuration and parameters.
        
        Args:
            db_config (DatabaseConfig): Settings for database connectivity.
            params (dict, optional): A dictionary of strategy parameters. Expected keys include:
                - cci_period (int): The period over which the CCI is computed.
                - cci_upper_band (float): Upper threshold for triggering a sell signal.
                - cci_lower_band (float): Lower threshold for triggering a buy signal.
                - lookback_period (int): Number of days to fetch for a full backtest.
                - data_source (str): Data source identifier (default: "yfinance").
                - stop_loss_pct (float): Percentage drop from entry price to trigger a stop loss.
                - take_profit_pct (float): Percentage rise from entry price to trigger a take profit.
                - slippage_pct (float): Estimated market slippage as a percentage of price.
                - transaction_cost_pct (float): Estimated transaction cost as a percentage of price.
                
        Raises:
            ValueError: If provided parameters do not conform to the expected ranges.
        """
        default_params = {
            'cci_period': 20,
            'cci_upper_band': 150,
            'cci_lower_band': -150,
            'lookback_period': 252,
            'data_source': 'yfinance',
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001
        }
        
        # Combine user-supplied parameters with defaults.
        params = default_params | (params or {})
        self._validate_params(params)
        
        super().__init__(db_config, params=params)
        self.risk_manager = RiskManager(
            stop_loss_pct=params['stop_loss_pct'],
            take_profit_pct=params['take_profit_pct'],
            slippage_pct=params['slippage_pct'],
            transaction_cost_pct=params['transaction_cost_pct']
        )

    def _validate_params(self, params: Dict) -> None:
        """
        Validate that the provided strategy parameters are logical and within acceptable ranges.
        
        Args:
            params (dict): Strategy parameters to validate.
            
        Raises:
            ValueError: If any parameter value is found to be invalid.
        """
        if params['cci_upper_band'] <= params['cci_lower_band']:
            raise ValueError("cci_upper_band must be greater than cci_lower_band")
        if params['cci_period'] < 2:
            raise ValueError("cci_period must be at least 2")
        if not (0 < params['stop_loss_pct'] < 1):
            raise ValueError("stop_loss_pct must be between 0 and 1")
        if not (0 < params['take_profit_pct'] < 1):
            raise ValueError("take_profit_pct must be between 0 and 1")

    def generate_signals(self, ticker: str, start_date: Optional[str] = None,
                         end_date: Optional[str] = None, initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate CCI-based trading signals and apply risk management adjustments.
        
        This method conducts the following steps:
          1. Retrieves historical price data (either a full date range for backtesting or only a recent segment
             for an up-to-date signal).
          2. Calculates the CCI indicator and generates raw trading signals based on CCI crossovers.
          3. Passes the signals and price information to the RiskManager, which adjusts entry/exit levels,
             applies stop loss and take profit rules, and computes trade and cumulative returns.
          4. Merges indicator and risk management data into a final DataFrame.
        
        Args:
            ticker (str): The stock ticker symbol.
            start_date (str, optional): Start date (in "YYYY-MM-DD" format) for backtesting.
            end_date (str, optional): End date (in "YYYY-MM-DD" format) for backtesting.
            initial_position (int, optional): Initial position (0 for no position, 1 for long, -1 for short).
            latest_only (bool, optional): If True, only the latest available signal is returned.
        
        Returns:
            pd.DataFrame: A DataFrame containing:
                - close: Closing price.
                - high: High price.
                - low: Low price.
                - cci: Commodity Channel Index value.
                - signal: Generated raw signal (1 for buy, -1 for sell, 0 otherwise).
                - signal_strength: Magnitude of the crossover relative to the threshold.
                - position: Position after risk management adjustment.
                - return: Realized return for a trade on exit.
                - cumulative_return: Running cumulative return.
                - exit_type: Reason for trade exit (stop_loss, take_profit, signal_exit, or none).
        """
        try:
            hist_data = self._get_data(ticker, start_date, end_date, latest_only)
            if not self._validate_data(hist_data):
                return pd.DataFrame()

            # Calculate the CCI and raw signals.
            signals = self._calculate_cci_signals(hist_data)
            
            # Apply risk management (stop loss, take profit, slippage, transaction cost)
            risk_managed = self.risk_manager.apply(signals, initial_position)
            
            # Merge CCI and signal strength with risk-managed performance metrics.
            return self._format_output(signals, risk_managed, latest_only)

        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return pd.DataFrame()

    def _get_data(self, ticker: str, start_date: Optional[str],
                  end_date: Optional[str], latest_only: bool) -> pd.DataFrame:
        """
        Retrieve historical price data from the database.
        
        For a full backtest, the data is fetched between the specified start and end dates using the
        defined lookback period. For generating only the latest signal, a shorter lookback period is
        used (3 * cci_period) to capture only recent, stable data.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date (YYYY-MM-DD) for data retrieval.
            end_date (str, optional): End date (YYYY-MM-DD) for data retrieval.
            latest_only (bool): If True, retrieves only a reduced lookback window for the latest signal.
        
        Returns:
            pd.DataFrame: A DataFrame of historical price data with datetime-index.
        """
        if latest_only:
            return self.get_historical_prices(
                ticker,
                lookback=3 * self.params['cci_period'],
                data_source=self.params['data_source']
            )
        else:
            # Use keyword arguments to correctly pass the from_date and to_date parameters.
            return self.get_historical_prices(
                ticker,
                lookback=self.params['lookback_period'],
                data_source=self.params['data_source'],
                from_date=start_date,
                to_date=end_date
            )

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Ensure that sufficient historical data exists for the CCI calculation.
        
        The function requires at minimum twice the number of data points as the CCI period
        to ensure a stable computation of the rolling averages and deviations.
        
        Args:
            data (pd.DataFrame): Historical price data.
        
        Returns:
            bool: True if there are enough records; False otherwise.
        """
        min_records = 2 * self.params['cci_period']
        if len(data) < min_records:
            self.logger.warning(f"Insufficient data: {len(data)} records, required {min_records}")
            return False
        return True

    def _calculate_cci_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the CCI indicator and generate raw trading signals.
        
        Steps:
          1. Calculate the Typical Price (TP):
                 TP = (High + Low + Close) / 3
          2. Compute the Simple Moving Average (SMA) of TP over the cci_period.
          3. Compute the Mean Absolute Deviation (MAD) of TP over the same period.
          4. Calculate the CCI using:
                 CCI = (TP - SMA) / (0.015 * MAD)
          5. Generate signals:
                 - A buy (1) is triggered when the previous CCI is less than cci_lower_band and
                   the current CCI crosses above it.
                 - A sell (-1) is triggered when the previous CCI is greater than cci_upper_band and
                   the current CCI crosses below it.
          6. Determine the signal strength as the difference between the current CCI and the
             threshold it has just crossed.
        
        Args:
            data (pd.DataFrame): Historical price data with 'high', 'low', and 'close' columns.
        
        Returns:
            pd.DataFrame: A DataFrame containing:
                - close: Closing price.
                - high: High price.
                - low: Low price.
                - cci: Computed CCI value.
                - signal: Raw trading signal (1, -1, or 0).
                - signal_strength: Magnitude difference from the threshold (where applicable).
        """
        # Calculate Typical Price (TP)
        tp = (data['high'] + data['low'] + data['close']) / 3
        
        # Compute the Simple Moving Average (SMA) of TP over the specified period.
        sma = tp.rolling(window=self.params['cci_period']).mean()
        
        # Compute the Mean Absolute Deviation (MAD) of TP.
        mad = tp.rolling(window=self.params['cci_period']).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        
        # Calculate the CCI with the scaling constant 0.015.
        cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
        
        # Store the computed indicator and prepare to generate signals.
        signals = pd.DataFrame({
            'close': data['close'],
            'high': data['high'],
            'low': data['low'],
            'cci': cci,
            'signal': 0
        }, index=data.index)
        
        # Generate trading signals using a vectorized crossover detection.
        prev_cci = signals['cci'].shift()
        buy = (prev_cci < self.params['cci_lower_band']) & (signals['cci'] > self.params['cci_lower_band'])
        sell = (prev_cci > self.params['cci_upper_band']) & (signals['cci'] < self.params['cci_upper_band'])
        
        signals['signal'] = np.select([buy, sell], [1, -1], default=0)
        
        # Compute signal strength based on the threshold crossed.
        signals['signal_strength'] = signals['cci'] - np.where(
            buy, self.params['cci_lower_band'],
            np.where(sell, self.params['cci_upper_band'], np.nan)
        )
        
        return signals

    def _format_output(self, signals: pd.DataFrame, risk_managed: pd.DataFrame, latest_only: bool) -> pd.DataFrame:
        """
        Merge raw signal data with risk-managed performance metrics and output the final result.
        
        The final DataFrame includes:
            - Price data from the risk-managed DataFrame.
            - Indicator data ('cci' and 'signal_strength') from the raw signals.
            - Trading signals, positions, realized returns, cumulative return, and exit types.
        
        Args:
            signals (pd.DataFrame): DataFrame containing the raw indicators and signals.
            risk_managed (pd.DataFrame): DataFrame output from the RiskManager with adjusted positions
                                         and performance metrics.
            latest_only (bool): If True, only the last (most recent) row is returned.
        
        Returns:
            pd.DataFrame: Final merged DataFrame with the columns:
                        [close, high, low, cci, signal, signal_strength, position, return,
                        cumulative_return, exit_type]
        """
        merged = risk_managed.join(signals[['cci', 'signal_strength']])
        cols = ['close', 'high', 'low', 'cci', 'signal', 'signal_strength',
                'position', 'return', 'cumulative_return', 'exit_type']
        
        return merged[cols].iloc[-1:] if latest_only else merged[cols]