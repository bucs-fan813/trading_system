import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager
from src.database.config import DatabaseConfig  # Assuming DatabaseConfig is defined here.

class BollingerBandsKCROStrategy(BaseStrategy):
    """
    Bollinger Bands + Keltner Channel + RSI Strategy with Risk Management.

    This strategy computes the Bollinger Bands (using a 20-period simple moving average and standard deviation),
    the Keltner Channels (using a 20-period EMA and a 10-period ATR), and Wilder's RSI (default period of 14).
    A trading signal is generated if the Bollinger Bands fully enclose the Keltner Channels and the RSI indicates an
    oversold condition (< 30) for a buy signal or an overbought condition (> 70) for a sell signal.
    The raw signal is then processed by a RiskManager component that applies stop-loss, take profit, slippage, and
    transaction cost adjustments while computing the realized returns and cumulative return over the backtest period.

    Parameters (passed via params dictionary):
        bb_lookback (int): Lookback period for Bollinger Bands (default: 20).
        kc_lookback (int): Lookback period for Keltner Channel EMA (default: 20).
        atr_lookback (int): Lookback period to compute ATR for Keltner Channel (default: 10).
        rsi_lookback (int): Lookback period for RSI (default: 14).
        stop_loss_pct (float): Stop loss percentage (default: 0.05).
        take_profit_pct (float): Take profit percentage (default: 0.10).
        slippage_pct (float): Slippage percentage (default: 0.001).
        transaction_cost_pct (float): Transaction cost percentage (default: 0.001).
        long_only (bool): If True, only long positions are allowed (default: True).
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the BollingerBandsKCROStrategy with database configuration and optional parameters.

        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (Dict, optional): Strategy-specific parameters.
        """
        super().__init__(db_config, params)
        self._set_default_params()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _set_default_params(self):
        """
        Set the default parameters for the strategy in case they are not provided.

        Default parameters:
            bb_lookback: 20,
            kc_lookback: 20,
            atr_lookback: 10,
            rsi_lookback: 14,
            stop_loss_pct: 0.05,
            take_profit_pct: 0.10,
            slippage_pct: 0.001,
            transaction_cost_pct: 0.001
            long_only: True
        """
        defaults = {
            'bb_lookback': 20,
            'kc_lookback': 20,
            'atr_lookback': 10,
            'rsi_lookback': 14,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001,
            'long_only': True
        }
        for k, v in defaults.items():
            self.params.setdefault(k, v)

    def generate_signals(self, ticker: str, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate full backtest signals (or the latest signal only) for a given ticker.

        The method retrieves historical price data for the specified date range, computes all technical
        indicators (Bollinger Bands, Keltner Channels, and RSI), and generates a raw trading signal
        based on the condition that the Bollinger Bands enclose the Keltner Channels and the RSI is extreme.
        The raw signal is then processed by the RiskManager to incorporate stop-loss/take-profit, slippage,
        and transaction costs. The result is a complete DataFrame over the entire backtest period (or a single
        latest row if latest_only is set) containing prices, signals, risk-managed positions, and performance metrics.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str, optional): Start date (YYYY-MM-DD) for backtest.
            end_date (str, optional): End date (YYYY-MM-DD) for backtest.
            initial_position (int): Initial position (0 for neutral, 1 for long, -1 for short).
            latest_only (bool): If True, return only the latest available signal (for forecasting).

        Returns:
            pd.DataFrame: A DataFrame containing:
                - Price data (open, high, low, close, etc.).
                - Calculated technical indicators (Bollinger Bands, Keltner Channels, RSI).
                - Generated raw signal.
                - Risk-managed position, trade return, cumulative return, and exit type.
        """
        # Retrieve historical price data.
        max_lookback = max(self.params['bb_lookback'], 
                           self.params['kc_lookback'],
                           self.params['rsi_lookback'])
        data = self.get_historical_prices(
            ticker,
            from_date=start_date,
            to_date=end_date,
            lookback=max_lookback * 2  # Retrieve extra data to ensure reliable indicator calculation.
        )
        if not self._validate_data(data, min_records=max_lookback + 1):
            raise DataRetrievalError(f"Insufficient historical data for ticker: {ticker}")

        # Calculate technical indicators.
        data = self._calculate_indicators(data)

        # Generate raw signals based on indicator conditions.
        data = self._generate_raw_signals(data)

        # Apply risk management rules to the raw signals.
        risk_manager = RiskManager(
            stop_loss_pct=self.params['stop_loss_pct'],
            take_profit_pct=self.params['take_profit_pct'],
            slippage_pct=self.params['slippage_pct'],
            transaction_cost_pct=self.params['transaction_cost_pct']
        )
        risk_data = risk_manager.apply(data[['signal', 'high', 'low', 'close']], initial_position)
        # Join risk-managed results with the original data.
        result = data.join(risk_data, rsuffix='_risk')

        return result.iloc[[-1]] if latest_only else result

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators (Bollinger Bands, Keltner Channels, and RSI).

        Args:
            data (pd.DataFrame): DataFrame with price data. Must contain 'close', 'high', and 'low'.

        Returns:
            pd.DataFrame: DataFrame with new columns:
                - 'upper_bb', 'middle_bb', 'lower_bb' for Bollinger Bands.
                - 'kc_middle', 'kc_upper', 'kc_lower' for Keltner Channels.
                - 'rsi' for the Relative Strength Index.
            Rows with insufficient data (i.e. NaNs) are dropped.
        """
        # Bollinger Bands calculation.
        data['upper_bb'], data['middle_bb'], data['lower_bb'] = \
            self._calculate_bollinger_bands(data['close'], self.params['bb_lookback'])

        # Keltner Channels calculation.
        data['kc_middle'], data['kc_upper'], data['kc_lower'] = \
            self._calculate_keltner_channels(
                data['high'], data['low'], data['close'],
                self.params['kc_lookback'], self.params['atr_lookback']
            )

        # RSI calculation.
        data['rsi'] = self._calculate_rsi(data['close'], self.params['rsi_lookback'])

        return data.dropna()

    def _generate_raw_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate raw trading signals based on technical indicator conditions.

        The condition requires that the Bollinger Bands fully enclose the Keltner Channels.
        A buy (long) signal is generated if, in addition, the RSI is below 30 and a sell (short)
        signal if the RSI is above 70. Otherwise, no signal (0) is produced.

        Args:
            data (pd.DataFrame): DataFrame with calculated indicators.

        Returns:
            pd.DataFrame: Input DataFrame with an added 'signal' column.
        """
        # Condition: Bollinger Bands fully enclose the Keltner Channels.
        bb_kc_condition = (data['lower_bb'] < data['kc_lower']) & (data['upper_bb'] > data['kc_upper'])

        buy_signal = bb_kc_condition & (data['rsi'] < 30)
        sell_signal = bb_kc_condition & (data['rsi'] > 70)

        data['signal'] = np.select(
            [buy_signal, sell_signal],
            [1, -1],
            default=0
        )

        if self.params['long_only']:
            # Override sell signals if long_only is True.
            data.loc[sell_signal, 'signal'] = 0

        return data

    def _calculate_bollinger_bands(self, close: pd.Series, lookback: int) -> tuple:
        """
        Calculate Bollinger Bands based on a rolling simple moving average and standard deviation.

        Args:
            close (pd.Series): Series of close prices.
            lookback (int): Lookback period for calculation.

        Returns:
            tuple: A tuple of (upper_band, middle_band, lower_band).
        """
        sma = close.rolling(lookback).mean()
        std = close.rolling(lookback).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return upper, sma, lower

    def _calculate_keltner_channels(self, high: pd.Series, low: pd.Series, 
                                    close: pd.Series, kc_lookback: int, 
                                    atr_lookback: int) -> tuple:
        """
        Calculate Keltner Channels using an EMA of the close and an ATR-based offset.

        Args:
            high (pd.Series): Series of high prices.
            low (pd.Series): Series of low prices.
            close (pd.Series): Series of close prices.
            kc_lookback (int): Lookback period for the EMA (middle channel).
            atr_lookback (int): Lookback period for computing the ATR.

        Returns:
            tuple: A tuple of (kc_middle, kc_upper, kc_lower).
        """
        # True Range computation.
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=atr_lookback, adjust=False).mean()
        kc_middle = close.ewm(span=kc_lookback, adjust=False).mean()
        kc_upper = kc_middle + 2 * atr
        kc_lower = kc_middle - 2 * atr
        return kc_middle, kc_upper, kc_lower

    def _calculate_rsi(self, close: pd.Series, lookback: int) -> pd.Series:
        """
        Calculate Wilder's Relative Strength Index (RSI).

        Args:
            close (pd.Series): Series of close prices.
            lookback (int): Lookback period for RSI calculation.

        Returns:
            pd.Series: RSI values.
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1/lookback, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/lookback, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi