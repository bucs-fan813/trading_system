"""
GARCH‑X Strategy Implementation

This module implements a GARCH‑X based strategy that preprocesses OHLCV data to
generate features, fits a GARCH‑X model to forecast returns 30 days ahead, and then
generates trading signals. The signal is set to +1 if the forecasted return is positive
(and volume is strong), –1 if the forecast is negative (with strong volume), and 0 otherwise.
Risk management (stop loss, take profit, slippage and transaction cost adjustments) is
applied via the RiskManager.

The preprocessing steps include:
    1. Log returns transformation.
    2. Volume preprocessing: log transformation, standardization over a 20‑day window,
       and volume ratio calculation.
    3. Price range variables: normalized high‑low and open‑close ranges.
    4. Lagged features for returns, standardized volume, and volatility proxies (normalized range).
    5. Winsorization of extreme values at the 1% and 99% levels.
    6. Partitioning of the backtest period into blocks and for each block a GARCH‑X model
       is fitted (using exogenous lag features) to generate a 30‑day forecast.
    7. Signals are generated only if the forecast (averaged over 30 days) is directional
       AND the “volume strength” (last vol_z value) exceeds a minimum threshold.
       
The strategy is built to process a list of tickers and returns a full DataFrame including
raw prices, forecast information, raw signal and risk‐managed positions & returns.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Union, List

from arch import arch_model

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.database.config import DatabaseConfig
from src.strategies.risk_management import RiskManager


class GarchXStrategy(BaseStrategy):
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the GARCH‑X Strategy.

        Expected params keys include:
            - 'forecast_horizon': int, forecast days ahead (default: 30)
            - 'forecast_lookback': int, lookback window (default: 200)
            - 'min_volume_strength': float, minimum vol_z threshold for signal (default: 1.0)
            - 'long_only': bool, whether to allow only long positions (default: True)
            - risk management parameters:
                * 'stop_loss_pct' (default: 0.05)
                * 'take_profit_pct' (default: 0.10)
                * 'slippage_pct' (default: 0.001)
                * 'transaction_cost_pct' (default: 0.001)
        """
        default_params = {
            'forecast_horizon': 30,
            'forecast_lookback': 200,
            'min_volume_strength': 1.0,
            'long_only': True,
            # Risk parameters
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001,
        }
        # Merge default params with user-supplied parameters.
        default_params.update(params or {})
        super().__init__(db_config, default_params)
        self.forecast_horizon = int(self.params.get('forecast_horizon'))
        self.forecast_lookback = int(self.params.get('forecast_lookback'))
        self.min_volume_strength = float(self.params.get('min_volume_strength'))

        # Initialize RiskManager with provided risk parameters.
        risk_params = {
            'stop_loss_pct': self.params.get('stop_loss_pct'),
            'take_profit_pct': self.params.get('take_profit_pct'),
            'slippage_pct': self.params.get('slippage_pct'),
            'transaction_cost_pct': self.params.get('transaction_cost_pct'),
        }
        self.risk_manager = RiskManager(**risk_params)
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_signals(
        self,
        ticker: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False,
    ) -> pd.DataFrame:
        """
        Generate trading signals using the GARCH‑X model.

        When start_date and end_date are provided (backtesting mode), the historical data
        are partitioned into blocks of length forecast_horizon, and for each block a rolling
        forecast is computed.

        When latest_only is True, only the latest forecast is computed based on the most recent
        forecast_lookback days.

        After the raw signals are computed, risk management is applied to compute positions,
        returns, and exit events.

        Returns:
            pd.DataFrame: DataFrame containing open, high, low, close prices,
                          forecast_return, signal, signal_strength, and performance metrics.
        """
        try:
            # Determine a lookback buffer based on the forecast_lookback period.
            lookback_buffer = max(self.forecast_lookback, self.forecast_horizon) + 10

            # Retrieve historical price data via the base class.
            if start_date and end_date:
                data = self.get_historical_prices(
                    ticker, from_date=start_date, to_date=end_date, lookback=lookback_buffer
                )
            else:
                data = self.get_historical_prices(ticker, lookback=lookback_buffer)
                data = data.sort_index()

            # Process single ticker or multiple tickers separately.
            if isinstance(ticker, list):
                signals_list = []
                for t, group in data.groupby(level=0):
                    if not self._validate_data(group, min_records=self.forecast_lookback):
                        self.logger.warning(
                            f"Insufficient data for {t}: requires at least {self.forecast_lookback} records."
                        )
                        continue
                    sig = self._calculate_signals_single(group, latest_only)
                    # Apply risk management.
                    sig = self.risk_manager.apply(sig, initial_position)
                    # Compute performance metrics.
                    sig['daily_return'] = sig['close'].pct_change().fillna(0)
                    sig['strategy_return'] = sig['daily_return'] * sig['position'].shift(1).fillna(0)
                    # Rename risk-managed columns.
                    sig.rename(
                        columns={'return': 'rm_strategy_return',
                                 'cumulative_return': 'rm_cumulative_return',
                                 'exit_type': 'rm_action'},
                        inplace=True,
                    )
                    # Insert ticker identifier.
                    sig['ticker'] = t
                    signals_list.append(sig)
                if not signals_list:
                    return pd.DataFrame()
                signals = pd.concat(signals_list)
                # If latest_only is specified, return only the last row per ticker.
                if latest_only:
                    signals = signals.groupby('ticker').tail(1)
            else:
                if not self._validate_data(data, min_records=self.forecast_lookback):
                    self.logger.warning(
                        f"Insufficient data for {ticker}: requires at least {self.forecast_lookback} records."
                    )
                    return pd.DataFrame()
                sig = self._calculate_signals_single(data, latest_only)
                sig = self.risk_manager.apply(sig, initial_position)
                sig['daily_return'] = sig['close'].pct_change().fillna(0)
                sig['strategy_return'] = sig['daily_return'] * sig['position'].shift(1).fillna(0)
                sig.rename(
                    columns={'return': 'rm_strategy_return',
                             'cumulative_return': 'rm_cumulative_return',
                             'exit_type': 'rm_action'},
                    inplace=True,
                )
                if latest_only:
                    sig = sig.iloc[[-1]].copy()
                signals = sig

            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals for {ticker}: {str(e)}")
            return pd.DataFrame()

    def _calculate_signals_single(self, data: pd.DataFrame, latest_only: bool) -> pd.DataFrame:
        """
        Calculate the raw forecast and trading signals for a single ticker's historical data.

        The method performs the following:

          1. Sort data chronologically.
          2. Compute log returns:
                r_t = ln(close_t / close_{t-1})
          3. Volume preprocessing:
                - Compute vol_log = ln(volume)
                - Standardize vol_log over a rolling 20‑day window to obtain vol_z.
                - Compute volume ratio = volume / rolling_mean(volume, window=20).
          4. Price range features:
                - Normalized high–low range = (high – low) / close_{t-1}
                - Normalized open–close range = |open – close| / close_{t-1}
          5. Realized volatility using the Parkinson estimator:
                parkinson_vol = sqrt((rolling_mean((ln(high/low))^2, window=20)) / (4 * ln(2)))
          6. Create lag features (lags 1 through 5) for log_return, vol_z and normalized high–low range.
          7. Winsorize extreme values (clip at the 1st and 99th percentiles).
          8. For forecasting:
                a. In backtesting mode (latest_only=False): partition the data into blocks
                   of length equal to forecast_horizon. For each block, use the most recent
                   forecast_lookback days (prior to the block start) as the training window,
                   fit a GARCH‑X model (with exogenous lag features), and forecast the next
                   forecast_horizon days. The average forecast return is used to decide:
                        if forecast > 0 and vol_z (from training) > min_volume_strength then signal = +1,
                        if forecast < 0 and vol_z > min_volume_strength then signal = –1,
                        else signal = 0.
                b. In "latest_only" mode, the forecast is computed only using the last forecast_lookback days.
          9. The computed forecast_return, signal and signal_strength are then assigned
             (constant over each block).

        Returns:
            pd.DataFrame: DataFrame with the original price columns and new columns:
                          'forecast_return', 'signal', and 'signal_strength'.
        """
        df = data.copy()
        # Ensure the index is datetime. In multi-ticker data, the date is in level 1.
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
        df.sort_index(inplace=True)

        # 1. Compute log returns: r_t = ln(close_t / close_{t-1})
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # 2. Volume preprocessing:
        #    a. Log-transform volume.
        df["vol_log"] = np.log(df["volume"])
        #    b. Standardize volume within a rolling 20-day window.
        df["vol_mean"] = df["vol_log"].rolling(window=20, min_periods=20).mean()
        df["vol_std"] = df["vol_log"].rolling(window=20, min_periods=20).std()
        df["vol_z"] = (df["vol_log"] - df["vol_mean"]) / df["vol_std"]
        #    c. Create volume ratio.
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(window=20, min_periods=20).mean()

        # 3. Price range features:
        df["hl_range"] = (df["high"] - df["low"]) / df["close"].shift(1)
        df["oc_range"] = (df["open"] - df["close"]).abs() / df["close"].shift(1)

        # 4. Realized volatility using Parkinson estimator.
        df["log_hl"] = np.log(df["high"] / df["low"])
        df["parkinson_vol"] = np.sqrt(
            df["log_hl"].rolling(window=20, min_periods=20).apply(
                lambda x: np.mean(x**2), raw=True
            ) / (4 * np.log(2))
        )

        # 5. Create lag features (lags 1 to 5) for log_return, vol_z and hl_range.
        for lag in range(1, 6):
            df[f"lag{lag}_return"] = df["log_return"].shift(lag)
            df[f"lag{lag}_vol"] = df["vol_z"].shift(lag)
            df[f"lag{lag}_hl"] = df["hl_range"].shift(lag)

        # Drop rows with NA values (due to shift and rolling windows).
        df.dropna(inplace=True)

        # 6. Winsorize the selected features at the 1st and 99th percentiles.
        winsorize_cols = ["log_return", "vol_z", "hl_range", "oc_range", "parkinson_vol"]
        for col in winsorize_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)

        # Initialize forecast result columns.
        df["forecast_return"] = np.nan
        df["signal"] = np.nan
        df["signal_strength"] = np.nan

        # Set forecast horizon.
        horizon = self.forecast_horizon

        # If latest_only is True: compute forecast using the last forecast_lookback days.
        if latest_only:
            train = df.iloc[-self.forecast_lookback :]
            fc_ret, sig, sig_str = self._fit_forecast(train, horizon)
            df.loc[df.index[-1], "forecast_return"] = fc_ret
            df.loc[df.index[-1], "signal"] = sig
            df.loc[df.index[-1], "signal_strength"] = sig_str
            # Forward fill these values to the last row.
            df.fillna(method="ffill", inplace=True)
        else:
            # Backtesting mode: Partition the data into blocks of length horizon.
            unique_dates = df.index.unique()
            # Create blocks where each block starts at indices 0, horizon, 2*horizon, ...
            block_starts = np.arange(0, len(unique_dates), horizon)
            for bs in block_starts:
                block_start_date = unique_dates[bs]
                # Training window is the forecast_lookback days preceding the block start.
                train = df.loc[:block_start_date].iloc[-self.forecast_lookback :]
                fc_ret, sig, sig_str = self._fit_forecast(train, horizon)
                # Determine all dates in the forecast block.
                block_dates = unique_dates[bs : bs + horizon]
                df.loc[block_dates, "forecast_return"] = fc_ret
                df.loc[block_dates, "signal"] = sig
                df.loc[block_dates, "signal_strength"] = sig_str

            # In case any rows remain NA (e.g. at the very beginning), fill forward.
            df.fillna(method="ffill", inplace=True)

        # Finally, select the relevant columns for risk management.
        # The RiskManager requires at least: 'signal', 'high', 'low', 'close'.
        result = df[
            [
                "open",
                "high",
                "low",
                "close",
                "forecast_return",
                "signal",
                "signal_strength",
            ]
        ].copy()

        return result

    def _fit_forecast(self, train: pd.DataFrame, horizon: int):
        """
        Fit the GARCH‑X model on the training data and forecast the mean return for the next horizon days.

        Parameters:
            train (pd.DataFrame): The training window from which to fit the model.
            horizon (int): Forecast horizon (number of days ahead).

        Returns:
            tuple: (avg_forecast_return, signal, signal_strength)
                - avg_forecast_return: The average forecasted return over the horizon.
                - signal: +1 if forecast >0 and volume is strong, –1 if forecast <0 and volume is strong, else 0.
                - signal_strength: A scalar measure of forecast strength, e.g. abs(avg_forecast_return)*vol_z.
        """
        # Define the columns for exogenous regressors (lag features).
        exog_cols = [
            f"lag{l}_return" for l in range(1, 6)
        ] + [f"lag{l}_vol" for l in range(1, 6)] + [f"lag{l}_hl" for l in range(1, 6)]
        X_train = train[exog_cols]
        y_train = train["log_return"]

        # Fit the GARCH-X model using arch. We use a GARCH(1,1) specification with an ARX in the mean.
        try:
            model = arch_model(
                y_train,
                x=X_train,
                mean="ARX",
                lags=0,  # because lags are provided in x
                vol="GARCH",
                p=1,
                q=1,
                dist="normal",
            )
            res = model.fit(disp="off")
            # For forecasting the exogenous variables, we use the last row of training exog and replicate it.
            last_exog = X_train.iloc[-1]
            X_forecast = pd.DataFrame(
                np.tile(last_exog.values, (horizon, 1)),
                columns=exog_cols,
            )
            forecast = res.forecast(horizon=horizon, x=X_forecast, reindex=False)
            # The forecast mean is a DataFrame with horizon columns; take the last row and average the values.
            fc_mean = forecast.mean.iloc[-1]
            avg_fc_ret = fc_mean.mean()
        except Exception as e:
            self.logger.error(f"GARCH‑X model fitting/forecasting error: {e}")
            avg_fc_ret = 0.0

        # Use the last available standardized volume from the training.
        vol_strength = train["vol_z"].iloc[-1]

        # Generate a signal (with simple volume condition):
        if avg_fc_ret > 0 and vol_strength > self.min_volume_strength:
            sig = 1
            sig_str = avg_fc_ret * vol_strength
        elif avg_fc_ret < 0 and vol_strength > self.min_volume_strength:
            if self.params.get('long_only'):
                sig = 0
                sig_str = abs(avg_fc_ret) * vol_strength
            else:
                sig = -1
                sig_str = abs(avg_fc_ret) * vol_strength
        else:
            sig = 0
            sig_str = 0.0

        return avg_fc_ret, sig, sig_str