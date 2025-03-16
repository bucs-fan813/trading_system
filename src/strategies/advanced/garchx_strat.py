# trading_system/src/strategies/advanced/garchx_strat.py

"""
GARCH‑X Strategy Implementation

This module implements a GARCH‑X strategy that:
   - Preprocesses OHLCV data by computing log returns, standardizing volume, calculating price ranges,
     generating lag features, and deriving volatility proxies (using both Parkinson and Yang–Zhang methods).
   - Uses a rolling forecast origin: for each day (when sufficient historical data is available), the preceding
     forecast lookback window is used to recalibrate an EGARCH‑X model.
   - Dynamically selects the best model configuration by trying several candidate orders (using AIC) on an EGARCH model
     with an ARX component in the mean, Student’s t‑errors, and PCA on exogenous regressors to reduce collinearity.
   - Avoids look‐ahead bias by applying winsorization strictly within the training sample for each forecast.
   - Generates risk‑adjusted trading signals by normalizing the cumulative forecast return by forecast volatility,
     and then scaling it by the latest standardized volume.
   - Provides a fallback to an EWMA forecast if the EGARCH‑X model cannot be fitted.
   - Optionally computes position sizing based on forecast volatility and allocated capital.
   - Supports multi-ticker processing using joblib parallelization.

The resulting DataFrame includes:
   open, high, low, close, forecast_return, signal, signal_strength, position_size.

Risk management (including stop loss, take profit, slippage, and transaction cost adjustments) is applied
using the RiskManager.
"""

import os
import numpy as np
import pandas as pd
import logging
import pickle  # added for saving/loading trained model files
from typing import Dict, Optional, Union, List, Tuple

from arch import arch_model
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.database.config import DatabaseConfig
from src.strategies.risk_management import RiskManager

class GarchXStrategy(BaseStrategy):
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the GARCH‑X Strategy.

        Parameters:
            db_config (DatabaseConfig): The configuration for database connectivity.
            params (Optional[Dict]): Dictionary of parameters. Expected keys include:
                - 'forecast_horizon': int, forecast days ahead (default: 30)
                - 'forecast_lookback': int, lookback window for training (default: 200)
                - 'min_volume_strength': float, minimum standardized volume (vol_z) threshold to trigger a signal (default: 1.0)
                - 'long_only': bool, whether to allow only long positions (default: True)
                - 'capital': float, capital allocated for position sizing (default: 1.0)
                - Risk parameters:
                    * 'stop_loss_pct' (default: 0.05)
                    * 'take_profit_pct' (default: 0.10)
                    * 'slippage_pct' (default: 0.001)
                    * 'transaction_cost_pct' (default: 0.001)
                - 'mode': str, either "train" to fit & save the model or "forecast" to use an existing saved model (default: "train")
        """
        default_params = {
            'forecast_horizon': 30,
            'forecast_lookback': 200,
            'min_volume_strength': 1.0,
            'pca_components': 5,
            'winsorize_quantiles': (0.01, 0.99),
            'vol_window': 20,
            'long_only': True,
            'capital': 1.0,
            # Risk parameters
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'trail_stop_pct': 0,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001,
            # Mode parameter: "train" will fit a model and store it; "forecast" will load an existing model.
            'mode': "train",
        }
        default_params.update(params or {})
        super().__init__(db_config, default_params)
        self.forecast_horizon = int(self.params.get('forecast_horizon'))
        self.forecast_lookback = int(self.params.get('forecast_lookback'))
        self.min_volume_strength = float(self.params.get('min_volume_strength'))
        self.pca_components = int(self.params.get('pca_components'))
        self.winsorize_quantiles = self.params.get('winsorize_quantiles')
        self.vol_window = int(self.params.get('vol_window'))
        self.long_only = bool(self.params.get('long_only'))
        self.capital = float(self.params.get('capital'))
        
        # Mode: "train" or "forecast"
        self.mode = self.params.get("mode", "train")
        
        risk_params = {
            'stop_loss_pct': self.params.get('stop_loss_pct'),
            'take_profit_pct': self.params.get('take_profit_pct'),
            'trail_stop_pct': self.params.get('trail_stop_pct'),
            'slippage_pct': self.params.get('slippage_pct'),
            'transaction_cost_pct': self.params.get('transaction_cost_pct'),
        }
        self.risk_manager = RiskManager(**risk_params)
        self.logger = logging.getLogger(self.__class__.__name__)
        # Create the folder for storing model configuration files if it does not exist.
        self.model_config_folder = os.path.join("trading_system", "models", "garchx")
        os.makedirs(self.model_config_folder, exist_ok=True)

    def generate_signals(
        self,
        ticker: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False,
    ) -> pd.DataFrame:
        """
        Generate trading signals by applying the GARCH‑X model on historical price data.

        The method:
          - Retrieves and preprocesses historical OHLCV data.
          - For single or multiple tickers, calculates model forecasts using either a rolling window approach or
            a single forecast based on the latest available data.
          - Applies risk management adjustments and computes additional performance metrics.

        Parameters:
            ticker (str or List[str]): Ticker symbol(s) for which signals are generated.
            start_date (Optional[str]): Start date for backtesting (if applicable).
            end_date (Optional[str]): End date for backtesting (if applicable).
            initial_position (int): Initial position count for risk management adjustments.
            latest_only (bool): If True, compute forecast only for the most recent data point.

        Returns:
            pd.DataFrame: DataFrame containing columns:
                - open, high, low, close
                - forecast_return, signal, signal_strength, position_size
              along with additional performance metrics.
        """
        try:
            # Define a lookback buffer to ensure sufficient historical data is available for model calibration.
            lookback_buffer = max(self.forecast_lookback, self.forecast_horizon) + 10

            if start_date and end_date:
                data = self.get_historical_prices(
                    ticker, from_date=start_date, to_date=end_date, lookback=lookback_buffer
                )
            else:
                data = self.get_historical_prices(ticker, lookback=lookback_buffer)
                data = data.sort_index()

            # Handle processing for either single or multiple tickers.
            if isinstance(ticker, list):
                # Utilize joblib parallel processing to compute signals for multiple tickers concurrently.
                signals_list = Parallel(n_jobs=-1)(
                    delayed(self._calculate_signals_single)(group, latest_only)
                    for t, group in data.groupby(level=0)
                )
                signals_list = [sig for sig in signals_list if not sig.empty]
                if not signals_list:
                    return pd.DataFrame()
                signals = pd.concat(signals_list)
                if latest_only:
                    signals = signals.groupby('ticker').tail(1)
            else:
                if not self._validate_data(data, min_records=self.forecast_lookback):
                    self.logger.warning(
                        f"Insufficient data for {ticker}: requires at least {self.forecast_lookback} records."
                    )
                    return pd.DataFrame()
                signals = self._calculate_signals_single(data, latest_only)
            # Apply risk management adjustments including position sizing, calculated returns, and exit actions.
            signals = self.risk_manager.apply(signals, initial_position)
            # Calculate daily returns and overall strategy performance metrics.
            signals['daily_return'] = signals['close'].pct_change().fillna(0)
            signals['strategy_return'] = signals['daily_return'] * signals['position'].shift(1).fillna(0)
            signals.rename(
                columns={'return': 'rm_strategy_return',
                         'cumulative_return': 'rm_cumulative_return',
                         'exit_type': 'rm_action'},
                inplace=True,
            )
            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals for {ticker}: {str(e)}")
            return pd.DataFrame()

    def _calculate_signals_single(self, data: pd.DataFrame, latest_only: bool) -> pd.DataFrame:
        """
        Process historical data for a single ticker to calculate forecasted returns and trading signals.

        This method:
          1. Computes log returns.
          2. Derives volume-related features (including log transform, rolling mean/STD, z-score, and ratio).
          3. Calculates price range features based on high, low, open, and close prices.
          4. Computes volatility proxies using both the Parkinson and a simplified Yang–Zhang method.
          5. Generates lag features for log return, volume z-score, and high-low range for lags 1 to 5.
          6. Applies a rolling forecast (or single-point forecast) using an EGARCH‑X model.

        Parameters:
            data (pd.DataFrame): DataFrame containing historical OHLCV data.
            latest_only (bool): If True, compute forecast only for the most recent data point.

        Returns:
            pd.DataFrame: DataFrame with the following columns:
                - open, high, low, close
                - forecast_return, signal, signal_strength, position_size
              If processing multiple tickers, a ticker identifier may also be included.
        """
        df = data.copy()
        # If the DataFrame index is a MultiIndex, reset and use only the date level.
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
        df.sort_index(inplace=True)

        # 1. Compute the logarithmic returns from close prices.
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        
        # 2. Compute volume-related features: log(volume), moving window mean and standard deviation, standardized volume,
        #    and ratio relative to a 20-day moving average.
        df["vol_log"] = np.log(df["volume"])
        df["vol_mean"] = df["vol_log"].rolling(window=self.vol_window, min_periods=self.vol_window).mean()
        df["vol_std"] = df["vol_log"].rolling(window=self.vol_window, min_periods=self.vol_window).std()
        df["vol_z"] = (df["vol_log"] - df["vol_mean"]) / df["vol_std"]
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(window=self.vol_window, min_periods=self.vol_window).mean()
        
        # 3. Calculate price range features based on high-low and open-close differences.
        df["hl_range"] = (df["high"] - df["low"]) / df["close"].shift(1)
        df["oc_range"] = (df["open"] - df["close"]).abs() / df["close"].shift(1)

        # 4. Compute volatility proxies.
        # Parkinson volatility: estimate volatility using high and low prices.
        df["log_hl"] = np.log(df["high"] / df["low"])
        df["parkinson_vol"] = np.sqrt(
            df["log_hl"].rolling(window=20, min_periods=20).apply(lambda x: np.mean(x**2), raw=True)
            / (4 * np.log(2))
        )
        # Simplified Yang–Zhang volatility proxy using overnight returns and open-to-close returns.
        df["overnight_ret"] = np.log(df["open"] / df["close"].shift(1))
        df["oc_ret"] = np.log(df["close"] / df["open"])
        df["yz_vol"] = np.sqrt(
            df["overnight_ret"].rolling(window=20, min_periods=20).var() +
            df["oc_ret"].rolling(window=20, min_periods=20).var() +
            0.5 * df["hl_range"].rolling(window=20, min_periods=20).var()
        )

        # 5. Generate lag features for log_return, standardized volume (vol_z), and high-low range for lags 1 to 5.
        for lag in range(1, 6):
            df[f"lag{lag}_return"] = df["log_return"].shift(lag)
            df[f"lag{lag}_vol"] = df["vol_z"].shift(lag)
            df[f"lag{lag}_hl"] = df["hl_range"].shift(lag)

        # Remove rows with missing values resulting from rolling window calculations.
        df.dropna(inplace=True)

        # Note: Winsorization to mitigate outliers is applied later in the _fit_forecast method to avoid look-ahead bias.

        # Initialize columns for storing forecast outputs.
        df["forecast_return"] = np.nan
        df["signal"] = np.nan
        df["signal_strength"] = np.nan
        df["position_size"] = np.nan

        horizon = self.forecast_horizon

        if latest_only:
            # For latest-only forecasting, use the most recent training window to compute forecasted values and assign them accordingly.
            train = df.iloc[-self.forecast_lookback:]
            fc_ret, sig, sig_str, pos_size = self._fit_forecast(train, horizon)
            df.loc[df.index[-1], "forecast_return"] = fc_ret
            df.loc[df.index[-1], "signal"] = sig
            df.loc[df.index[-1], "signal_strength"] = sig_str
            df.loc[df.index[-1], "position_size"] = pos_size
            df.fillna(method="ffill", inplace=True)
        else:
            # Perform a rolling forecast using overlapping training windows for each forecast horizon.
            for i in range(len(df) - self.forecast_lookback - horizon + 1):
                train = df.iloc[i : i + self.forecast_lookback]
                forecast_dates = df.index[i + self.forecast_lookback : i + self.forecast_lookback + horizon]
                fc_ret, sig, sig_str, pos_size = self._fit_forecast(train, horizon)
                df.loc[forecast_dates, "forecast_return"] = fc_ret
                df.loc[forecast_dates, "signal"] = sig
                df.loc[forecast_dates, "signal_strength"] = sig_str
                df.loc[forecast_dates, "position_size"] = pos_size
            df.fillna(method="ffill", inplace=True)

        # Retain only the essential columns required for risk management processing.
        cols = ["open", "high", "low", "close", "forecast_return", "signal", "signal_strength", "position_size"]
        result = df[cols].copy()
        # Include a ticker identifier if processing data for multiple tickers.
        if 'ticker' not in result.columns and 'ticker' in data.columns:
            result['ticker'] = data['ticker']
        return result

    def _fit_forecast(self, train: pd.DataFrame, horizon: int) -> Tuple[float, int, float, float]:
        """
        Fit an EGARCH‑X model on a training subset of data and forecast the cumulative log return over a specified horizon.

        The method:
          - Defines lag features as exogenous regressors.
          - Applies winsorization to both the exogenous features and target variable to reduce the impact of outliers.
          - Uses PCA on the exogenous regressors to mitigate multicollinearity.
          - Tries multiple candidate EGARCH configurations and selects the one with the lowest AIC.
          - In case all candidate models fail to fit, falls back to an EWMA forecast.

        Additionally:
          - In "train" mode, the fitted model (along with its PCA object and configuration) is saved to disk.
          - In "forecast" mode, the saved model file is loaded and used directly to generate the forecast.

        Parameters:
            train (pd.DataFrame): Training data for model fitting.
            horizon (int): Forecast horizon (number of days ahead).

        Returns:
            tuple:
                - cumulative_forecast_return (float): Forecasted cumulative return, computed as exp(sum(forecasted log returns)) - 1.
                - signal (int): Trading signal indicator (1 for buy, 0 or -1 for sell/hold).
                - signal_strength (float): Risk-adjusted signal strength value.
                - position_size (float): Computed position sizing value based on forecast volatility and allocated capital.
        """
        # If operating in forecast mode, load the pre-trained model from file.
        if self.mode == "forecast":
            model_pickle_file = os.path.join(self.model_config_folder, "trained_model.pkl")
            if not os.path.exists(model_pickle_file):
                self.logger.error("Pre-trained model file not found. Run in train mode first.")
                raise Exception("Pre-trained model file not found. Run in train mode first.")
            with open(model_pickle_file, "rb") as f:
                model_data = pickle.load(f)
            best_result = model_data["model"]
            pca = model_data["pca"]
            n_components = model_data["n_components"]

            # Prepare exogenous regressors using the same winsorization as in training.
            exog_cols = (
                [f"lag{l}_return" for l in range(1, 6)] +
                [f"lag{l}_vol" for l in range(1, 6)] +
                [f"lag{l}_hl" for l in range(1, 6)]
            )
            X_train = train[exog_cols].copy()
            for col in X_train.columns:
                lower = X_train[col].quantile(self.winsorize_quantiles[0])
                upper = X_train[col].quantile(self.winsorize_quantiles[1])
                X_train[col] = X_train[col].clip(lower, upper)
            last_exog = X_train.iloc[-1].values.reshape(1, -1)
            last_exog_pca = pca.transform(last_exog)
            X_forecast_pca = pd.DataFrame(
                np.tile(last_exog_pca, (horizon, 1)),
                columns=[f'pca_{i}' for i in range(n_components)]
            )
            forecast = best_result.forecast(horizon=horizon, x=X_forecast_pca, reindex=False)
            fc_means = forecast.mean.iloc[-1].values
            cumulative_ret = np.exp(np.sum(fc_means)) - 1
            forecast_vol = np.sqrt(forecast.variance.iloc[-1].mean())
        else:
            # Train Mode: Fit candidate EGARCH models.
            exog_cols = (
                [f"lag{l}_return" for l in range(1, 6)] +
                [f"lag{l}_vol" for l in range(1, 6)] +
                [f"lag{l}_hl" for l in range(1, 6)]
            )
            X_train = train[exog_cols].copy()
            y_train = train["log_return"].copy()

            # Apply winsorization to each column using training sample quantiles (1% and 99%) to limit the influence of outliers.
            for col in X_train.columns:
                lower = X_train[col].quantile(self.winsorize_quantiles[0])
                upper = X_train[col].quantile(self.winsorize_quantiles[1])
                X_train[col] = X_train[col].clip(lower, upper)
            lower_y = y_train.quantile(self.winsorize_quantiles[0])
            upper_y = y_train.quantile(self.winsorize_quantiles[1])
            y_train = y_train.clip(lower_y, upper_y)

            # Reduce feature dimensionality and mitigate multicollinearity via PCA.
            n_components = min(self.pca_components, X_train.shape[1])
            pca = PCA(n_components=n_components)
            X_train_pca = pd.DataFrame(pca.fit_transform(X_train),
                                       index=X_train.index,
                                       columns=[f'pca_{i}' for i in range(n_components)])

            # Define candidate orders for the EGARCH model for model selection based on AIC.
            candidate_orders = [(1, 1), (1, 2), (2, 1)]
            best_aic = np.inf
            best_result = None
            best_order = None

            for (p, q) in candidate_orders:
                try:
                    model = arch_model(
                        y_train,
                        x=X_train_pca,
                        mean="ARX",
                        lags=0,  # Lags are provided via the exogenous PCA features.
                        vol="EGARCH",
                        p=p,
                        q=q,
                        dist="t"
                    )
                    res = model.fit(disp="off")
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_result = res
                        best_order = (p, q)
                except Exception as e:
                    self.logger.warning(f"EGARCH({p},{q}) model failed: {e}")
                    continue

            # Log the selected EGARCH model configuration (order and AIC) for tracking.
            model_filename_txt = os.path.join(self.model_config_folder, "model_config.txt")
            if best_order is not None:
                with open(model_filename_txt, "a") as f:
                    f.write(f"Ticker: [unknown] | Best Order: EGARCH{best_order} | AIC: {best_aic}\n")
            else:
                self.logger.error("All candidate EGARCH models failed. Falling back to EWMA.")

            try:
                if best_result is not None:
                    # Transform the most recent exogenous features using the same PCA transformation
                    # and replicate them to align with the forecast horizon.
                    last_exog = X_train.iloc[-1].values.reshape(1, -1)
                    last_exog_pca = pca.transform(last_exog)
                    X_forecast_pca = pd.DataFrame(
                        np.tile(last_exog_pca, (horizon, 1)),
                        columns=[f'pca_{i}' for i in range(n_components)]
                    )
                    forecast = best_result.forecast(horizon=horizon, x=X_forecast_pca, reindex=False)
                    # Calculate the cumulative forecasted return by exponentiating the sum of forecasted log returns and subtracting 1.
                    fc_means = forecast.mean.iloc[-1].values
                    cumulative_ret = np.exp(np.sum(fc_means)) - 1
                    # Obtain an overall forecasted volatility as the square root of the mean forecasted variance.
                    forecast_vol = np.sqrt(forecast.variance.iloc[-1].mean())
                    avg_fc_ret = cumulative_ret  # Use cumulative return as the representative forecast signal.
                else:
                    raise Exception("No suitable EGARCH model fitted.")

            except Exception as e:
                self.logger.error(f"Forecasting error with EGARCH model: {e}. Using EWMA fallback.")
                avg_fc_ret = train['log_return'].ewm(span=30).mean().iloc[-1]
                forecast_vol = train['parkinson_vol'].iloc[-1]  # Use Parkinson volatility as a fallback proxy.
                cumulative_ret = avg_fc_ret  # Fallback cumulative return based on EWMA.

            # Calculate a risk-adjusted return by normalizing the cumulative return with forecast volatility.
            if forecast_vol > 0:
                risk_adj_return = cumulative_ret / forecast_vol
            else:
                risk_adj_return = 0.0

            vol_strength = train["vol_z"].iloc[-1]
            signal_strength = risk_adj_return * vol_strength

            # Generate the trading signal based on risk-adjusted return and volume strength.
            if risk_adj_return > 0 and vol_strength > self.min_volume_strength:
                sig = 1
            elif risk_adj_return < 0 and vol_strength > self.min_volume_strength:
                sig = 0 if self.long_only else -1
            else:
                sig = 0

            # Compute the position size based on the risk-adjusted return scaled by the allocated capital.
            position_size = risk_adj_return * self.capital

            # In train mode, save the fitted model and PCA transformation for later forecasting.
            model_pickle_file = os.path.join(self.model_config_folder, "trained_model.pkl")
            if best_result is not None:
                model_data = {
                    "model": best_result,
                    "pca": pca,
                    "n_components": n_components,
                }
                with open(model_pickle_file, "wb") as f:
                    pickle.dump(model_data, f)

        # Calculate a risk-adjusted return by normalizing the cumulative return with forecast volatility.
        if forecast_vol > 0:
            risk_adj_return = cumulative_ret / forecast_vol
        else:
            risk_adj_return = 0.0

        vol_strength = train["vol_z"].iloc[-1]
        signal_strength = risk_adj_return * vol_strength

        # Generate the trading signal based on risk-adjusted return and volume strength.
        if risk_adj_return > 0 and vol_strength > self.min_volume_strength:
            sig = 1
        elif risk_adj_return < 0 and vol_strength > self.min_volume_strength:
            sig = 0 if self.long_only else -1
        else:
            sig = 0

        # Compute the position size based on the risk-adjusted return scaled by the allocated capital.
        position_size = risk_adj_return * self.capital

        return cumulative_ret, sig, signal_strength, position_size