# trading_system/src/strategies/advanced/prophet_momentum_strat.py

import os
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import timedelta
from prophet import Prophet
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from scipy import stats
from hmmlearn.hmm import GaussianHMM
import concurrent.futures
from typing import  Union, List

from src.strategies.base_strat import BaseStrategy
from src.database.config import DatabaseConfig
from src.strategies.risk_management import RiskManager

class ProphetMomentumStrategy(BaseStrategy):
    def __init__(self, db_config: DatabaseConfig, params: dict = None):
        """
        Initialize the Prophet Momentum Strategy.
        The default parameters include forecast horizons and risk management settings.
        """
        default_params = {
            'horizon_short': 7,
            'horizon_mid': 14,
            'horizon_long': 30,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'trailing_stop_pct': 0,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001,
            # additional flag if only the latest signal is desired:
            'latest_only': False
        }
        if params:
            default_params.update(params)
        super().__init__(db_config, default_params)
        self.risk_manager = RiskManager(
            stop_loss_pct=self.params.get('stop_loss_pct', 0.05),
            take_profit_pct=self.params.get('take_profit_pct', 0.10),
            trailing_stop_pct=self.params.get('trailing_stop_pct', 0),
            slippage_pct=self.params.get('slippage_pct', 0.001),
            transaction_cost_pct=self.params.get('transaction_cost_pct', 0.001)
        )
        # Create directory for caching final models.
        self.model_dir = os.path.join("trading_system", "models", "prophet_momentum")
        os.makedirs(self.model_dir, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _tune_hyperparameters(self, df_cv: pd.DataFrame) -> dict:
        """
        Use forward crossvalidation and Hyperopt to select the best hyperparameters.
        We use the latest 3 years of data (df_cv) and roll forward in 12 folds.
        Each fold uses 2 years of training data and 30 days ahead for testing.
        
        The objective is to maximize directional accuracy.
        Directional accuracy is defined as:
        
            DA = (1/N) * Σ I[ sign(y_pred - y_last_train) == sign(y_test - y_last_train) ]
        
        and we set loss = 1 - DA.
        """
        def objective(params):
            # Ensure fourier_order is an integer
            params['fourier_order'] = int(params['fourier_order'])
            accuracies = []
            num_folds = 0

            # Set up fold dates – start from the minimum date in df_cv.
            current_fold_start = df_cv['ds'].min()
            while num_folds < 12:
                train_end = current_fold_start + pd.DateOffset(years=2)
                test_end = train_end + pd.Timedelta(days=30)
                if test_end > df_cv['ds'].max():
                    break
                train_data = df_cv[(df_cv['ds'] >= current_fold_start) & (df_cv['ds'] <= train_end)]
                test_data = df_cv[(df_cv['ds'] > train_end) & (df_cv['ds'] <= test_end)]
                # Ensure sufficient observations in each fold.
                if len(train_data) < 100 or len(test_data) < 10:
                    current_fold_start += pd.DateOffset(months=1)
                    continue

                # Setup Prophet model for this fold.
                model_fold = Prophet(
                    changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_prior_scale=params['seasonality_prior_scale'],
                    holidays_prior_scale=params['holidays_prior_scale'],
                    seasonality_mode=params['seasonality_mode']
                )
                model_fold.add_seasonality(name='custom', period=365.25, fourier_order=params['fourier_order'])
                # Add extra regressors if available.
                for col in ['volume', 'open', 'high', 'low', 'rsi']:
                    if col in train_data.columns:
                        model_fold.add_regressor(col)
                try:
                    model_fold.fit(train_data)
                except Exception as e:
                    self.logger.warning(f"Fold training failed: {e}")
                    current_fold_start += pd.DateOffset(months=1)
                    continue
                
                future_fold = model_fold.make_future_dataframe(periods=30)
                # For extra regressors assume constant (last observed) value over the forecast horizon.
                for col in ['volume', 'open', 'high', 'low', 'rsi']:
                    if col in train_data.columns:
                        future_fold[col] = train_data.iloc[-1][col]
                forecast_fold = model_fold.predict(future_fold)
                # Use the forecast at the end of the test period.
                forecast_point = forecast_fold.loc[(forecast_fold['ds'] - test_end).abs().idxmin()]
                if test_data.empty:
                    current_fold_start += pd.DateOffset(months=1)
                    continue
                # Use the last training price as reference.
                last_train_price = train_data.iloc[-1]['y']
                # Compare direction: if (forecast - last_train) and (actual - last_train) have same sign.
                forecast_direction = np.sign(forecast_point['yhat'] - last_train_price)
                actual_point = test_data.iloc[-1]
                actual_direction = np.sign(actual_point['y'] - last_train_price)
                correct = 1 if forecast_direction == actual_direction else 0
                accuracies.append(correct)
                num_folds += 1
                current_fold_start += pd.DateOffset(months=1)
            if accuracies:
                directional_accuracy = np.mean(accuracies)
            else:
                directional_accuracy = 0
            loss = 1 - directional_accuracy
            return {'loss': loss, 'status': STATUS_OK}

        # Define the search space; using a nested hp.choice so that the best dictionary is returned.
        search_space = {
            'changepoint_prior_scale': hp.uniform('changepoint_prior_scale', 0.001, 0.5),
            'seasonality_prior_scale': hp.uniform('seasonality_prior_scale', 0.01, 10),
            'holidays_prior_scale': hp.uniform('holidays_prior_scale', 0.01, 10),
            'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative']),
            'fourier_order': hp.quniform('fourier_order', 3, 20, 1)
        }

        trials = Trials()
        best = fmin(fn=objective, space=search_space, algo=tpe.suggest,
                    max_evals=50, trials=trials, rstate=np.random.RandomState(42))
        # Adjust the fourier_order to integer.
        best['fourier_order'] = int(best['fourier_order'])
        # For seasonality_mode, hyperopt returns the index; adjust if needed.
        if isinstance(best['seasonality_mode'], int):
            best['seasonality_mode'] = ['additive', 'multiplicative'][best['seasonality_mode']]
        self.logger.info(f"Best hyperparameters: {best}")
        return best

    def _compute_slope_ttest(self, x: np.ndarray, y: np.ndarray,
                               yhat_lower: np.ndarray, yhat_upper: np.ndarray):
        """
        Compute a simple linear regression on forecast points and return:
          - slope, standard error, t-statistic and p-value.
        For each forecast point, we approximate its standard error as:
            
            se_i ≈ (yhat_upper_i - yhat_lower_i) / (2 * 1.96)

        Then a linear regression is performed (using stats.linregress)
        and the t-statistic is computed as slope / std_err.
        """
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        t_stat = slope / std_err if std_err != 0 else 0
        return slope, std_err, t_stat, p_value

    def _detect_market_regime(self, df: pd.DataFrame) -> bool:
        """
        Use a Hidden Markov Model (HMM) to detect market regimes.
        Here, we use the 21-day rolling standard deviation of the returns as a proxy for volatility.
        We then fit a Gaussian HMM with 2 states and label the state having the highest mean volatility
        as the high volatility regime.
        Returns True if the current (last date) regime is high volatility.
        """
        df_sorted = df.sort_values(by='ds')
        df_sorted['return'] = df_sorted['y'].pct_change().fillna(0)
        volatility = df_sorted['return'].rolling(window=21).std().fillna(0)
        try:
            model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
            reshaped_vol = volatility.values.reshape(-1, 1)
            model.fit(reshaped_vol)
            hidden_states = model.predict(reshaped_vol)
            state_means = [np.mean(reshaped_vol[hidden_states == i]) for i in range(model.n_components)]
            high_vol_state = np.argmax(state_means)
            current_state = hidden_states[-1]
            is_high_vol = (current_state == high_vol_state)
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
            is_high_vol = False
        return is_high_vol

    def _process_single_ticker(self, ticker: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single ticker:
          • Ensure at least 3 years of data
          • Filter the latest 3 years (for CV and final training)
          • Compute technical indicators (e.g. RSI)
          • Detect market regime (HMM on volatility)
          • Load a cached model if available; otherwise run hyperparameter tuning
          • Train a final Prophet model (with extra regressors)
          • Forecast future prices (for horizon_short, mid, long)
          • Use confidence intervals along with a t–test on forecast slope
          • Decide on a probabilistic trading signal and strength.
          • Merge the forecast signals with price data for backtesting and then apply risk management.
        """
        data = data.sort_index()
        if data.empty or len(data) < 750:
            self.logger.warning(f"Insufficient data for {ticker} to process (requires ~3 years)")
            return pd.DataFrame()

        # Ensure the index is datetime and filter last 3 years.
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        max_date = data.index.max()
        three_years_ago = max_date - pd.DateOffset(years=3)
        df_recent = data[data.index >= three_years_ago].copy()
        df_recent.reset_index(inplace=True)
        if 'date' in df_recent.columns:
            df_recent.rename(columns={'date': 'ds'}, inplace=True)
        else:
            df_recent.rename(columns={df_recent.columns[0]: 'ds'}, inplace=True)
        df_recent['ds'] = pd.to_datetime(df_recent['ds'])
        # Prophet requires the target column to be named "y" (we use 'close').
        df_recent['y'] = df_recent['close']

        # Compute technical indicator: 14-day RSI on closing price.
        delta = df_recent['y'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        df_recent['rsi'] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-6)))
        df_recent.fillna(method='bfill', inplace=True)

        # Detect market regime.
        high_vol_regime = self._detect_market_regime(df_recent)

        # Prepare a cached model filepath.
        model_filepath = os.path.join(self.model_dir, f"{ticker}.pkl")
        model_cached = None
        if os.path.exists(model_filepath):
            try:
                model_cached = joblib.load(model_filepath)
            except Exception as e:
                self.logger.warning(f"Error loading cached model for {ticker}: {e}")

        if model_cached is None:
            # Tune hyperparameters using forward crossvalidation.
            best_params = self._tune_hyperparameters(df_recent)
        else:
            # If cached model contains stored hyperparameters use them.
            best_params = getattr(model_cached, 'hyper_parameters', None)
            if best_params is None:
                best_params = self._tune_hyperparameters(df_recent)

        # Train final Prophet model on the full 3-years (df_recent)
        model_final = Prophet(
            changepoint_prior_scale=best_params['changepoint_prior_scale'],
            seasonality_prior_scale=best_params['seasonality_prior_scale'],
            holidays_prior_scale=best_params['holidays_prior_scale'],
            seasonality_mode=best_params['seasonality_mode']
        )
        model_final.add_seasonality(name='custom', period=365.25, fourier_order=best_params['fourier_order'])
        for reg in ['volume', 'open', 'high', 'low', 'rsi']:
            model_final.add_regressor(reg)
        model_final.fit(df_recent)
        model_final.hyper_parameters = best_params
        try:
            joblib.dump(model_final, model_filepath)
        except Exception as e:
            self.logger.warning(f"Error saving model for {ticker}: {e}")

        # Forecast into the future (up to horizon_long days)
        horizon_long = self.params.get('horizon_long', 30)
        future = model_final.make_future_dataframe(periods=horizon_long, freq='D')
        # For extra regressors, fill with the last observed value.
        last_values = {}
        for reg in ['volume', 'open', 'high', 'low', 'rsi']:
            last_values[reg] = df_recent.iloc[-1][reg]
            future[reg] = last_values[reg]
        forecast = model_final.predict(future)

        # Define target dates for each horizon.
        current_date = df_recent['ds'].max()
        current_price = df_recent.iloc[-1]['y']
        horizon_short = self.params.get('horizon_short', 7)
        horizon_mid = self.params.get('horizon_mid', 14)
        target_date_short = current_date + pd.Timedelta(days=horizon_short)
        target_date_mid = current_date + pd.Timedelta(days=horizon_mid)
        target_date_long = current_date + pd.Timedelta(days=horizon_long)

        def get_forecast_row(target):
            return forecast.loc[(forecast['ds'] - target).abs().idxmin()]

        row_short = get_forecast_row(target_date_short)
        row_mid = get_forecast_row(target_date_mid)
        row_long = get_forecast_row(target_date_long)

        # Extract the median forecast and the confidence bounds.
        f_short = row_short['yhat']
        f_mid = row_mid['yhat']
        f_long = row_long['yhat']
        f_short_lower = row_short['yhat_lower']
        f_short_upper = row_short['yhat_upper']
        f_mid_lower = row_mid['yhat_lower']
        f_mid_upper = row_mid['yhat_upper']
        f_long_lower = row_long['yhat_lower']
        f_long_upper = row_long['yhat_upper']

        # Compute the linear regression on the forecast points.
        x_points = np.array([horizon_short, horizon_mid, horizon_long])
        y_points = np.array([f_short, f_mid, f_long])
        y_lower_points = np.array([f_short_lower, f_mid_lower, f_long_lower])
        y_upper_points = np.array([f_short_upper, f_mid_upper, f_long_upper])
        slope, slope_std, t_stat, p_val = self._compute_slope_ttest(
            x_points, y_points, y_lower_points, y_upper_points)

        # ----- Signal Generation Logic -----
        # For a long signal:
        #   (i) f_short_lower > P_t  (forecast short lower bound above current price)
        #   (ii) f_long_upper > f_mid_lower (forecast confidence intervals are monotonically rising)
        #   (iii) t_stat > 2 (slope significantly positive)
        #
        # For a short signal:
        #   (i) f_short_upper < P_t
        #   (ii) f_long_lower < f_mid_upper
        #   (iii) t_stat < -2 (slope significantly negative)
        signal = 0
        if (f_short_lower > current_price and f_long_upper > f_mid_lower and t_stat > 2):
            signal = 1
        elif (f_short_upper < current_price and f_long_lower < f_mid_upper and t_stat < -2):
            signal = -1

        # Compute a weighted signal strength.
        signal_strength = 0
        if signal == 1:
            signal_strength = (f_long - current_price) / current_price * (t_stat / 2)
        elif signal == -1:
            signal_strength = (current_price - f_long) / current_price * (abs(t_stat) / 2)

        # If market is in a high–volatility regime, disable trading.
        if high_vol_regime:
            signal = 0
            signal_strength = 0

        # Create the result. If latest_only is True, output one row;
        # otherwise, create a backtest DataFrame including every row from the current date onward.
        if self.params.get('latest_only', False):
            result = pd.DataFrame([{
                'ticker': ticker,
                'date': current_date,
                'current_price': current_price,
                'forecast_short': f_short,
                'forecast_mid': f_mid,
                'forecast_long': f_long,
                'signal': signal,
                'signal_strength': signal_strength,
                't_stat': t_stat,
                'p_val': p_val
            }])
        else:
            backtest_data = data[data.index >= current_date].copy()
            backtest_data['ticker'] = ticker
            backtest_data['current_price'] = current_price
            backtest_data['forecast_short'] = f_short
            backtest_data['forecast_mid'] = f_mid
            backtest_data['forecast_long'] = f_long
            backtest_data['signal'] = signal
            backtest_data['signal_strength'] = signal_strength
            backtest_data['t_stat'] = t_stat
            backtest_data['p_val'] = p_val
            result = backtest_data.reset_index()
        
        # Apply risk management (stop loss, take profit, slippage, transaction cost)
        result = self.risk_manager.apply(result, initial_position=0)
        return result

    def generate_signals(self,
                         tickers: Union[str, List[str]],
                         start_date: str = None,
                         end_date: str = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals for a list of tickers using the Prophet Momentum strategy.
        This method:
          (1) Retrieves price data using get_historical_prices().
          (2) For each ticker, processes the data (including technical indicator calculation,
              hyperparameter tuning via forward crossvalidation, regime detection, and final forecasting).
          (3) Merges the forecast with historical data if backtesting is desired, or returns
              only the latest row for end–of–day (EOD) signals.
          (4) Applies risk management to the full daily time–series.
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        price_data_all = self.get_historical_prices(tickers, from_date=start_date, to_date=end_date)
        results = []

        # You can parallelize this loop via ThreadPoolExecutor if you wish.
        for ticker in tickers:
            if isinstance(price_data_all.index, pd.MultiIndex):
                try:
                    ticker_data = price_data_all.xs(ticker, level='ticker')
                except KeyError:
                    self.logger.warning(f"No data found for ticker {ticker}")
                    continue
            else:
                ticker_data = price_data_all.copy()
            res = self._process_single_ticker(ticker, ticker_data)
            if not res.empty:
                results.append(res)

        if results:
            final_df = pd.concat(results, ignore_index=True)
            return final_df
        else:
            return pd.DataFrame()