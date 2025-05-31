# trading_system/src/strategies/advanced/garchx_strat.py

"""
GARCH‑X Strategy Implementation with Two-Phase Optimization

This module implements a GARCH‑X strategy with separate time series model training
and strategy parameter optimization phases:

Phase 1 (Time Series Training):
   - Uses forward cross-validation to train EGARCH‑X models for each ticker
   - Prevents data leakage by applying winsorization only within each training fold
   - Stores trained models per ticker to avoid retraining during strategy optimization
   - Selects best model configuration using out-of-sample AIC on validation sets

Phase 2 (Strategy Optimization):
   - Loads pre-trained time series models for forecasting
   - Optimizes non-time series parameters (signal thresholds, risk management)
   - Uses scale-invariant measures to ensure unit independence
   - Leverages cached models for faster evaluation

Key Features:
   - Proper forward cross-validation for time series models
   - Per-ticker model storage and management
   - Two-phase optimization to separate concerns
   - Scale-invariant signal generation
   - Robust error handling and fallback mechanisms
   - Efficient parallel processing and caching
"""

import os
import numpy as np
import pandas as pd
import logging
import pickle
import hashlib
import json
from typing import Dict, Optional, Union, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

from arch import arch_model
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.database.config import DatabaseConfig
from src.strategies.risk_management import RiskManager

class GarchXStrategyStrategy(BaseStrategy):
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the GARCH‑X Strategy with two-phase optimization support.

        Parameters:
            db_config (DatabaseConfig): Database configuration for data access.
            params (Optional[Dict]): Strategy parameters including:
                Mode Parameters:
                - 'mode': str, "train" to fit & save models or "forecast" to use existing models
                
                Forecast Parameters:
                - 'forecast_horizon': int, forecast days ahead (default: 30)
                - 'forecast_lookback': int, lookback window for training (default: 200)
                
                Signal Parameters (Scale-Invariant):
                - 'min_volume_strength': float, minimum z-score threshold (default: 1.0)
                - 'signal_strength_threshold': float, minimum signal strength (default: 0.1)
                
                Model Parameters:
                - 'pca_components': int, PCA components for dimensionality reduction (default: 5)
                - 'winsorize_quantiles': tuple, quantiles for outlier handling (default: (0.01, 0.99))
                - 'vol_window': int, volatility calculation window (default: 20)
                - 'cv_folds': int, forward CV folds for model training (default: 3)
                
                Position Sizing:
                - 'capital': float, base capital for position sizing (default: 1.0)
                - 'long_only': bool, allow only long positions (default: True)
                
                Risk Management:
                - 'stop_loss_pct', 'take_profit_pct', 'trail_stop_pct': risk parameters
                - 'slippage_pct', 'transaction_cost_pct': execution cost parameters
        """
        default_params = {
            # Mode control
            'mode': "train",
            
            # Forecast parameters
            'forecast_horizon': 30,
            'forecast_lookback': 200,
            
            # Signal parameters (scale-invariant)
            'min_volume_strength': 1.0,
            'signal_strength_threshold': 0.1,
            
            # Model parameters
            'pca_components': 5,
            'winsorize_quantiles': (0.01, 0.99),
            'vol_window': 20,
            'cv_folds': 3,
            
            # Position sizing
            'capital': 1.0,
            'long_only': True,
            
            # Risk management
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'trail_stop_pct': 0,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001,
        }
        default_params.update(params or {})
        super().__init__(db_config, default_params)
        
        # Core parameters
        self.mode = self.params.get("mode", "train")
        self.forecast_horizon = int(self.params.get('forecast_horizon'))
        self.forecast_lookback = int(self.params.get('forecast_lookback'))
        self.cv_folds = int(self.params.get('cv_folds'))
        
        # Signal parameters (scale-invariant)
        self.min_volume_strength = float(self.params.get('min_volume_strength'))
        self.signal_strength_threshold = float(self.params.get('signal_strength_threshold'))
        
        # Model parameters
        self.pca_components = int(self.params.get('pca_components'))
        self.winsorize_quantiles = self.params.get('winsorize_quantiles')
        self.vol_window = int(self.params.get('vol_window'))
        
        # Position sizing
        self.capital = float(self.params.get('capital'))
        self.long_only = bool(self.params.get('long_only'))
        
        # Initialize risk manager
        risk_params = {
            'stop_loss_pct': self.params.get('stop_loss_pct'),
            'take_profit_pct': self.params.get('take_profit_pct'),
            'trail_stop_pct': self.params.get('trail_stop_pct'),
            'slippage_pct': self.params.get('slippage_pct'),
            'transaction_cost_pct': self.params.get('transaction_cost_pct'),
        }
        self.risk_manager = RiskManager(**risk_params)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Model storage setup
        self.model_base_folder = os.path.join("trading_system", "models", "garchx")
        self.model_config_folder = os.path.join(self.model_base_folder, "configs")
        self.model_cache_folder = os.path.join(self.model_base_folder, "cache")
        
        # Create necessary directories
        for folder in [self.model_base_folder, self.model_config_folder, self.model_cache_folder]:
            os.makedirs(folder, exist_ok=True)
        
        # Model cache for performance
        self._model_cache = {}
        self._data_cache = {}

    def train_time_series_models(
        self,
        tickers: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Train and store GARCH-X models for specified tickers using forward cross-validation.
        
        This method implements Phase 1 of the two-phase optimization:
        1. Retrieves historical data for each ticker
        2. Applies forward cross-validation to prevent data leakage
        3. Trains EGARCH-X models on each training fold
        4. Validates on out-of-sample data
        5. Stores best performing model per ticker
        
        Parameters:
            tickers (Union[str, List[str]]): Ticker(s) to train models for
            start_date (Optional[str]): Training start date
            end_date (Optional[str]): Training end date
            
        Returns:
            Dict[str, bool]: Success status for each ticker's model training
        """
        if isinstance(tickers, str):
            tickers = [tickers]
            
        self.logger.info(f"Starting time series model training for {len(tickers)} tickers")
        self.logger.info(f"Training period: {start_date} to {end_date}")
        self.logger.info(f"Using {self.cv_folds} forward CV folds")
        
        # Use parallel processing for multiple tickers
        results = {}
        if len(tickers) > 1:
            results = Parallel(n_jobs=-1)(
                delayed(self._train_single_ticker_model)(ticker, start_date, end_date)
                for ticker in tickers
            )
            results = {ticker: success for ticker, success in zip(tickers, results)}
        else:
            ticker = tickers[0]
            success = self._train_single_ticker_model(ticker, start_date, end_date)
            results = {ticker: success}
        
        successful_models = sum(results.values())
        self.logger.info(f"Training completed: {successful_models}/{len(tickers)} models trained successfully")
        
        return results

    def _train_single_ticker_model(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> bool:
        """
        Train GARCH-X model for a single ticker using forward cross-validation.
        
        Parameters:
            ticker (str): Ticker symbol to train model for
            start_date (Optional[str]): Training start date  
            end_date (Optional[str]): Training end date
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            self.logger.info(f"Training model for ticker: {ticker}")
            
            # Get training data with buffer for feature engineering
            lookback_buffer = max(self.forecast_lookback, self.forecast_horizon) + 50
            
            if start_date and end_date:
                data = self.get_historical_prices(
                    ticker, from_date=start_date, to_date=end_date, lookback=lookback_buffer
                )
            else:
                data = self.get_historical_prices(ticker, lookback=lookback_buffer)
            
            if data.empty or len(data) < self.forecast_lookback:
                self.logger.warning(f"Insufficient data for {ticker}: {len(data)} records")
                return False
            
            # Prepare features for model training
            processed_data = self._prepare_model_features(data)
            if processed_data.empty:
                self.logger.warning(f"Feature preparation failed for {ticker}")
                return False
            
            # Implement forward cross-validation
            cv_results = self._forward_cross_validate_model(processed_data, ticker)
            if not cv_results:
                self.logger.warning(f"Cross-validation failed for {ticker}")
                return False
            
            # Select and store best model
            best_model_info = self._select_best_model(cv_results, ticker)
            if best_model_info is None:
                self.logger.warning(f"No valid model found for {ticker}")
                return False
            
            # Save model to disk
            success = self._save_ticker_model(ticker, best_model_info)
            if success:
                self.logger.info(f"Successfully trained and saved model for {ticker}")
            else:
                self.logger.error(f"Failed to save model for {ticker}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error training model for {ticker}: {str(e)}")
            return False

    def _prepare_model_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for GARCH-X model training with proper handling of MultiIndex.
        
        Parameters:
            data (pd.DataFrame): Raw price data
            
        Returns:
            pd.DataFrame: Processed data with features
        """
        try:
            df = data.copy()
            
            # Handle MultiIndex if present
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0, drop=True)
            df.sort_index(inplace=True)

            # Core feature engineering
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))
            
            # Volume features (scale-invariant)
            df["vol_log"] = np.log(df["volume"])
            df["vol_mean"] = df["vol_log"].rolling(window=self.vol_window, min_periods=self.vol_window).mean()
            df["vol_std"] = df["vol_log"].rolling(window=self.vol_window, min_periods=self.vol_window).std()
            df["vol_z"] = (df["vol_log"] - df["vol_mean"]) / (df["vol_std"] + 1e-8)
            df["vol_ratio"] = df["volume"] / (df["volume"].rolling(window=self.vol_window, min_periods=self.vol_window).mean() + 1e-8)
            
            # Price range features (scale-invariant)
            df["hl_range"] = (df["high"] - df["low"]) / (df["close"].shift(1) + 1e-8)
            df["oc_range"] = (df["open"] - df["close"]).abs() / (df["close"].shift(1) + 1e-8)

            # Volatility proxies
            df["log_hl"] = np.log((df["high"] + 1e-8) / (df["low"] + 1e-8))
            df["parkinson_vol"] = np.sqrt(
                df["log_hl"].rolling(window=20, min_periods=20).apply(
                    lambda x: np.mean(x**2), raw=True
                ) / (4 * np.log(2))
            )
            
            # Overnight and intraday returns
            df["overnight_ret"] = np.log((df["open"] + 1e-8) / (df["close"].shift(1) + 1e-8))
            df["oc_ret"] = np.log((df["close"] + 1e-8) / (df["open"] + 1e-8))
            df["yz_vol"] = np.sqrt(
                df["overnight_ret"].rolling(window=20, min_periods=20).var() +
                df["oc_ret"].rolling(window=20, min_periods=20).var() +
                0.5 * df["hl_range"].rolling(window=20, min_periods=20).var()
            )

            # Lag features (1-5 periods)
            for lag in range(1, 6):
                df[f"lag{lag}_return"] = df["log_return"].shift(lag)
                df[f"lag{lag}_vol"] = df["vol_z"].shift(lag)
                df[f"lag{lag}_hl"] = df["hl_range"].shift(lag)

            # Remove rows with NaN values
            df.dropna(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in feature preparation: {str(e)}")
            return pd.DataFrame()

    def _forward_cross_validate_model(self, data: pd.DataFrame, ticker: str) -> List[Dict[str, Any]]:
        """
        Perform forward cross-validation for GARCH-X model training.
        
        This prevents data leakage by ensuring that:
        1. Each fold uses only past data for training
        2. Winsorization parameters are computed only on training data
        3. Model validation is performed on truly out-of-sample data
        
        Parameters:
            data (pd.DataFrame): Processed feature data
            ticker (str): Ticker symbol for logging
            
        Returns:
            List[Dict[str, Any]]: Results from each CV fold
        """
        cv_results = []
        
        # Calculate fold boundaries
        total_periods = len(data)
        min_train_size = self.forecast_lookback
        test_size = max(30, self.forecast_horizon)  # Minimum test size
        
        if total_periods < min_train_size + test_size * self.cv_folds:
            self.logger.warning(f"Insufficient data for {self.cv_folds} folds on {ticker}")
            return cv_results
        
        # Progressive validation splits
        for fold in range(self.cv_folds):
            train_end_idx = min_train_size + fold * test_size
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + test_size, total_periods)
            
            if test_end_idx <= test_start_idx:
                break
                
            train_data = data.iloc[:train_end_idx].copy()
            test_data = data.iloc[test_start_idx:test_end_idx].copy()
            
            self.logger.debug(f"Fold {fold+1}: Train={len(train_data)}, Test={len(test_data)}")
            
            try:
                # Train model on fold data (winsorization applied only to training data)
                fold_result = self._train_model_on_fold(train_data, test_data, fold, ticker)
                if fold_result:
                    cv_results.append(fold_result)
                    
            except Exception as e:
                self.logger.warning(f"Fold {fold+1} failed for {ticker}: {str(e)}")
                continue
        
        return cv_results

    def _train_model_on_fold(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        fold: int,
        ticker: str
    ) -> Optional[Dict[str, Any]]:
        """
        Train EGARCH-X model on a single fold and validate on test data.
        
        Parameters:
            train_data (pd.DataFrame): Training data for this fold
            test_data (pd.DataFrame): Test data for validation
            fold (int): Fold number
            ticker (str): Ticker symbol
            
        Returns:
            Optional[Dict[str, Any]]: Model results if successful, None otherwise
        """
        try:
            # Prepare features (winsorization on training data only)
            exog_cols = [f"lag{l}_{feat}" for l in range(1, 6) 
                        for feat in ["return", "vol", "hl"]]
            
            X_train = train_data[exog_cols].copy()
            y_train = train_data["log_return"].copy()
            
            # Apply winsorization using ONLY training data statistics
            train_winsorize_params = {}
            for col in X_train.columns:
                lower = X_train[col].quantile(self.winsorize_quantiles[0])
                upper = X_train[col].quantile(self.winsorize_quantiles[1])
                train_winsorize_params[col] = (lower, upper)
                X_train[col] = X_train[col].clip(lower, upper)
            
            # Winsorize target variable
            y_lower = y_train.quantile(self.winsorize_quantiles[0])
            y_upper = y_train.quantile(self.winsorize_quantiles[1])
            y_train = y_train.clip(y_lower, y_upper)
            train_winsorize_params['target'] = (y_lower, y_upper)
            
            # PCA on training data
            n_components = min(self.pca_components, X_train.shape[1])
            pca = PCA(n_components=n_components)
            X_train_pca = pd.DataFrame(
                pca.fit_transform(X_train),
                index=X_train.index,
                columns=[f'pca_{i}' for i in range(n_components)]
            )
            
            # Try different EGARCH configurations
            candidate_orders = [(1, 1), (1, 2), (2, 1)]
            best_model = None
            best_aic = np.inf
            best_order = None
            
            for (p, q) in candidate_orders:
                try:
                    model = arch_model(
                        y_train,
                        x=X_train_pca,
                        mean="ARX",
                        lags=0,
                        vol="EGARCH",
                        p=p,
                        q=q,
                        dist="t"
                    )
                    result = model.fit(disp="off")
                    
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_model = result
                        best_order = (p, q)
                        
                except Exception as e:
                    self.logger.debug(f"EGARCH({p},{q}) failed on fold {fold}: {str(e)}")
                    continue
            
            if best_model is None:
                return None
            
            # Validate on test data
            X_test = test_data[exog_cols].copy()
            y_test = test_data["log_return"].copy()
            
            # Apply same winsorization parameters from training
            for col in X_test.columns:
                if col in train_winsorize_params:
                    lower, upper = train_winsorize_params[col]
                    X_test[col] = X_test[col].clip(lower, upper)
            
            # Transform test data using fitted PCA
            X_test_pca = pd.DataFrame(
                pca.transform(X_test),
                index=X_test.index,
                columns=[f'pca_{i}' for i in range(n_components)]
            )
            
            # Calculate out-of-sample performance
            try:
                forecast = best_model.forecast(horizon=len(test_data), x=X_test_pca, reindex=False)
                forecast_returns = forecast.mean.iloc[-1].values
                actual_returns = y_test.values
                
                # Calculate validation metrics
                mse = np.mean((forecast_returns - actual_returns) ** 2)
                directional_accuracy = np.mean(
                    np.sign(forecast_returns) == np.sign(actual_returns)
                )
                
            except Exception as e:
                self.logger.debug(f"Forecast validation failed on fold {fold}: {str(e)}")
                mse = np.inf
                directional_accuracy = 0.0
            
            return {
                'fold': fold,
                'model': best_model,
                'pca': pca,
                'order': best_order,
                'aic': best_aic,
                'mse': mse,
                'directional_accuracy': directional_accuracy,
                'winsorize_params': train_winsorize_params,
                'n_components': n_components
            }
            
        except Exception as e:
            self.logger.error(f"Error training model on fold {fold}: {str(e)}")
            return None

    def _select_best_model(self, cv_results: List[Dict[str, Any]], ticker: str) -> Optional[Dict[str, Any]]:
        """
        Select best model from cross-validation results based on combined AIC and validation performance.
        
        Parameters:
            cv_results (List[Dict[str, Any]]): Results from all CV folds
            ticker (str): Ticker symbol for logging
            
        Returns:
            Optional[Dict[str, Any]]: Best model information
        """
        if not cv_results:
            return None
        
        # Score models based on AIC and validation metrics
        scored_results = []
        for result in cv_results:
            # Normalize metrics (lower is better for both)
            normalized_aic = result['aic'] / np.mean([r['aic'] for r in cv_results])
            normalized_mse = result['mse'] / np.mean([r['mse'] for r in cv_results if np.isfinite(r['mse'])])
            
            # Directional accuracy (higher is better)
            directional_score = result['directional_accuracy']
            
            # Combined score (lower is better)
            combined_score = normalized_aic + normalized_mse - directional_score
            
            scored_results.append((combined_score, result))
        
        # Select best model
        best_score, best_result = min(scored_results, key=lambda x: x[0])
        
        self.logger.info(
            f"Selected model for {ticker}: Fold {best_result['fold']}, "
            f"Order {best_result['order']}, AIC: {best_result['aic']:.2f}, "
            f"MSE: {best_result['mse']:.6f}, Dir.Acc: {best_result['directional_accuracy']:.3f}"
        )
        
        return best_result

    def _save_ticker_model(self, ticker: str, model_info: Dict[str, Any]) -> bool:
        """
        Save trained model information for a ticker.
        
        Parameters:
            ticker (str): Ticker symbol
            model_info (Dict[str, Any]): Model information to save
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Create ticker-specific filename
            safe_ticker = ticker.replace('.', '_').replace('/', '_')
            model_file = os.path.join(self.model_cache_folder, f"{safe_ticker}_model.pkl")
            
            # Prepare data for serialization
            model_data = {
                'model': model_info['model'],
                'pca': model_info['pca'],
                'order': model_info['order'],
                'aic': model_info['aic'],
                'mse': model_info['mse'],
                'directional_accuracy': model_info['directional_accuracy'],
                'winsorize_params': model_info['winsorize_params'],
                'n_components': model_info['n_components'],
                'trained_at': pd.Timestamp.now(),
                'params_hash': self._get_model_params_hash()
            }
            
            # Save to file
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Also save configuration summary
            config_file = os.path.join(self.model_config_folder, f"{safe_ticker}_config.json")
            config_data = {
                'ticker': ticker,
                'order': model_info['order'],
                'aic': float(model_info['aic']),
                'mse': float(model_info['mse']),
                'directional_accuracy': float(model_info['directional_accuracy']),
                'n_components': model_info['n_components'],
                'trained_at': pd.Timestamp.now().isoformat(),
                'params_hash': self._get_model_params_hash()
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model for {ticker}: {str(e)}")
            return False

    def _get_model_params_hash(self) -> str:
        """Generate hash of model-relevant parameters to detect changes."""
        model_params = {
            'forecast_horizon': self.forecast_horizon,
            'forecast_lookback': self.forecast_lookback,
            'pca_components': self.pca_components,
            'winsorize_quantiles': self.winsorize_quantiles,
            'vol_window': self.vol_window,
            'cv_folds': self.cv_folds
        }
        param_str = json.dumps(model_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def _load_ticker_model(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Load trained model for a ticker with validation.
        
        Parameters:
            ticker (str): Ticker symbol
            
        Returns:
            Optional[Dict[str, Any]]: Loaded model data if valid, None otherwise
        """
        try:
            # Check cache first
            if ticker in self._model_cache:
                return self._model_cache[ticker]
            
            safe_ticker = ticker.replace('.', '_').replace('/', '_')
            model_file = os.path.join(self.model_cache_folder, f"{safe_ticker}_model.pkl")
            
            if not os.path.exists(model_file):
                self.logger.warning(f"No trained model found for {ticker}")
                return None
            
            # Load model data
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            # Validate model parameters match current configuration
            current_hash = self._get_model_params_hash()
            saved_hash = model_data.get('params_hash', '')
            
            if saved_hash != current_hash:
                self.logger.warning(
                    f"Model parameters changed for {ticker}. "
                    f"Retrain required (saved: {saved_hash[:8]}, current: {current_hash[:8]})"
                )
                return None
            
            # Cache for future use
            self._model_cache[ticker] = model_data
            
            return model_data
            
        except Exception as e:
            self.logger.error(f"Error loading model for {ticker}: {str(e)}")
            return None

    def generate_signals(
        self,
        ticker: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False,
    ) -> pd.DataFrame:
        """
        Generate trading signals using trained GARCH-X models or train new ones.
        
        This method serves as the main entry point and handles both phases:
        - In "train" mode: trains models and generates signals
        - In "forecast" mode: loads pre-trained models for signal generation
        
        Parameters:
            ticker (Union[str, List[str]]): Ticker symbol(s)
            start_date (Optional[str]): Start date for signal generation
            end_date (Optional[str]): End date for signal generation  
            initial_position (int): Initial position for risk management
            latest_only (bool): If True, return only latest signal per ticker
            
        Returns:
            pd.DataFrame: DataFrame with trading signals and performance metrics
        """
        try:
            # Validate mode and handle training if needed
            if self.mode == "train":
                self.logger.info("Running in training mode - will train models if needed")
            elif self.mode == "forecast":
                self.logger.info("Running in forecast mode - using pre-trained models")
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Must be 'train' or 'forecast'")
            
            # Ensure data buffer for feature engineering
            lookback_buffer = max(self.forecast_lookback, self.forecast_horizon) + 50

            # Get historical data
            if start_date and end_date:
                data = self.get_historical_prices(
                    ticker, from_date=start_date, to_date=end_date, lookback=lookback_buffer
                )
            else:
                data = self.get_historical_prices(ticker, lookback=lookback_buffer)
                data = data.sort_index()

            # Handle multiple tickers
            if isinstance(ticker, list):
                signals_list = Parallel(n_jobs=-1)(
                    delayed(self._calculate_signals_single)(group, latest_only, ticker_name)
                    for ticker_name, group in data.groupby(level=0)
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
                signals = self._calculate_signals_single(data, latest_only, ticker)

            # Apply risk management
            signals = self.risk_manager.apply(signals, initial_position)
            
            # Add performance metrics
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

    def _calculate_signals_single(
        self,
        data: pd.DataFrame,
        latest_only: bool,
        ticker_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate signals for a single ticker using appropriate model mode.
        
        Parameters:
            data (pd.DataFrame): Historical price data
            latest_only (bool): If True, compute only latest signal
            ticker_name (Optional[str]): Ticker name for model loading
            
        Returns:
            pd.DataFrame: DataFrame with signals and forecasts
        """
        try:
            # Prepare features
            df = self._prepare_model_features(data)
            if df.empty:
                self.logger.warning(f"Feature preparation failed for {ticker_name}")
                return pd.DataFrame()

            # Initialize signal columns
            df["forecast_return"] = np.nan
            df["signal"] = np.nan
            df["signal_strength"] = np.nan
            df["position_size"] = np.nan

            if latest_only:
                # Generate forecast for most recent period
                train = df.iloc[-self.forecast_lookback:] if len(df) >= self.forecast_lookback else df
                fc_ret, sig, sig_str, pos_size = self._fit_forecast(train, self.forecast_horizon, ticker_name)
                
                df.loc[df.index[-1], "forecast_return"] = fc_ret
                df.loc[df.index[-1], "signal"] = sig
                df.loc[df.index[-1], "signal_strength"] = sig_str
                df.loc[df.index[-1], "position_size"] = pos_size
                df.fillna(method="ffill", inplace=True)
            else:
                # Rolling forecast approach
                for i in range(len(df) - self.forecast_lookback - self.forecast_horizon + 1):
                    train = df.iloc[i : i + self.forecast_lookback]
                    forecast_dates = df.index[i + self.forecast_lookback : i + self.forecast_lookback + self.forecast_horizon]
                    
                    fc_ret, sig, sig_str, pos_size = self._fit_forecast(train, self.forecast_horizon, ticker_name)
                    
                    df.loc[forecast_dates, "forecast_return"] = fc_ret
                    df.loc[forecast_dates, "signal"] = sig
                    df.loc[forecast_dates, "signal_strength"] = sig_str
                    df.loc[forecast_dates, "position_size"] = pos_size
                
                df.fillna(method="ffill", inplace=True)

            # Prepare output
            cols = ["open", "high", "low", "close", "forecast_return", "signal", "signal_strength", "position_size"]
            result = df[cols].copy()
            
            # Add ticker identifier for multi-ticker processing
            if ticker_name and 'ticker' not in result.columns:
                result['ticker'] = ticker_name
            elif 'ticker' not in result.columns and 'ticker' in data.columns:
                result['ticker'] = data['ticker']
                
            return result

        except Exception as e:
            self.logger.error(f"Error calculating signals for {ticker_name}: {str(e)}")
            return pd.DataFrame()

    def _fit_forecast(
        self,
        train: pd.DataFrame,
        horizon: int,
        ticker_name: Optional[str] = None
    ) -> Tuple[float, int, float, float]:
        """
        Generate forecast using trained model or train new model based on mode.
        
        Parameters:
            train (pd.DataFrame): Training data
            horizon (int): Forecast horizon
            ticker_name (Optional[str]): Ticker name for model loading/saving
            
        Returns:
            Tuple[float, int, float, float]: (forecast_return, signal, signal_strength, position_size)
        """
        try:
            if self.mode == "forecast" and ticker_name:
                # Load pre-trained model
                model_data = self._load_ticker_model(ticker_name)
                if model_data is None:
                    self.logger.warning(f"No trained model for {ticker_name}, falling back to training mode")
                    return self._train_and_forecast(train, horizon, ticker_name)
                else:
                    return self._forecast_with_model(train, horizon, model_data)
            else:
                # Train new model (or retrain)
                return self._train_and_forecast(train, horizon, ticker_name)
                
        except Exception as e:
            self.logger.error(f"Error in fit_forecast for {ticker_name}: {str(e)}")
            # Fallback to simple forecast
            return self._fallback_forecast(train)

    def _forecast_with_model(
        self,
        train: pd.DataFrame,
        horizon: int,
        model_data: Dict[str, Any]
    ) -> Tuple[float, int, float, float]:
        """
        Generate forecast using pre-trained model.
        
        Parameters:
            train (pd.DataFrame): Recent data for forecast input
            horizon (int): Forecast horizon  
            model_data (Dict[str, Any]): Pre-trained model data
            
        Returns:
            Tuple[float, int, float, float]: Forecast results
        """
        try:
            model = model_data['model']
            pca = model_data['pca']
            winsorize_params = model_data['winsorize_params']
            n_components = model_data['n_components']
            
            # Prepare exogenous features
            exog_cols = [f"lag{l}_{feat}" for l in range(1, 6) 
                        for feat in ["return", "vol", "hl"]]
            
            X_recent = train[exog_cols].iloc[-1:].copy()
            
            # Apply winsorization using stored parameters
            for col in X_recent.columns:
                if col in winsorize_params:
                    lower, upper = winsorize_params[col]
                    X_recent[col] = X_recent[col].clip(lower, upper)
            
            # Transform using stored PCA
            X_recent_pca = pd.DataFrame(
                pca.transform(X_recent),
                columns=[f'pca_{i}' for i in range(n_components)]
            )
            
            # Replicate for forecast horizon
            X_forecast_pca = pd.DataFrame(
                np.tile(X_recent_pca.values, (horizon, 1)),
                columns=X_recent_pca.columns
            )
            
            # Generate forecast
            forecast = model.forecast(horizon=horizon, x=X_forecast_pca, reindex=False)
            fc_means = forecast.mean.iloc[-1].values
            cumulative_ret = np.exp(np.sum(fc_means)) - 1
            forecast_vol = np.sqrt(forecast.variance.iloc[-1].mean())
            
            return self._calculate_signal_from_forecast(cumulative_ret, forecast_vol, train)
            
        except Exception as e:
            self.logger.error(f"Error in forecast with model: {str(e)}")
            return self._fallback_forecast(train)

    def _train_and_forecast(
        self,
        train: pd.DataFrame,
        horizon: int,
        ticker_name: Optional[str] = None
    ) -> Tuple[float, int, float, float]:
        """
        Train model and generate forecast (original functionality with improvements).
        
        Parameters:
            train (pd.DataFrame): Training data
            horizon (int): Forecast horizon
            ticker_name (Optional[str]): Ticker name for potential model saving
            
        Returns:
            Tuple[float, int, float, float]: Forecast results
        """
        try:
            # Prepare features
            exog_cols = [f"lag{l}_{feat}" for l in range(1, 6) 
                        for feat in ["return", "vol", "hl"]]
            
            X_train = train[exog_cols].copy()
            y_train = train["log_return"].copy()

            # Apply winsorization (no look-ahead bias within training sample)
            winsorize_params = {}
            for col in X_train.columns:
                lower = X_train[col].quantile(self.winsorize_quantiles[0])
                upper = X_train[col].quantile(self.winsorize_quantiles[1])
                winsorize_params[col] = (lower, upper)
                X_train[col] = X_train[col].clip(lower, upper)
            
            # Winsorize target
            y_lower = y_train.quantile(self.winsorize_quantiles[0])
            y_upper = y_train.quantile(self.winsorize_quantiles[1])
            y_train = y_train.clip(y_lower, y_upper)
            winsorize_params['target'] = (y_lower, y_upper)

            # PCA for dimensionality reduction
            n_components = min(self.pca_components, X_train.shape[1])
            pca = PCA(n_components=n_components)
            X_train_pca = pd.DataFrame(
                pca.fit_transform(X_train),
                index=X_train.index,
                columns=[f'pca_{i}' for i in range(n_components)]
            )

            # Try different EGARCH configurations
            candidate_orders = [(1, 1), (1, 2), (2, 1)]
            best_result = None
            best_aic = np.inf
            best_order = None

            for (p, q) in candidate_orders:
                try:
                    model = arch_model(
                        y_train,
                        x=X_train_pca,
                        mean="ARX",
                        lags=0,
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
                    self.logger.debug(f"EGARCH({p},{q}) model failed: {e}")
                    continue

            if best_result is None:
                self.logger.warning("All EGARCH models failed, using fallback")
                return self._fallback_forecast(train)

            # Generate forecast
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

            # Optionally save model in train mode
            if self.mode == "train" and ticker_name:
                model_info = {
                    'model': best_result,
                    'pca': pca,
                    'order': best_order,
                    'aic': best_aic,
                    'mse': 0.0,  # Not calculated in this context
                    'directional_accuracy': 0.0,  # Not calculated in this context
                    'winsorize_params': winsorize_params,
                    'n_components': n_components
                }
                self._save_ticker_model(ticker_name, model_info)

            return self._calculate_signal_from_forecast(cumulative_ret, forecast_vol, train)

        except Exception as e:
            self.logger.error(f"Error in train and forecast: {str(e)}")
            return self._fallback_forecast(train)

    def _calculate_signal_from_forecast(
        self,
        cumulative_ret: float,
        forecast_vol: float,
        train: pd.DataFrame
    ) -> Tuple[float, int, float, float]:
        """
        Calculate trading signal from forecast using scale-invariant measures.
        
        Parameters:
            cumulative_ret (float): Forecasted cumulative return
            forecast_vol (float): Forecasted volatility
            train (pd.DataFrame): Training data for additional indicators
            
        Returns:
            Tuple[float, int, float, float]: (forecast_return, signal, signal_strength, position_size)
        """
        try:
            # Risk-adjusted return (scale-invariant)
            if forecast_vol > 1e-8:
                risk_adj_return = cumulative_ret / forecast_vol
            else:
                risk_adj_return = 0.0

            # Volume strength (already z-score, scale-invariant)
            vol_strength = train["vol_z"].iloc[-1] if "vol_z" in train.columns else 0.0
            
            # Signal strength combines risk-adjusted return and volume
            signal_strength = risk_adj_return * vol_strength

            # Generate signal based on thresholds
            signal_meets_strength = abs(signal_strength) > self.signal_strength_threshold
            volume_meets_threshold = abs(vol_strength) > self.min_volume_strength
            
            if risk_adj_return > 0 and signal_meets_strength and volume_meets_threshold:
                sig = 1
            elif risk_adj_return < 0 and signal_meets_strength and volume_meets_threshold:
                sig = 0 if self.long_only else -1
            else:
                sig = 0

            # Position sizing based on risk-adjusted return
            position_size = risk_adj_return * self.capital

            return cumulative_ret, sig, signal_strength, position_size

        except Exception as e:
            self.logger.error(f"Error calculating signal from forecast: {str(e)}")
            return 0.0, 0, 0.0, 0.0

    def _fallback_forecast(self, train: pd.DataFrame) -> Tuple[float, int, float, float]:
        """
        Generate simple fallback forecast when GARCH models fail.
        
        Parameters:
            train (pd.DataFrame): Training data
            
        Returns:
            Tuple[float, int, float, float]: Simple forecast results
        """
        try:
            # Simple EWMA forecast
            if "log_return" in train.columns and len(train) > 10:
                ewma_return = train['log_return'].ewm(span=30).mean().iloc[-1]
                ewma_vol = train['log_return'].ewm(span=30).std().iloc[-1]
            else:
                ewma_return = 0.0
                ewma_vol = 0.01
            
            # Use Parkinson volatility if available
            if "parkinson_vol" in train.columns:
                forecast_vol = train["parkinson_vol"].iloc[-1]
                if pd.isna(forecast_vol) or forecast_vol <= 0:
                    forecast_vol = ewma_vol
            else:
                forecast_vol = ewma_vol

            return self._calculate_signal_from_forecast(ewma_return, forecast_vol, train)

        except Exception as e:
            self.logger.error(f"Error in fallback forecast: {str(e)}")
            return 0.0, 0, 0.0, 0.0

    def clear_model_cache(self):
        """Clear in-memory model cache to free memory."""
        self._model_cache.clear()
        self._data_cache.clear()
        self.logger.info("Model cache cleared")

    def get_model_status(self, tickers: Union[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """
        Get status of trained models for specified tickers.
        
        Parameters:
            tickers (Union[str, List[str]]): Ticker(s) to check
            
        Returns:
            Dict[str, Dict[str, Any]]: Status information per ticker
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        status = {}
        for ticker in tickers:
            safe_ticker = ticker.replace('.', '_').replace('/', '_')
            model_file = os.path.join(self.model_cache_folder, f"{safe_ticker}_model.pkl")
            config_file = os.path.join(self.model_config_folder, f"{safe_ticker}_config.json")
            
            ticker_status = {
                'model_exists': os.path.exists(model_file),
                'config_exists': os.path.exists(config_file),
                'cached': ticker in self._model_cache
            }
            
            if ticker_status['config_exists']:
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    ticker_status.update(config)
                except Exception as e:
                    ticker_status['config_error'] = str(e)
            
            status[ticker] = ticker_status
        
        return status