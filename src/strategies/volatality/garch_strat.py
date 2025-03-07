# trading_system/src/strategies/volatality/garch_model.py

# TODO Long Only

import pandas as pd
import numpy as np
import warnings
from arch import arch_model
from datetime import timedelta
from typing import Dict, Optional, Union, List

from src.strategies.base_strat import BaseStrategy
from src.database.config import DatabaseConfig
from src.strategies.risk_management import RiskManager


class GARCHModel(BaseStrategy):
    """
    GARCH Model Strategy with Integrated Risk Management.
    
    This strategy uses a GARCH(1,1) model to forecast volatility and generate trading signals based 
    on expected volatility changes.
    
    Mathematical Formulation:
    -------------------------
    1. Returns are computed as either log returns:
           r_t = 100 * ln(P_t / P_{t-1})
       or simple returns:
           r_t = 100 * (P_t / P_{t-1} - 1)
       
    2. A rolling window of size N (window_size) is used to fit a GARCH(1,1) model:
           σ_t² = ω + α·r_{t-1}² + β·σ_{t-1}²
       
    3. The model forecasts the variance; its square-root gives the forecasted volatility:
           forecast_volatility = √(forecast_variance)
       
    4. Historical volatility is computed over the same window (annualized):
           hist_vol = std(window_returns) * √252
       
    5. The relative volatility change is computed as:
           vol_change = (forecast_volatility - hist_vol) / hist_vol
       
    6. Trading signals are generated as follows:
           - If vol_change > vol_threshold, signal = 1 (expecting increased volatility).
           - If vol_change < -vol_threshold, signal = -1 (expecting decreased volatility).
           - Otherwise, signal = 0.
       
    7. Signal strength is quantified as the normalized absolute volatility change:
           signal_strength = |vol_change| / (rolling_std(vol_change) + 1e-6)
    
    8. Risk Management:
       The raw generated signals are passed to a RiskManager component that adjusts for slippage, 
       transaction costs, and applies stop-loss/take-profit rules. It then computes risk–managed 
       returns and cumulative performance.
    
    This strategy supports backtesting via start and end dates, and it can process multiple tickers 
    in a vectorized fashion.
    
    Args:
        db_config (DatabaseConfig): Database configuration settings.
        params (dict, optional): Strategy parameters. Supported keys include:
            - 'window_size' (int): Rolling window size for GARCH model fitting (default: 100).
            - 'forecast_horizon' (int): Forecast horizon (default: 1).
            - 'vol_threshold' (float): Threshold for volatility change to trigger signals (default: 0.1).
            - 'p' (int): ARCH model order (default: 1).
            - 'q' (int): GARCH model order (default: 1).
            - 'return_type' (str): 'log' or 'simple' returns (default: 'log').
            - 'stop_loss_pct' (float): Stop loss percentage (default: 0.05).
            - 'take_profit_pct' (float): Take profit percentage (default: 0.10).
            - 'slippage_pct' (float): Slippage percentage (default: 0.001).
            - 'transaction_cost_pct' (float): Transaction cost percentage (default: 0.001).
    
    Output:
        A pandas DataFrame containing:
            - 'open', 'close', 'high', 'low', 'volume': Price data.
            - 'returns': Computed daily returns.
            - 'forecast_volatility': Forecasted volatility from the GARCH model.
            - 'volatility_change': Relative change in volatility.
            - 'signal': Generated trading signal (1, -1, or 0).
            - 'signal_strength': Normalized strength of the signal.
            - 'position': Adjusted trading position after risk management.
            - 'rm_strategy_return': Realized return following risk management.
            - 'rm_cumulative_return': Cumulative risk–managed return.
            - 'rm_action': Risk management action (e.g. stop_loss, take_profit, signal_exit).
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        """
        Initialize the GARCH Model strategy with given parameters.
        
        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Dictionary of strategy-specific parameters.
        """
        default_params = {
            'window_size': 100,
            'forecast_horizon': 1,
            'vol_threshold': 0.1,
            'p': 1,
            'q': 1,
            'return_type': 'log',
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001
        }
        if params:
            default_params.update(params)
        super().__init__(db_config, default_params)
    
    def generate_signals(self,
                         tickers: Union[str, List[str]],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         initial_position: int = 0,
                         latest_only: bool = False) -> pd.DataFrame:
        """
        Generate trading signals based on GARCH volatility forecasts and apply risk management.
        
        This method retrieves historical price data, computes returns, and for each ticker fits a GARCH(1,1) 
        model in a rolling window to forecast volatility. The relative change in volatility (forecasted vs. 
        historical) is compared to a threshold to generate a signal. A normalized signal strength is computed, 
        and then risk management (incorporating slippage, transaction costs, stop-loss and take-profit) is applied.
        
        Args:
            tickers (str or List[str]): Stock ticker symbol or list of tickers.
            start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format. An extra data buffer 
                                        is retrieved prior to start_date for model calibration.
            end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
            initial_position (int): Starting trading position (default: 0).
            latest_only (bool): If True, returns only the latest generated signal for each ticker.
        
        Returns:
            pd.DataFrame: DataFrame containing price data, forecasted metrics, generated signals, signal strength,
                          and risk-managed performance metrics.
        """
        if isinstance(tickers, str):
            tickers_list = [tickers]
        else:
            tickers_list = tickers
        
        results = []
        for tic in tickers_list:
            # Determine data retrieval parameters.
            window_size = self.params['window_size']
            if start_date is not None:
                # Fetch additional historical data (buffer) for rolling window calculation.
                start_date_buffer = (pd.to_datetime(start_date) - pd.Timedelta(days=window_size * 3)).strftime('%Y-%m-%d')
                price_data = self.get_historical_prices(tic, from_date=start_date_buffer, to_date=end_date)
            else:
                price_data = self.get_historical_prices(tic, lookback=500)
            
            # Process the ticker's data only if sufficient historical data exists.
            if price_data.empty or not self._validate_data(price_data, min_records=window_size + 50):
                self.logger.warning(f"Insufficient data for {tic} to generate GARCH model signals")
                continue
            
            # If data for multiple tickers is returned as a MultiIndex, select current ticker's data.
            if isinstance(price_data.index, pd.MultiIndex):
                try:
                    price_data_tic = price_data.xs(tic, level='ticker')
                except KeyError:
                    self.logger.warning(f"No data found for ticker {tic}")
                    continue
            else:
                price_data_tic = price_data
            
            # Generate signals for the current ticker using a helper method.
            signal_df = self._generate_signal_for_ticker(tic, price_data_tic, initial_position)
            
            # Restrict to the backtest period if start_date is provided.
            if start_date is not None:
                signal_df = signal_df.loc[signal_df.index >= pd.to_datetime(start_date)]
            
            # If latest_only is True, keep only the last generated signal for this ticker.
            if latest_only and not signal_df.empty:
                signal_df = signal_df.iloc[[-1]]
            
            # Add a column for the ticker identifier.
            signal_df['ticker'] = tic
            results.append(signal_df)
        
        # Combine results from all tickers.
        if results:
            final_result = pd.concat(results)
            final_result.sort_index(inplace=True)
            return final_result
        else:
            return pd.DataFrame()
    
    def _generate_signal_for_ticker(self, ticker: str, price_data: pd.DataFrame, initial_position: int) -> pd.DataFrame:
        """
        Compute returns, forecast volatility using a rolling GARCH model, generate trading signals,
        and apply risk management for a single ticker.
        
        Args:
            ticker (str): The stock ticker symbol.
            price_data (pd.DataFrame): Historical pricing data with columns ['open', 'high', 'low', 'close', 'volume'].
            initial_position (int): Starting trading position.
        
        Returns:
            pd.DataFrame: DataFrame containing the computed returns, forecasted volatility, volatility change, 
                          generated signal, signal strength, and columns output by the risk management process.
        """
        # Compute returns based on the chosen method.
        if self.params['return_type'] == 'log':
            returns = np.log(price_data['close'] / price_data['close'].shift(1)) * 100
        else:
            returns = (price_data['close'] / price_data['close'].shift(1) - 1) * 100
        
        # Initialize a DataFrame to store computed metrics.
        result = price_data.copy()
        result['returns'] = returns
        result['forecast_volatility'] = np.nan
        result['volatility_change'] = np.nan
        result['signal'] = 0
        
        # Determine the starting index that contains valid returns.
        start_idx = returns.first_valid_index()
        if start_idx is None:
            self.logger.warning(f"No valid returns computed for ticker {ticker}")
            return pd.DataFrame()
        
        window_size = self.params['window_size']
        
        # Rolling window: For each point where there is enough history, fit the GARCH model.
        for i in range(window_size, len(returns)):
            window_returns = returns.iloc[i - window_size:i].dropna()
            if len(window_returns) < window_size * 0.9:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    garch_model = arch_model(window_returns, vol='Garch', p=self.params['p'], q=self.params['q'], rescale=False)
                    garch_fit = garch_model.fit(disp='off', show_warning=False)
                    # Forecast the next period variance and compute volatility.
                    forecast = garch_fit.forecast(horizon=self.params['forecast_horizon'])
                    forecast_variance = forecast.variance.iloc[-1, 0]
                    forecast_vol = np.sqrt(forecast_variance)
            except Exception as e:
                self.logger.warning(f"GARCH model fitting failed for {ticker} at index {result.index[i]}: {str(e)}")
                continue
            
            # Store the forecasted volatility.
            result.iloc[i, result.columns.get_loc('forecast_volatility')] = forecast_vol
            
            # Compute historical volatility (annualized) from window returns.
            hist_vol = window_returns.std() * np.sqrt(252)
            vol_change = (forecast_vol - hist_vol) / hist_vol if hist_vol != 0 else np.nan
            result.iloc[i, result.columns.get_loc('volatility_change')] = vol_change
            
            # Generate trading signal based on the volatility change.
            if vol_change > self.params['vol_threshold']:
                result.iloc[i, result.columns.get_loc('signal')] = 1
            elif vol_change < -self.params['vol_threshold']:
                result.iloc[i, result.columns.get_loc('signal')] = -1
        
        # Compute signal strength as the normalized absolute volatility change.
        result['signal_strength'] = (result['volatility_change'].abs() /
                                     (result['volatility_change'].rolling(window=window_size, min_periods=1).std() + 1e-6))
        
        # Drop rows that never received a forecast (and hence lacking signals).
        result = result.dropna(subset=['forecast_volatility'])
        
        # Apply risk management adjustments via the RiskManager.
        risk_manager = RiskManager(
            stop_loss_pct=self.params.get('stop_loss_pct', 0.05),
            take_profit_pct=self.params.get('take_profit_pct', 0.10),
            slippage_pct=self.params.get('slippage_pct', 0.001),
            transaction_cost_pct=self.params.get('transaction_cost_pct', 0.001)
        )
        result = risk_manager.apply(result, initial_position=initial_position)
        
        # Rename risk-managed return columns for clarity.
        result = result.rename(columns={
            'return': 'rm_strategy_return',
            'cumulative_return': 'rm_cumulative_return',
            'exit_type': 'rm_action'
        })
        
        return result