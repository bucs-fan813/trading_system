# trading_system/src/strategies/pairs_trading.py

"""
Pairs Trading Strategy with Integrated Risk Management

This strategy identifies and trades statistically-related pairs of securities based on cointegration.
When the spread between pair components deviates significantly from its mean, the strategy
takes positions with the expectation that the relationship will revert to its historical norm.

Key mathematical components:
- Two securities X and Y are cointegrated if Z = X - βY is stationary (mean-reverting)
- β is the hedge ratio determining how many units of Y to trade against X
- Z-score of the spread determines entry/exit points
- Half-life of mean reversion provides measure of reversion speed

The strategy follows these steps:
1. Identify potential pairs using correlation as initial filter
2. Test for cointegration to confirm statistical relationship
3. Calculate hedge ratio and generate spread series
4. Normalize spread using z-scores
5. Generate trading signals based on z-score thresholds
6. Apply risk management including stop-loss and take-profit

Trading signals:
- Long spread (buy X, sell Y): When z-score falls below negative threshold
- Short spread (sell X, buy Y): When z-score rises above positive threshold
- Exit positions: When z-score reverts toward mean (crosses exit threshold)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from itertools import combinations
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from time import perf_counter

from src.database.config import DatabaseConfig
from src.strategies.base_strat import BaseStrategy
from src.strategies.risk_management import RiskManager


class PairsTrading(BaseStrategy):
    """
    A pairs trading strategy that identifies and trades cointegrated securities.
    
    Attributes:
        db_config (DatabaseConfig): Database configuration settings.
        params (dict): Strategy-specific parameters including:
            - 'lookback_window': Period for calculating rolling statistics (default: 60)
            - 'entry_threshold': Z-score threshold for entering positions (default: 2.0)
            - 'exit_threshold': Z-score threshold for exiting positions (default: 0.5)
            - 'correlation_threshold': Minimum correlation to consider a pair (default: 0.7)
            - 'halflife_threshold': Maximum half-life for mean reversion (default: 30)
            - 'pvalue_threshold': Maximum p-value for cointegration test (default: 0.05)
            - 'min_data_points': Minimum data points required (default: 252)
            - 'stop_loss_pct': Stop loss percentage (default: 0.05)
            - 'take_profit_pct': Take profit percentage (default: 0.10)
            - 'slippage_pct': Slippage percentage (default: 0.001)
            - 'transaction_cost_pct': Transaction cost percentage (default: 0.001)
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the pairs trading strategy.
        
        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Strategy-specific parameters.
        """
        # Set default parameters
        default_params = {
            'lookback_window': 60,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'correlation_threshold': 0.7,
            'halflife_threshold': 30,
            'pvalue_threshold': 0.05,
            'min_data_points': 252,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001
        }
        
        # Merge provided params with defaults
        if params:
            default_params.update(params)
        
        super().__init__(db_config, default_params)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            stop_loss_pct=self.params['stop_loss_pct'],
            take_profit_pct=self.params['take_profit_pct'],
            slippage_pct=self.params['slippage_pct'],
            transaction_cost_pct=self.params['transaction_cost_pct']
        )
    
    def find_cointegrated_pairs(self, price_data: pd.DataFrame, tickers: List[str]) -> List[Tuple[str, str, float, float, float]]:
        """
        Find cointegrated pairs from a list of tickers.
        
        Args:
            price_data (pd.DataFrame): Price data for all tickers.
            tickers (List[str]): List of ticker symbols.
            
        Returns:
            List[Tuple[str, str, float, float, float]]: List of tuples containing
                ticker1, ticker2, hedge_ratio, p-value, and half-life.
        """
        t0 = perf_counter()
        cointegrated_pairs = []
        
        # Reshape multi-index DataFrame to have tickers as columns
        if isinstance(price_data.index, pd.MultiIndex):
            # Pivot the DataFrame to have dates as index and tickers as columns
            close_prices = price_data['close'].unstack(level=0)
        else:
            # Assume it's already in the correct format
            close_prices = price_data['close']
        
        # Ensure we have data for all requested tickers
        available_tickers = [ticker for ticker in tickers if ticker in close_prices.columns]
        if len(available_tickers) < len(tickers):
            missing = set(tickers) - set(available_tickers)
            self.logger.warning(f"Missing price data for tickers: {missing}")
        
        # Fill missing values with previous values
        close_prices = close_prices[available_tickers].fillna(method='ffill')
        
        # Minimum required data points
        min_points = self.params['min_data_points']
        if len(close_prices) < min_points:
            self.logger.warning(f"Insufficient data points: {len(close_prices)} < {min_points}")
            return []
        
        # Calculate correlation matrix for preliminary filtering
        correlation_matrix = close_prices.corr()
        
        # Test each potential pair for cointegration
        pairs_tested = 0
        for ticker1, ticker2 in combinations(available_tickers, 2):
            # Skip pairs with low correlation
            if correlation_matrix.loc[ticker1, ticker2] < self.params['correlation_threshold']:
                continue
            
            pairs_tested += 1
            
            # Extract price series for the pair
            stock1 = close_prices[ticker1].dropna()
            stock2 = close_prices[ticker2].dropna()
            
            # Ensure both series have the same length and dates
            common_idx = stock1.index.intersection(stock2.index)
            if len(common_idx) < min_points:  # Minimum data requirement
                continue
                
            stock1 = stock1.loc[common_idx]
            stock2 = stock2.loc[common_idx]
            
            # Perform cointegration test
            result = coint(stock1, stock2)
            pvalue = result[1]
            
            if pvalue < self.params['pvalue_threshold']:
                # Calculate hedge ratio using OLS
                model = sm.OLS(stock1, sm.add_constant(stock2)).fit()
                hedge_ratio = model.params[1]
                
                # Calculate spread
                spread = stock1 - hedge_ratio * stock2
                
                # Calculate half-life of mean reversion
                half_life = self._calculate_half_life(spread)
                
                # Only include pairs with acceptable half-life
                if 0 < half_life < self.params['halflife_threshold']:
                    cointegrated_pairs.append((ticker1, ticker2, hedge_ratio, pvalue, half_life))
        
        # Sort pairs by p-value (most significant first)
        cointegrated_pairs.sort(key=lambda x: x[3])
        
        t1 = perf_counter()
        self.logger.info(f"Tested {pairs_tested} pairs, found {len(cointegrated_pairs)} cointegrated pairs in {t1-t0:.2f} seconds")
        
        return cointegrated_pairs
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate the half-life of mean reversion for a spread series.
        
        Args:
            spread (pd.Series): The spread between two assets.
            
        Returns:
            float: Half-life value in days.
        """
        try:
            # Clean the spread data
            spread = spread.dropna()
            
            # Check if we have enough data
            if len(spread) < 20:  # Minimum length for reliable estimation
                return float('inf')
            
            # Calculate lagged spread
            spread_lag = spread.shift(1)
            spread_ret = spread - spread_lag
            spread_ret = spread_ret.dropna()
            spread_lag = spread_lag.dropna()
            
            # Run OLS regression: spread return = gamma * lagged_spread + constant
            model = sm.OLS(spread_ret, sm.add_constant(spread_lag)).fit()
            
            # Extract gamma from the model
            gamma = model.params[1]
            
            # If gamma is positive, the series is not mean-reverting
            if gamma >= 0:
                return float('inf')
            
            # Calculate half-life: ln(2) / abs(gamma)
            half_life = -np.log(2) / gamma
            
            return half_life
            
        except Exception as e:
            self.logger.error(f"Error calculating half-life: {str(e)}")
            return float('inf')
    
    def generate_pair_signals(
        self,
        ticker1: str,
        ticker2: str,
        price_data: pd.DataFrame,
        hedge_ratio: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate trading signals for a specific pair.
        
        Args:
            ticker1 (str): First ticker in the pair.
            ticker2 (str): Second ticker in the pair.
            price_data (pd.DataFrame): Price data for the pair.
            hedge_ratio (float, optional): Pre-calculated hedge ratio. If None, will be calculated.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            initial_position (int): Starting position (0, 1, or -1).
            latest_only (bool): If True, returns only the most recent signal.
            
        Returns:
            pd.DataFrame: DataFrame with signals and relevant data.
        """
        t0 = perf_counter()
        
        # Restructure data for easier processing
        if isinstance(price_data.index, pd.MultiIndex):
            # Extract data for each ticker and align on common dates
            df1 = price_data.loc[ticker1].copy()
            df2 = price_data.loc[ticker2].copy()
            
            # Ensure both DataFrames have the same dates
            common_dates = df1.index.intersection(df2.index)
            df1 = df1.loc[common_dates]
            df2 = df2.loc[common_dates]
            
            # Create a merged DataFrame
            merged_data = pd.DataFrame({
                f'{ticker1}_open': df1['open'],
                f'{ticker1}_high': df1['high'],
                f'{ticker1}_low': df1['low'],
                f'{ticker1}_close': df1['close'],
                f'{ticker2}_open': df2['open'],
                f'{ticker2}_high': df2['high'],
                f'{ticker2}_low': df2['low'],
                f'{ticker2}_close': df2['close']
            }, index=common_dates)
        else:
            # If already in appropriate format
            merged_data = price_data
        
        # Apply date filtering if specified
        if start_date:
            merged_data = merged_data[merged_data.index >= start_date]
        if end_date:
            merged_data = merged_data[merged_data.index <= end_date]
        
        # Check if we have sufficient data
        if merged_data.empty or len(merged_data) < self.params['lookback_window']:
            self.logger.warning(f"Insufficient data for pair {ticker1}-{ticker2}")
            return pd.DataFrame()
        
        # Calculate hedge ratio if not provided
        if hedge_ratio is None:
            # Use a portion of the data to calculate hedge ratio
            training_size = min(len(merged_data) // 2, self.params['min_data_points'])
            training_data = merged_data.iloc[:training_size]
            
            # Run OLS regression
            y = training_data[f'{ticker1}_close']
            X = sm.add_constant(training_data[f'{ticker2}_close'])
            model = sm.OLS(y, X).fit()
            hedge_ratio = model.params[1]
        
        # Calculate spread
        merged_data['spread'] = merged_data[f'{ticker1}_close'] - hedge_ratio * merged_data[f'{ticker2}_close']
        
        # Calculate rolling statistics for z-score
        lookback_window = self.params['lookback_window']
        merged_data['spread_mean'] = merged_data['spread'].rolling(window=lookback_window).mean()
        merged_data['spread_std'] = merged_data['spread'].rolling(window=lookback_window).std()
        
        # Calculate z-score with error handling for zero standard deviation
        merged_data['zscore'] = np.nan
        valid_std = merged_data['spread_std'] > 0
        merged_data.loc[valid_std, 'zscore'] = (
            (merged_data.loc[valid_std, 'spread'] - merged_data.loc[valid_std, 'spread_mean']) / 
            merged_data.loc[valid_std, 'spread_std']
        )
        
        # Generate signals based on z-score
        entry_threshold = self.params['entry_threshold']
        exit_threshold = self.params['exit_threshold']
        
        # Initialize signal column
        merged_data['signal'] = 0
        
        # Entry signals
        merged_data.loc[merged_data['zscore'] < -entry_threshold, 'signal'] = 1  # Long spread
        merged_data.loc[merged_data['zscore'] > entry_threshold, 'signal'] = -1  # Short spread
        
        # Exit signals (when z-score crosses back)
        long_exit = (merged_data['zscore'] > -exit_threshold) & (merged_data['zscore'].shift(1) <= -exit_threshold)
        short_exit = (merged_data['zscore'] < exit_threshold) & (merged_data['zscore'].shift(1) >= exit_threshold)
        merged_data.loc[long_exit | short_exit, 'signal'] = 0
        
        # Calculate signal strength (normalized by the standard deviation of z-scores)
        z_std = merged_data['zscore'].rolling(window=lookback_window).std().fillna(1)
        merged_data['signal_strength'] = merged_data['zscore'].abs() / (z_std + 1e-6)
        
        # Prepare data for risk management
        risk_data = pd.DataFrame({
            'open': merged_data[f'{ticker1}_open'],
            'high': merged_data[f'{ticker1}_high'],
            'low': merged_data[f'{ticker1}_low'],
            'close': merged_data[f'{ticker1}_close'],
            'signal': merged_data['signal']
        })
        
        # Apply risk management
        risk_managed = self.risk_manager.apply(risk_data, initial_position)
        
        # Merge risk management results back
        result = pd.DataFrame({
            'open': merged_data[f'{ticker1}_open'],
            'high': merged_data[f'{ticker1}_high'],
            'low': merged_data[f'{ticker1}_low'],
            'close': merged_data[f'{ticker1}_close'],
            'spread': merged_data['spread'],
            'zscore': merged_data['zscore'],
            'signal': merged_data['signal'],
            'signal_strength': merged_data['signal_strength'],
            'position': risk_managed['position'],
            'return': risk_managed['return'],
            'cumulative_return': risk_managed['cumulative_return'],
            'exit_type': risk_managed['exit_type']
        })
        
        # Add ticker information
        result['ticker1'] = ticker1
        result['ticker2'] = ticker2
        result['hedge_ratio'] = hedge_ratio
        
        # Return only latest signal if requested
        if latest_only:
            result = result.iloc[-1:].copy()
        
        t1 = perf_counter()
        self.logger.debug(f"Generated pair signals in {t1-t0:.2f} seconds")
        
        return result
    
    def generate_signals(
        self, 
        tickers: Union[str, List[str], Tuple[str, str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate trading signals for pairs trading.
        
        Args:
            tickers (Union[str, List[str], Tuple[str, str]]): Ticker symbols to analyze.
                If a list with more than 2 tickers, will find the best cointegrated pair.
                If a tuple of exactly 2 tickers, will use those as the pair.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format.
            end_date (str, optional): End date in 'YYYY-MM-DD' format.
            initial_position (int): Starting position (0, 1, or -1).
            latest_only (bool): If True, returns only the most recent signal.
            
        Returns:
            pd.DataFrame: DataFrame with signals and relevant data.
        """
        t0 = perf_counter()
        
        # Determine if we're working with a specific pair or need to find one
        if isinstance(tickers, tuple) and len(tickers) == 2:
            # Use the provided pair
            ticker1, ticker2 = tickers
            pairs_to_process = [(ticker1, ticker2, None)]  # None for hedge_ratio will be calculated later
            search_for_pairs = False
        elif isinstance(tickers, list) and len(tickers) >= 2:
            # Find the best cointegrated pair from the list
            self.logger.info(f"Finding cointegrated pairs among {len(tickers)} tickers")
            
            # Get historical data for all tickers
            lookback = max(self.params['lookback_window'] * 3, self.params['min_data_points']) if latest_only else None
            price_data = self.get_historical_prices(tickers, lookback, from_date=start_date, to_date=end_date)
            
            # Find cointegrated pairs
            cointegrated_pairs = self.find_cointegrated_pairs(price_data, tickers)
            
            if not cointegrated_pairs:
                self.logger.warning("No cointegrated pairs found")
                # Return empty DataFrame with expected columns
                columns = ['open', 'high', 'low', 'close', 'spread', 'zscore', 
                           'signal', 'signal_strength', 'position', 'return', 
                           'cumulative_return', 'exit_type', 'ticker1', 'ticker2', 'hedge_ratio']
                return pd.DataFrame(columns=columns)
            
            # Use the top cointegrated pairs (lowest p-value)
            pairs_to_process = [(pair[0], pair[1], pair[2]) for pair in cointegrated_pairs[:1]]
            search_for_pairs = True
        elif isinstance(tickers, str):
            # Single ticker provided, which doesn't work for pairs trading
            raise ValueError("Pairs trading requires at least two tickers")
        else:
            raise ValueError("Invalid ticker format")
        
        # Process each selected pair
        result_dfs = []
        
        for ticker1, ticker2, hedge_ratio in pairs_to_process:
            self.logger.info(f"Processing pair: {ticker1} - {ticker2}")
            
            # Get historical data for the pair if not already retrieved
            if search_for_pairs:
                # We already have the data from the find_cointegrated_pairs call
                price_data_pair = price_data
            else:
                # Need to retrieve data for the specific pair
                lookback = max(self.params['lookback_window'] * 3, self.params['min_data_points']) if latest_only else None
                price_data_pair = self.get_historical_prices([ticker1, ticker2], lookback, from_date=start_date, to_date=end_date)
            
            # Generate signals for this pair
            pair_result = self.generate_pair_signals(
                ticker1=ticker1,
                ticker2=ticker2,
                price_data=price_data_pair,
                hedge_ratio=hedge_ratio,
                start_date=start_date,
                end_date=end_date,
                initial_position=initial_position,
                latest_only=latest_only
            )
            
            if not pair_result.empty:
                result_dfs.append(pair_result)
        
        if not result_dfs:
            self.logger.warning("No valid results generated")
            # Return empty DataFrame with expected columns
            columns = ['open', 'high', 'low', 'close', 'spread', 'zscore', 
                       'signal', 'signal_strength', 'position', 'return', 
                       'cumulative_return', 'exit_type', 'ticker1', 'ticker2', 'hedge_ratio']
            return pd.DataFrame(columns=columns)
        
        # Combine results if multiple pairs were processed
        final_result = pd.concat(result_dfs) if len(result_dfs) > 1 else result_dfs[0]
        
        t1 = perf_counter()
        self.logger.info(f"Signal generation completed in {t1-t0:.2f} seconds")
        
        return final_result