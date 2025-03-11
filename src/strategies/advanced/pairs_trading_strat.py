# trading_system/src/strategies/pairs_trading.py

"""
Pairs Trading Strategy with Integrated Risk Management.

This strategy implements statistical arbitrage based on pairs trading principles,
identifying cointegrated pairs of securities and generating signals when their
relative value spread deviates significantly from its mean.

The implementation follows these key steps:
1. Data Collection: Retrieves historical price data for multiple tickers.
2. Pair Identification: Filters for correlated pairs and tests for cointegration.
3. Spread Calculation: Computes the spread using hedge ratios from regression.
4. Signal Generation: Creates trading signals based on z-score thresholds.
5. Risk Management: Applies stop-loss and take-profit rules with transaction costs.

The strategy is fully vectorized for performance and supports both backtesting
and latest-only signal generation for immediate trading decisions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from itertools import combinations
from time import perf_counter

from src.strategies.base_strat import BaseStrategy
from src.strategies.risk_management import RiskManager
from src.database.config import DatabaseConfig


class PairsTrading(BaseStrategy):
    """
    Pairs Trading Strategy that identifies and trades cointegrated pairs of securities.
    
    Attributes:
        db_config (DatabaseConfig): Database configuration for data retrieval.
        params (dict): Strategy parameters including correlation thresholds, 
                       z-score entry/exit levels, and risk management settings.
    """
    
    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Pairs Trading strategy with database configuration and parameters.
        
        Args:
            db_config (DatabaseConfig): Configuration for database access.
            params (dict, optional): Strategy parameters. Defaults are used if None.
        """
        default_params = {
            'lookback_period': 252,          # One trading year of data
            'correlation_threshold': 0.7,    # Minimum correlation for pair consideration
            'zscore_entry_threshold': 2.0,   # Z-score threshold for trade entry
            'zscore_exit_threshold': 0.5,    # Z-score threshold for trade exit
            'minimum_half_life': 1,          # Minimum half-life for mean reversion in days
            'maximum_half_life': 30,         # Maximum half-life for mean reversion in days
            'rolling_window': 30,            # Window for rolling statistics
            'coint_pvalue_threshold': 0.05,  # P-value threshold for cointegration test
            'stop_loss_pct': 0.05,           # Stop loss percentage
            'take_profit_pct': 0.10,         # Take profit percentage
            'slippage_pct': 0.001,           # Slippage percentage
            'transaction_cost_pct': 0.001,   # Transaction cost percentage
            'data_source': 'yfinance'
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(db_config, default_params)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize RiskManager with the specified risk management parameters
        self.risk_manager = RiskManager(
            stop_loss_pct=self.params['stop_loss_pct'],
            take_profit_pct=self.params['take_profit_pct'],
            slippage_pct=self.params['slippage_pct'],
            transaction_cost_pct=self.params['transaction_cost_pct']
        )

    def find_cointegrated_pairs(self, price_data: pd.DataFrame, tickers: List[str]) -> List[Tuple[str, str, float, float, float]]:
        """
        Find cointegrated pairs from the given price data using statistical tests.
        
        Args:
            price_data (pd.DataFrame): DataFrame of closing prices indexed by date and
                                       with ticker symbols as the columns.
            tickers (List[str]): List of ticker symbols to consider for pairs.
            
        Returns:
            List[Tuple[str, str, float, float, float]]: A list of tuples containing:
                - ticker1: First ticker in the pair.
                - ticker2: Second ticker in the pair.
                - hedge_ratio: Calculated hedge ratio between the pair.
                - half_life: Half-life of mean reversion (in days).
                - pvalue: P-value from the cointegration test.
        """
        self.logger.info("Finding cointegrated pairs from %d tickers", len(tickers))
        t0 = perf_counter()
        
        # Calculate correlation matrix (absolute correlation)
        correlation_matrix = price_data.corr().abs()
        
        cointegrated_pairs = []
        pairs = list(combinations(tickers, 2))
        self.logger.info("Evaluating %d potential pairs", len(pairs))
        
        for ticker1, ticker2 in pairs:
            # Skip pairs that do not meet the correlation threshold
            corr = correlation_matrix.loc[ticker1, ticker2]
            if corr < self.params['correlation_threshold']:
                continue
            
            price1 = price_data[ticker1].dropna()
            price2 = price_data[ticker2].dropna()
            
            if len(price1) < 30 or len(price2) < 30:
                continue
            
            # Align the two series on common dates
            common_data = pd.concat([price1, price2], axis=1).dropna()
            if len(common_data) < 30:
                continue
                
            price1 = common_data[ticker1]
            price2 = common_data[ticker2]
            
            # Perform cointegration test
            coint_result = coint(price1, price2)
            pvalue = coint_result[1]
            
            if pvalue < self.params['coint_pvalue_threshold']:
                # Estimate hedge ratio using OLS regression (with intercept)
                model = sm.OLS(price1, sm.add_constant(price2)).fit()
                hedge_ratio = model.params[1]
                
                # Calculate the spread and its half-life
                spread = price1 - hedge_ratio * price2
                half_life = self.calculate_half_life(spread)
                
                if (self.params['minimum_half_life'] <= half_life <= self.params['maximum_half_life']):
                    cointegrated_pairs.append((ticker1, ticker2, hedge_ratio, half_life, pvalue))
                    
        t1 = perf_counter()
        self.logger.info("Found %d cointegrated pairs in %.2f seconds", len(cointegrated_pairs), t1 - t0)
        
        # Sort pairs by half-life (ascending) to prioritize reversion speed
        cointegrated_pairs.sort(key=lambda x: x[3])
        return cointegrated_pairs

    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate the half-life of mean reversion for a given spread series.
        
        Args:
            spread (pd.Series): Time series of the spread between two assets.
            
        Returns:
            float: Estimated half-life of mean reversion in days. Returns infinity if no mean reversion.
        """
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        spread_lag = spread_lag.dropna()
        spread_diff = spread_diff.dropna()
        
        model = sm.OLS(spread_diff, sm.add_constant(spread_lag)).fit()
        gamma = model.params[1]
        
        if gamma < 0:
            half_life = -np.log(2) / gamma
            return half_life
        else:
            return float('inf')

    def compute_zscore(self, spread: pd.Series, window: int) -> pd.Series:
        """
        Compute the z-score of a spread using a rolling window.
        
        Args:
            spread (pd.Series): Time series of the spread.
            window (int): Rolling window size for calculating mean and standard deviation.
            
        Returns:
            pd.Series: Z-score of the spread.
        """
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        z_score = (spread - rolling_mean) / (rolling_std + 1e-9)
        return z_score

    def generate_pair_signals(
        self,
        price_data1: pd.DataFrame,
        price_data2: pd.DataFrame,
        hedge_ratio: float,
        rolling_window: int,
        zscore_entry: float,
        zscore_exit: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate trading signals for a pair based on the z-score of their spread.
        
        Args:
            price_data1 (pd.DataFrame): OHLC price data for the first asset.
            price_data2 (pd.DataFrame): OHLC price data for the second asset.
            hedge_ratio (float): Hedge ratio between the two assets.
            rolling_window (int): Rolling window size for the z-score calculation.
            zscore_entry (float): Z-score threshold for entering positions.
            zscore_exit (float): Z-score threshold for exiting positions.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames with trading signals for each asset.
                Each DataFrame includes the spread, z-score, raw signal, signal strength, daily returns,
                and preliminary strategy returns.
        """
        # Compute the spread using the closing prices
        spread = price_data1['close'] - hedge_ratio * price_data2['close']
        
        # Compute the rolling z-score
        zscore = self.compute_zscore(spread, rolling_window)
        
        signals1 = price_data1.copy()
        signals2 = price_data2.copy()
        
        signals1['spread'] = spread
        signals1['zscore'] = zscore
        signals2['spread'] = spread
        signals2['zscore'] = zscore
        
        # Initialize signal columns (0 = no position)
        signals1['signal'] = 0
        signals2['signal'] = 0
        
        # Generate entry signals:
        # For asset1: a long signal when z-score is below -entry threshold, short when above +entry threshold.
        signals1.loc[zscore < -zscore_entry, 'signal'] = 1
        signals1.loc[zscore > zscore_entry, 'signal'] = -1
        # For asset2, the positions are reversed.
        signals2.loc[zscore < -zscore_entry, 'signal'] = -1
        signals2.loc[zscore > zscore_entry, 'signal'] = 1
        
        # Generate exit signals when the z-score reverts
        signals1.loc[(zscore > -zscore_exit) & (zscore < 0) & (signals1['signal'].shift(1) == 1), 'signal'] = 0
        signals1.loc[(zscore < zscore_exit) & (zscore > 0) & (signals1['signal'].shift(1) == -1), 'signal'] = 0
        signals2.loc[(zscore > -zscore_exit) & (zscore < 0) & (signals2['signal'].shift(1) == -1), 'signal'] = 0
        signals2.loc[(zscore < zscore_exit) & (zscore > 0) & (signals2['signal'].shift(1) == 1), 'signal'] = 0
        
        # Signal strength (normalized absolute z-score)
        signals1['signal_strength'] = np.abs(zscore) / (np.std(zscore.dropna()) + 1e-6)
        signals2['signal_strength'] = np.abs(zscore) / (np.std(zscore.dropna()) + 1e-6)
        
        # Daily returns based on closing prices
        signals1['daily_return'] = signals1['close'].pct_change()
        signals2['daily_return'] = signals2['close'].pct_change()
        
        # Preliminary strategy return (position taken from previous day)
        signals1['strategy_return'] = signals1['daily_return'] * signals1['signal'].shift(1)
        signals2['strategy_return'] = signals2['daily_return'] * signals2['signal'].shift(1)
        
        # Forward fill signals for the backtest
        signals1['signal'] = signals1['signal'].replace(0, np.nan).ffill().fillna(0)
        signals2['signal'] = signals2['signal'].replace(0, np.nan).ffill().fillna(0)
        
        return signals1, signals2

    def generate_signals(
        self,
        tickers: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate trading signals for a list of tickers using the pairs trading strategy.
        
        Args:
            tickers (Union[str, List[str]]): Single ticker or list of tickers.
            start_date (str, optional): Start date for the analysis in 'YYYY-MM-DD' format.
            end_date (str, optional): End date for the analysis in 'YYYY-MM-DD' format.
            initial_position (int): Initial trading position (0, 1, or -1).
            latest_only (bool): If True, returns only the latest signal for each pair.
            
        Returns:
            pd.DataFrame: A DataFrame with backtest results, including synthetic signals,
                          risk-managed positions, trade returns, and metadata (pair tickers,
                          hedge ratio, half-life, cointegration p-value). In multi-pair settings,
                          a multi-index is used.
        """
        # Ensure tickers is a list (if a single ticker is provided, raise an error)
        if isinstance(tickers, str):
            tickers = [tickers]
            single_ticker = True
        else:
            single_ticker = False
        
        if len(tickers) < 2:
            raise ValueError("Pairs trading requires at least 2 tickers")
        
        # Determine the required lookback period and data retrieval parameters
        lookback_period = max(
            self.params['lookback_period'],
            self.params['rolling_window'] * 3  # Ensure sufficient history for rolling calculations
        )
        
        if latest_only and not start_date:
            # Retrieve minimal data based on lookback
            price_data = self.get_historical_prices(
                tickers=tickers,
                lookback=lookback_period,
                data_source=self.params.get("data_source", "yfinance")
            )
        else:
            price_data = self.get_historical_prices(
                tickers=tickers,
                from_date=start_date,
                to_date=end_date,
                data_source=self.params.get("data_source", "yfinance")
            )
        
        if price_data.empty:
            self.logger.error("No historical price data retrieved")
            return pd.DataFrame()
        
        # Pivot the closing prices to create a DataFrame with dates as index and tickers as columns
        df_close = price_data["close"].unstack(level=0)
        
        # Identify cointegrated pairs
        cointegrated_pairs = self.find_cointegrated_pairs(df_close, tickers)
        if not cointegrated_pairs:
            self.logger.info("No valid cointegrated pairs were found")
            return pd.DataFrame()

        results = []

        # Process each cointegrated pair
        for ticker1, ticker2, hedge_ratio, half_life, pvalue in cointegrated_pairs:
            self.logger.info("Processing pair %s & %s: hedge_ratio=%.4f, half_life=%.2f, p_value=%.4f",
                             ticker1, ticker2, hedge_ratio, half_life, pvalue)
            try:
                # Retrieve OHLC data for each ticker
                price_data1 = price_data.xs(ticker1, level=0).sort_index()
                price_data2 = price_data.xs(ticker2, level=0).sort_index()
            except Exception as e:
                self.logger.error("Error retrieving price data for pair %s & %s: %s", ticker1, ticker2, str(e))
                continue

            # Generate raw signals for the pair (each asset receives mirror-image signals)
            signals1, signals2 = self.generate_pair_signals(
                price_data1,
                price_data2,
                hedge_ratio=hedge_ratio,
                rolling_window=self.params['rolling_window'],
                zscore_entry=self.params['zscore_entry_threshold'],
                zscore_exit=self.params['zscore_exit_threshold']
            )
            
            # Apply risk management to the first asset signal for this pair.
            # (Since the positions are mirror images, risk management on one leg suffices.)
            risk_managed = self.risk_manager.apply(signals1, initial_position=initial_position)
            
            # Add pair metadata to the risk-managed DataFrame
            risk_managed['ticker_A'] = ticker1
            risk_managed['ticker_B'] = ticker2
            risk_managed['hedge_ratio'] = hedge_ratio
            risk_managed['half_life'] = half_life
            risk_managed['cointegration_pvalue'] = pvalue
            
            results.append(risk_managed)
        
        # Combine results for all pairs.
        if results:
            final_df = pd.concat(results).sort_index()
        else:
            final_df = pd.DataFrame()
        
        if latest_only and not final_df.empty:
            # Retain only the most recent observation per pair.
            final_df = final_df.groupby(['ticker_A', 'ticker_B']).tail(1)
        
        return final_df


if __name__ == "__main__":
    # Example usage for backtesting the Pairs Trading Strategy.
    import logging
    from src.database.config import DatabaseConfig
    
    logging.basicConfig(level=logging.DEBUG)
    # Set up the default database configuration.
    db_config = DatabaseConfig.default()
    
    # Define strategy parameters.
    params = {
        'lookback_period': 252,
        'correlation_threshold': 0.7,
        'zscore_entry_threshold': 2.0,
        'zscore_exit_threshold': 0.5,
        'minimum_half_life': 1,
        'maximum_half_life': 30,
        'rolling_window': 30,
        'coint_pvalue_threshold': 0.05,
        'stop_loss_pct': 0.05,
        'take_profit_pct': 0.10,
        'slippage_pct': 0.001,
        'transaction_cost_pct': 0.001,
        'data_source': 'yfinance'
    }
    
    # Instantiate the Pairs Trading strategy.
    strategy = PairsTrading(db_config, params)
    
    # Define a ticker universe and backtest period.
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # Generate signals and execute the backtest.
    df_results = strategy.generate_signals(tickers, start_date, end_date, initial_position=0, latest_only=False)
    
    # Output a sample of the backtest results.
    print(df_results.tail())