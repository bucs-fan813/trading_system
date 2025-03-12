"""
Market Pressure Strategy with Integrated Risk Management

This strategy computes an enhanced market pressure indicator based on a normalized 
price position and a volatility adjustment. Mathematically, for each bar:

    norm_pos  = (close - low) / (high - low)
    volatility = rolling_std(high - low) over a lookback window
    vap       = norm_pos * (1 + volatility/rolling_mean(volatility))
    z_pos     = (norm_pos - rolling_mean(norm_pos)) / (rolling_std(norm_pos) + ε)

Over a rolling window of recent norm_pos values (optionally using volume‐based weights),
a Beta distribution is fitted (using MLE or method‐of‐moments as fallback) to yield parameters α and β.
These parameters produce a buy probability: 1 – BetaCDF(0.6, α, β) and a sell probability: BetaCDF(0.4, α, β).
The pressure ratio is then defined as:
    
    pressure_ratio = (buy_prob - sell_prob) / (buy_prob + sell_prob)

A high absolute pressure ratio indicates a strong signal (with sign determining “buying” vs “selling”).
This raw signal is then passed to a risk management component (RiskManager) that applies adjustments 
for slippage and transaction costs, and enforces stop-loss/take-profit thresholds.

The strategy supports full backtesting (entire DataFrame output) as well as EOD forecasting (latest row only).
It supports a vectorized multi–ticker analysis by grouping the historical data and applying the calculations
in an efficient fashion.

Outputs:
    A DataFrame with the following columns for each bar:
        - open, high, low, close [, volume]
        - norm_pos, vap, volatility, z_pos
        - buy_prob, sell_prob, pressure_ratio, signal_confidence, net_pressure, indicator_direction
        - signal (raw trading signal: 1 for buy, -1 for sell, 0 for hold)
        - daily_return: percentage change of close price
        - strategy_return: daily return scaled by previous signal (raw return)
        - position: updated position from risk management
        - rm_strategy_return: risk-managed (realized) trade return
        - rm_cumulative_return: cumulative return from trades after risk management
        - rm_action: indicator (stop_loss, take_profit, or signal_exit)

Args:
    ticker (str or List[str]): Stock ticker symbol or list of tickers.
    start_date (str, optional): Backtest start date in 'YYYY-MM-DD' format.
    end_date (str, optional): Backtest end date in 'YYYY-MM-DD' format.
    initial_position (int): Starting trading position (default=0, meaning flat).
    latest_only (bool): If True, returns only the final row (per ticker).
    
Strategy-specific parameters (in self.params or defaults):
    - position_window       : Rolling window for Beta estimation (default: 20).
    - volume_enabled        : Whether to use volume weighting for Beta estimation (default: True).
    - default_confidence      : Minimum absolute pressure ratio to trigger a trade (default: 0.7).
    - signal_expiry         : Number of bars a given signal is active (default: 3).
    - risk_per_trade        : Proportion of account risked on each trade (default: 0.01).
    - max_exposure          : Maximum allowed exposure (default: 0.04).
    - stop_loss_pct         : Stop loss percentage (default: 0.05).
    - take_profit_pct       : Take profit percentage (default: 0.10).
    - slippage_pct          : Slippage percentage (default: 0.001).
    - transaction_cost_pct  : Transaction cost percentage (default: 0.001).
"""

import math
import random
import numpy as np
import pandas as pd
from scipy.stats import beta, mannwhitneyu
from scipy import optimize
from time import perf_counter

# Import the BaseStrategy and RiskManager classes from your modules
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager


def advanced_normalization(ohlc_df, lookback_vol=20, lookback_z=50):
    """
    Compute normalized positioning, volatility adjustment and z-score for a given OHLC DataFrame.
    
    Args:
        ohlc_df (pd.DataFrame): DataFrame containing 'high', 'low', 'close' columns.
        lookback_vol (int): Lookback window for volatility calculation.
        lookback_z (int): Lookback window for z-score calculation.
    
    Returns:
        pd.DataFrame: DataFrame with columns 'norm_pos', 'vap', 'volatility', and 'z_pos'.
    """
    df = ohlc_df.copy()
    df['range'] = df['high'] - df['low']
    df['norm_pos'] = (df['close'] - df['low']) / df['range'].replace(0, 1e-9)
    df['volatility'] = df['range'].rolling(lookback_vol, min_periods=1).std()
    rolling_mean_vol = df['volatility'].rolling(lookback_vol, min_periods=1).mean().replace(0, 1e-9)
    df['vap'] = df['norm_pos'] * (1 + (df['volatility'] / rolling_mean_vol))
    norm_mean = df['norm_pos'].rolling(lookback_z, min_periods=1).mean()
    norm_std = df['norm_pos'].rolling(lookback_z, min_periods=1).std().replace(0, 1e-9)
    df['z_pos'] = (df['norm_pos'] - norm_mean) / norm_std
    return df[['norm_pos', 'vap', 'volatility', 'z_pos']]


def transform_volume_confidence(volume_data, window=20):
    """
    Compute volume-based confidence weights using a rolling average.
    
    Args:
        volume_data (list): List of volume values.
        window (int): Rolling window period.
    
    Returns:
        list: Confidence weights transformed via a sigmoid function.
    """
    if len(volume_data) < window:
        return [1.0] * len(volume_data)
    
    rel_volume = []
    for i in range(len(volume_data)):
        if i < window:
            avg_vol = sum(volume_data[:i+1]) / (i+1)
        else:
            avg_vol = sum(volume_data[i-window:i]) / window
        rel_vol = volume_data[i] / avg_vol if avg_vol > 0 else 1.0
        rel_volume.append(rel_vol)
    
    confidence_weights = []
    for rv in rel_volume:
        confidence = 1.0 + (2.0 / (1.0 + math.exp(-2.0 * (rv - 1.0))) - 1.0)
        confidence_weights.append(confidence)
    
    return confidence_weights


def beta_distribution_mle(data, weights=None):
    """
    Estimate Beta distribution parameters via (weighted) maximum likelihood estimation.
    
    Args:
        data (list): List of normalized positions.
        weights (list, optional): List of weights for each data point.
    
    Returns:
        tuple: Estimated (alpha, beta) if successful.
    
    Raises:
        ValueError: If the optimization fails.
    """
    if weights is None:
        weights = [1.0] * len(data)
    
    def neg_log_likelihood(params):
        a, b = params
        if a <= 0 or b <= 0:
            return float('inf')
        nll = 0
        for x, w in zip(data, weights):
            x = max(0.001, min(0.999, x))
            log_pdf = (a - 1) * math.log(x) + (b - 1) * math.log(1 - x)
            log_pdf -= math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
            nll -= w * log_pdf
        return nll

    weighted_mean = sum(x * w for x, w in zip(data, weights)) / sum(weights)
    weighted_var = sum(w * (x - weighted_mean)**2 for x, w in zip(data, weights)) / sum(weights)
    if weighted_var < weighted_mean * (1 - weighted_mean):
        t = weighted_mean * (1 - weighted_mean) / weighted_var - 1
        a_init = weighted_mean * t
        b_init = (1 - weighted_mean) * t
    else:
        a_init = 1.0
        b_init = 1.0

    a_init = max(0.5, min(10.0, a_init))
    b_init = max(0.5, min(10.0, b_init))
    result = optimize.minimize(
        neg_log_likelihood,
        [a_init, b_init],
        bounds=[(0.01, 100), (0.01, 100)]
    )
    
    if result.success:
        return result.x
    else:
        raise ValueError("MLE optimization failed")


def compute_optimal_distribution(normalized_positions, window=20, volume_weights=None):
    """
    Compute Beta distribution parameters over the last 'window' normalized positions.
    
    Args:
        normalized_positions (list): List of normalized positions.
        window (int): Rolling window size.
        volume_weights (list, optional): Corresponding volume weights.
    
    Returns:
        dict: Dictionary with distribution parameters (alpha, beta), a validity flag, and method used.
    """
    if len(normalized_positions) < window:
        return {'alpha': 1.0, 'beta': 1.0, 'valid': False}
    
    positions = normalized_positions[-window:]
    if volume_weights and len(volume_weights) >= window:
        weights = volume_weights[-window:]
    else:
        weights = [1.0] * window
    
    valid_positions = []
    valid_weights = []
    for p, w in zip(positions, weights):
        if 0.001 < p < 0.999:
            valid_positions.append(p)
            valid_weights.append(w)
    
    if len(valid_positions) < 5:
        return {'alpha': 1.0, 'beta': 1.0, 'valid': False}
    
    try:
        a, b = beta_distribution_mle(valid_positions, valid_weights)
        if 0.01 <= a <= 100 and 0.01 <= b <= 100:
            return {'alpha': a, 'beta': b, 'valid': True, 'method': 'MLE'}
    except Exception:
        pass
    
    try:
        mean = sum(p * w for p, w in zip(valid_positions, valid_weights)) / sum(valid_weights)
        variance = sum(w * (p - mean)**2 for p, w in zip(valid_positions, valid_weights)) / sum(valid_weights)
        if variance < mean * (1 - mean):
            t = mean * (1 - mean) / variance - 1
            a = mean * t
            b = (1 - mean) * t
            if 0.01 <= a <= 100 and 0.01 <= b <= 100:
                return {'alpha': a, 'beta': b, 'valid': True, 'method': 'MoM'}
    except Exception:
        pass

    mean_pos = sum(valid_positions) / len(valid_positions)
    if mean_pos <= 0.5:
        return {'alpha': 1.0, 'beta': 1.0 + 3.0 * (0.5 - mean_pos) * 2, 'valid': True, 'method': 'default'}
    else:
        return {'alpha': 1.0 + 3.0 * (mean_pos - 0.5) * 2, 'beta': 1.0, 'valid': True, 'method': 'default'}


class MarketPressureStrategy(BaseStrategy):
    """
    Enhanced Market Pressure Strategy that integrates risk management.
    
    This strategy extends BaseStrategy and supports vectorized backtesting over one or 
    multiple tickers. It calculates a market pressure indicator based on the normalized 
    price position, volatility adjustment, and rolling Beta distribution estimation.
    The resulting indicator is used to generate a raw trading signal (1 for buy, 
    -1 for sell, and 0 for hold). The raw signal is then risk-managed using the RiskManager
    to adjust for stop-loss and take-profit thresholds.
    
    The strategy returns a DataFrame containing price references, indicator values,
    the raw and adjusted signals, and return metrics.
    """
    def __init__(self, db_config, params=None):
        """
        Initialize the MarketPressureStrategy using database configuration and strategy parameters.
        
        Args:
            db_config (DatabaseConfig): Database configuration settings.
            params (dict, optional): Strategy-specific parameters.
        """
        super().__init__(db_config, params)
        defaults = {
            'position_window': 20,
            'volume_enabled': True,
            'default_confidence': 0.7,
            'signal_expiry': 3,
            'risk_per_trade': 0.01,
            'max_exposure': 0.04,
            'lookback_vol': 20,
            'lookback_z': 50,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001
        }
        # Set any missing parameters to defaults.
        self.params = {**defaults, **(params or {})}
        self.risk_manager = RiskManager(
            stop_loss_pct=self.params['stop_loss_pct'],
            take_profit_pct=self.params['take_profit_pct'],
            slippage_pct=self.params['slippage_pct'],
            transaction_cost_pct=self.params['transaction_cost_pct']
        )
    
    def generate_signals(
        self,
        ticker,
        start_date: str = None,
        end_date: str = None,
        initial_position: int = 0,
        latest_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate trading signals and compute risk-managed returns.
        
        For each ticker the following steps are performed:
            1. Historical OHLC data is retrieved.
            2. Advanced normalization is computed (norm_pos, vap, volatility, z_pos).
            3. (If enabled) volume confidence weights are computed.
            4. Over a rolling window (position_window), a Beta distribution is fitted 
               to the normalized positions and used to derive:
                   - buy_prob = 1 - BetaCDF(0.6, α, β)
                   - sell_prob = BetaCDF(0.4, α, β)
                   - pressure_ratio = (buy_prob - sell_prob) / (buy_prob + sell_prob)
            5. A raw signal is generated based on whether the absolute pressure (confidence) 
               exceeds the default threshold.
            6. Daily returns and strategy returns are computed.
            7. The RiskManager is applied to handle stop-loss/take-profit rules.
        
        Args:
            ticker (str or List[str]): Single ticker symbol or a list.
            start_date (str, optional): Start date (YYYY-MM-DD).
            end_date (str, optional): End date (YYYY-MM-DD).
            initial_position (int): Starting position.
            latest_only (bool): If True, only the final signal row is returned per ticker.
            
        Returns:
            pd.DataFrame: DataFrame containing price data, indicator values, trading signals, 
                          return calculations and risk-managed returns.
        """
        # Retrieve historical price data using the BaseStrategy method.
        ohlc = self.get_historical_prices(ticker, from_date=start_date, to_date=end_date)
        dfs = []  # list for each ticker

        # Handle single ticker vs. multiple tickers (multi-index DataFrame)
        if isinstance(ticker, str):
            ohlc['_ticker'] = ticker
            groups = [(ticker, ohlc)]
        else:
            # In multiple-ticker case, the ohlc has a MultiIndex (ticker, date)
            ohlc = ohlc.reset_index(level='ticker')
            groups = list(ohlc.groupby('ticker'))
        
        # Process each ticker's group
        for tk, group in groups:
            group = group.sort_index()  # sort by date
            group.index = pd.to_datetime(group.index)
            
            # Compute daily return (percentage change)
            group['daily_return'] = group['close'].pct_change()
            
            # Compute advanced normalization for indicator calculations
            norm_df = advanced_normalization(group, lookback_vol=self.params['lookback_vol'], lookback_z=self.params['lookback_z'])
            group = group.join(norm_df)
            
            # Compute volume confidence weights if enabled and volume is available
            if self.params['volume_enabled'] and 'volume' in group.columns:
                vol_weights = transform_volume_confidence(group['volume'].tolist(), window=self.params['position_window'])
            else:
                vol_weights = [1.0] * len(group)
            
            # Initialize new indicator columns with NaN defaults.
            group['buy_prob'] = np.nan
            group['sell_prob'] = np.nan
            group['pressure_ratio'] = np.nan
            group['signal_confidence'] = np.nan
            group['net_pressure'] = np.nan
            group['indicator_direction'] = np.nan

            norm_positions = group['norm_pos'].tolist()

            # Rolling window calculation over the group
            window = self.params['position_window']
            for i in range(len(group)):
                if i < window - 1:
                    # Not enough data for a rolling window
                    continue
                window_norm = norm_positions[i - window + 1:i + 1]
                window_vol = vol_weights[i - window + 1:i + 1]
                distribution = compute_optimal_distribution(window_norm, window, volume_weights=window_vol)
                if not distribution.get('valid', False):
                    bp = 0.5
                    sp = 0.5
                else:
                    a = distribution['alpha']
                    b_param = distribution['beta']
                    bp = 1 - beta.cdf(0.6, a, b_param)
                    sp = beta.cdf(0.4, a, b_param)
                denominator = bp + sp if (bp + sp) != 0 else 1e-9
                pratio = (bp - sp) / denominator
                confidence = abs(pratio)
                direction = 'buying' if pratio > 0 else ('selling' if pratio < 0 else 'neutral')
                group.iloc[i, group.columns.get_loc('buy_prob')] = bp
                group.iloc[i, group.columns.get_loc('sell_prob')] = sp
                group.iloc[i, group.columns.get_loc('pressure_ratio')] = pratio
                group.iloc[i, group.columns.get_loc('signal_confidence')] = confidence
                group.iloc[i, group.columns.get_loc('net_pressure')] = bp - sp
                group.iloc[i, group.columns.get_loc('indicator_direction')] = direction

            # Generate raw trading signal based on computed net_pressure (and minimum confidence threshold)
            def derive_signal(row):
                if pd.isna(row['signal_confidence']):
                    return 0
                if row['signal_confidence'] >= self.params['default_confidence']:
                    return 1 if row['indicator_direction'] == 'buying' else -1
                return 0
            group['signal'] = group.apply(derive_signal, axis=1)
            
            # Calculate a simple strategy return: yesterday's signal times today's daily return.
            group['strategy_return'] = group['daily_return'].shift(-1) * group['signal']
            
            # Apply risk management to generate risk–managed returns and positions.
            rm_results = self.risk_manager.apply(group[['high', 'low', 'close', 'signal']].copy(), initial_position=initial_position)
            # Rename risk manager outputs to avoid name collisions.
            rm_results.rename(columns={
                'return': 'rm_strategy_return',
                'cumulative_return': 'rm_cumulative_return',
                'exit_type': 'rm_action'
            }, inplace=True)
            group = group.join(rm_results[['position', 'rm_strategy_return', 'rm_cumulative_return', 'rm_action']])
            
            # Add ticker information
            group['ticker'] = tk
            dfs.append(group)
        
        result = pd.concat(dfs) if len(dfs) > 1 else dfs[0]
        result.sort_index(inplace=True)
        
        # If only the latest signal is needed, filter to last row per ticker.
        if latest_only:
            if 'ticker' in result.columns:
                result = result.groupby('ticker').tail(1)
            else:
                result = result.tail(1)
        
        return result
        

# --- EXAMPLE USAGE (for testing/backtest) ---
if __name__ == "__main__":
    # For demonstration, create a random OHLCV dataset.
    np.random.seed(42)
    periods = 100
    data = {
        'open': np.random.uniform(100, 105, periods),
        'high': np.random.uniform(105, 110, periods),
        'low': np.random.uniform(95, 100, periods),
        'close': np.random.uniform(100, 105, periods),
        'volume': np.random.uniform(1000, 5000, periods),
        'timestamp': pd.date_range(start="2025-03-01", periods=periods, freq="D")
    }
    ohlc_df = pd.DataFrame(data)
    
    # Assume an in-memory SQLite DB for demonstration (adjust DatabaseConfig accordingly)
    from src.database.config import DatabaseConfig
    db_config = DatabaseConfig.default()
    
    # Here we simulate storing the data into your database,
    # but for backtesting one might use get_historical_prices. This demo bypasses the DB.
    # Instead, we directly pass the ohlc_df as if it were returned by get_historical_prices.
    # (In real use, ensure your daily_prices table has corresponding data.)
    
    # Instantiate the strategy.
    strategy = MarketPressureStrategy(db_config)
    
    # For demonstration purposes, we simulate a single-ticker backtest.
    # In real use, the historical data is pulled from the database.
    # Here, we temporarily override get_historical_prices.
    strategy.get_historical_prices = lambda tickers, **kwargs: ohlc_df.set_index('timestamp')
    
    # Generate the full backtest signals.
    signals_df = strategy.generate_signals("DEMO", start_date="2025-03-01", end_date="2025-06-01", initial_position=0)
    
    print("Full Backtest Signals:")
    print(signals_df.tail(10))
    
    # Generate signal for the most recent date only.
    latest_signal = strategy.generate_signals("DEMO", start_date="2025-03-01", end_date="2025-06-01", initial_position=0, latest_only=True)
    print("\nLatest Signal:")
    print(latest_signal)