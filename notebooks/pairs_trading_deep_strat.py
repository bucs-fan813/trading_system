# trading_system/src/strategies/pairs_trading.py

import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from itertools import combinations
from typing import List, Dict, Optional, Union
import logging
from src.strategies.base_strat import BaseStrategy, DataRetrievalError
from src.strategies.risk_management import RiskManager
from src.database.config import DatabaseConfig

class PairTradingStrategy(BaseStrategy):
    """
    Pairs Trading Strategy based on Cointegration and Statistical Arbitrage.

    This strategy identifies pairs of assets with historical cointegration,
    computes a mean-reverting spread, and generates trades when the spread
    deviates from its historical mean. Risk management is applied to each
    asset's aggregated positions.

    Steps:
    1. Retrieve and align historical prices for all tickers.
    2. Pre-filter pairs using rolling correlation.
    3. Test cointegration on candidate pairs.
    4. Compute spread and generate signals based on z-score thresholds.
    5. Aggregate positions across all pairs and apply risk management.

    Args:
        db_config (DatabaseConfig): Database configuration.
        params (dict): Strategy parameters including:
            - correlation_window: Rolling window for correlation (default: 30)
            - correlation_threshold: Minimum correlation (default: 0.7)
            - adf_p_value: Max p-value for cointegration (default: 0.05)
            - max_half_life: Max allowed half-life (default: 30)
            - zscore_window: Window for z-score (default: 30)
            - entry_threshold: Z-score entry level (default: 2.0)
            - exit_threshold: Z-score exit level (default: 0.5)
            - stop_loss_pct, take_profit_pct, slippage_pct, transaction_cost_pct

    Output DataFrame includes:
        - Price data (open, high, low, close)
        - Signals and positions for each asset
        - Risk-managed returns and cumulative returns
    """

    def __init__(self, db_config: DatabaseConfig, params: Optional[Dict] = None):
        super().__init__(db_config, params)
        self.default_params = {
            'correlation_window': 30,
            'correlation_threshold': 0.7,
            'adf_p_value': 0.05,
            'max_half_life': 30,
            'zscore_window': 30,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'slippage_pct': 0.001,
            'transaction_cost_pct': 0.001
        }
        self.params = {**self.default_params, **(params or {})}

    def generate_signals(self, tickers: Union[str, List[str]],
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        initial_position: int = 0,
                        latest_only: bool = False) -> pd.DataFrame:
        """
        Generate signals for pairs trading. Handles multiple tickers and backtesting.

        Args:
            tickers: List of tickers to form pairs.
            start_date: Backtest start date (None for earliest).
            end_date: Backtest end date (None for latest).
            initial_position: Initial position (0 for flat).
            latest_only: Return only latest signals.

        Returns:
            DataFrame with signals, positions, and risk metrics.
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        # Retrieve and align price data
        prices = self.get_historical_prices(tickers, from_date=start_date, to_date=end_date)
        close_prices = prices['close'].unstack('ticker')
        close_prices.ffill(inplace=True)
        close_prices.dropna(axis=1, how='any', inplace=True)
        aligned_prices = prices.swaplevel().unstack().ffill().dropna()

        # Step 2: Pre-filter pairs by rolling correlation
        valid_pairs = self._filter_pairs_by_correlation(close_prices)

        # Step 3: Cointegration testing
        cointegrated = self._test_cointegration(valid_pairs, close_prices)

        # Step 4: Compute spreads and z-scores
        spreads = self._compute_spreads(cointegrated, close_prices)

        # Step 5: Generate signals and aggregate positions
        signals = self._generate_pair_signals(spreads, close_prices)

        # Combine with price data and apply risk management
        return self._apply_risk_management(signals, aligned_prices, initial_position, latest_only)

    def _filter_pairs_by_correlation(self, close_prices: pd.DataFrame) -> List:
        """Pre-filter pairs using rolling correlation."""
        pairs = list(combinations(close_prices.columns, 2))
        window = self.params['correlation_window']
        threshold = self.params['correlation_threshold']

        valid = []
        for a, b in pairs:
            rolling_corr = close_prices[a].rolling(window).corr(close_prices[b])
            if rolling_corr.mean() > threshold:
                valid.append((a, b))
        return valid

    def _test_cointegration(self, pairs: List, close_prices: pd.DataFrame) -> Dict:
        """Test pairs for cointegration and return valid pairs with hedge ratios."""
        cointegrated = {}
        for a, b in pairs:
            try:
                # OLS without intercept for hedge ratio
                model = OLS(close_prices[a], close_prices[b]).fit()
                hr = model.params[0]
                spread = close_prices[a] - hr * close_prices[b]

                # ADF test with constant
                adf = adfuller(spread, maxlag=1, regression='c')
                if adf[1] > self.params['adf_p_value']:
                    continue

                # Check half-life
                hl = self._calculate_half_life(spread)
                if hl <= self.params['max_half_life']:
                    cointegrated[(a, b)] = {'hedge_ratio': hr, 'spread': spread}
            except Exception as e:
                self.logger.error(f"Cointegration test failed for {a}-{b}: {e}")
        return cointegrated

    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life of mean reversion using AR(1) model."""
        try:
            model = AutoReg(spread, lags=1).fit()
            beta = model.params[1]
            return max(0.1, -np.log(2) / np.log(1 + beta))  # Avoid negative/inf
        except:
            return np.inf

    def _compute_spreads(self, cointegrated: Dict, close_prices: pd.DataFrame) -> Dict:
        """Compute z-scores for valid pairs."""
        window = self.params['zscore_window']
        for key in cointegrated:
            spread = cointegrated[key]['spread']
            mean = spread.rolling(window).mean()
            std = spread.rolling(window).std()
            cointegrated[key]['zscore'] = (spread - mean) / std.replace(0, 1e-6)
        return cointegrated

    def _generate_pair_signals(self, spreads: Dict, close_prices: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for all pairs and aggregate positions."""
        entry = self.params['entry_threshold']
        exit = self.params['exit_threshold']
        signals = pd.DataFrame(0, index=close_prices.index, columns=close_prices.columns)

        for (a, b), data in spreads.items():
            z = data['zscore']
            hr = data['hedge_ratio']

            # Vectorized signal logic
            long_entry = (z < -entry) & (z.shift(1) >= -entry)
            short_entry = (z > entry) & (z.shift(1) <= entry)
            long_exit = (z >= -exit) & (z.shift(1) < -exit)
            short_exit = (z <= exit) & (z.shift(1) > exit)

            signals[a] += long_entry.astype(int) - short_entry.astype(int)
            signals[b] += -hr * (long_entry.astype(int) - short_entry.astype(int))

            signals[a] -= long_exit.astype(int) - short_exit.astype(int)
            signals[b] += hr * (long_exit.astype(int) - short_exit.astype(int))

        return signals.clip(-1, 1)  # Cap positions at +/-1

    def _apply_risk_management(self, signals: pd.DataFrame, prices: pd.DataFrame,
                              initial_position: int, latest_only: bool) -> pd.DataFrame:
        """Integrate risk management for each asset's aggregated positions."""
        dfs = []
        for ticker in signals.columns:
            df = prices.xs(ticker, level='ticker')[['open', 'high', 'low', 'close']].copy()
            df['signal'] = signals[ticker]
            rm = RiskManager(
                stop_loss_pct=self.params['stop_loss_pct'],
                take_profit_pct=self.params['take_profit_pct'],
                slippage_pct=self.params['slippage_pct'],
                transaction_cost_pct=self.params['transaction_cost_pct']
            )
            rm_df = rm.apply(df, initial_position)
            rm_df['ticker'] = ticker
            dfs.append(rm_df.reset_index())

        full_df = pd.concat(dfs).set_index(['ticker', 'date'])
        return full_df.groupby('ticker').last() if latest_only else full_df