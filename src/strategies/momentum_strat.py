# trading_system/src/strategies/momentum_strat.py

import pandas as pd
from src.strategies.base_strat import BaseStrategy 

class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy"""

    def generate_signals(self, ticker: str) -> pd.DataFrame:
        # Get price data with default 1-year lookback
        prices = self.get_historical_prices(ticker)

        if not self._validate_data(prices, min_records=20):
            return pd.DataFrame()

        # Calculate 12-month momentum
        prices['returns'] = prices['close'].pct_change(periods=21)
        prices['12m_momentum'] = prices['close'].pct_change(periods=252)

        # Generate signals
        prices['signal'] = 0
        prices.loc[prices['12m_momentum'] > 0.15, 'signal'] = 1  # Buy
        prices.loc[prices['12m_momentum'] < -0.15, 'signal'] = -1  # Sell

        return prices[['close', 'returns', '12m_momentum', 'signal']].dropna()