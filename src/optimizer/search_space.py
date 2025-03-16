import numpy as np
from hyperopt import hp

enhanced_market_pressure_strat_search_space = {
    # Analysis window parameters
    'window': hp.quniform('window', 10, 50, 1),  # Rolling window size for analysis
    
    # Statistical parameters
    'confidence_threshold': hp.uniform('confidence_threshold', 0.85, 0.99),  # Statistical confidence threshold
    'pressure_threshold': hp.uniform('pressure_threshold', 0.1, 0.5),  # Threshold for pressure to generate signal
    
    # Distribution fitting parameters
    'volume_weighted': hp.choice('volume_weighted', [True, False]),  # Whether to weight by volume
    'use_multiple_dists': hp.choice('use_multiple_dists', [True]),  # Use multiple distributions
    
    # Trend detection parameters for divergence
    'price_trend_days': hp.quniform('price_trend_days', 3, 15, 1),  # Days to compute price trend
    'pressure_trend_days': hp.quniform('pressure_trend_days', 3, 15, 1),  # Days to compute pressure trend
    'bull_div_threshold': hp.uniform('bull_div_threshold', 0.005, 0.03),  # Threshold for bullish divergence
    'bear_div_threshold': hp.uniform('bear_div_threshold', 0.005, 0.03),  # Threshold for bearish divergence
    
    # Strategy behavior
    'long_only': hp.choice('long_only', [True, False]),  # Whether to only take long positions
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.02, 0.1),  # Stop loss percentage
    'take_profit_pct': hp.uniform('take_profit_pct', 0.03, 0.2),  # Take profit percentage
    'trailing_stop_pct': hp.uniform('trailing_stop_pct', 0.01, 0.08),  # Trailing stop percentage

}

garchx_strat_search_space = {
    # Forecast parameters
    'forecast_horizon': hp.quniform('forecast_horizon', 5, 30, 1),
    'forecast_lookback': hp.quniform('forecast_lookback', 100, 500, 20),
    
    # Signal generation parameters
    'min_volume_strength': hp.uniform('min_volume_strength', 0.5, 2.5),
    'long_only': hp.choice('long_only', [True, False]),
    
    # Position sizing parameters
    'capital': hp.choice('capital', [1.0]),  # Fixed for optimization purposes
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.02, 0.10),
    'take_profit_pct': hp.uniform('take_profit_pct', 0.05, 0.25),
    'trailing_stop_pct': hp.uniform('trailing_stop_pct', 0.01, 0.08),
    
    # Model parameters (for extended optimization)
    'pca_components': hp.quniform('pca_components', 3, 10, 1),
    'winsorize_quantiles': hp.choice('winsorize_quantiles', [
        (0.005, 0.995),
        (0.01, 0.99),
        (0.02, 0.98)
    ]),
    'vol_window': hp.quniform('vol_window', 10, 40, 5)
}


pairs_trading_strat_search_space = {
    # Pair selection parameters
    'correlation_threshold': hp.uniform('correlation_threshold', 0.65, 0.9),  # Tighter lower bound for daily alignment
    'pvalue_threshold': hp.uniform('pvalue_threshold', 0.01, 0.07),  # Stricter cointegration (Engle-Granger)
    'halflife_threshold': hp.quniform('halflife_threshold', 5, 40, 1),  # Extended upper limit for slower mean reversion
    'use_johansen': hp.choice('use_johansen', [True, False]),  # Test both cointegration methods
    
    # Spread calculation
    'lookback_window': hp.quniform('lookback_window', 30, 120, 10),  # 1-4 months (daily granularity)
    'min_data_points': hp.quniform('min_data_points', 63, 504, 21),  # 3 months to 2 years of daily data
    
    # Signal generation
    'entry_threshold': hp.uniform('entry_threshold', 1.8, 3.2),  # Slightly higher for daily noise
    'exit_threshold': hp.uniform('exit_threshold', 0.4, 1.2),  # Wider range for volatility
    
    # Risk management
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.03, 0.2),  # 3%-20% (broader for small-cap swings)
    'take_profit_pct': hp.uniform('take_profit_pct', 0.05, 0.3),  # 5%-30%
    'trailing_stop_pct': hp.uniform('trailing_stop_pct', 0.01, 0.08),  # 1%-8%
}


prophet_momentum_strat_search_space = {
    # Forecast horizons
    'horizon_short': hp.quniform('horizon_short', 5, 10, 1),
    'horizon_mid': hp.quniform('horizon_mid', 12, 21, 1),
    'horizon_long': hp.quniform('horizon_long', 25, 45, 1),
    
    # Prophet model hyperparameters
    'changepoint_prior_scale': hp.loguniform('changepoint_prior_scale', np.log(0.001), np.log(0.5)),
    'seasonality_prior_scale': hp.loguniform('seasonality_prior_scale', np.log(0.01), np.log(15)),
    'holidays_prior_scale': hp.loguniform('holidays_prior_scale', np.log(0.01), np.log(15)),
    'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative']),
    'fourier_order': hp.quniform('fourier_order', 3, 15, 1),
    
    # Signal generation parameters
    't_stat_threshold_long': hp.uniform('t_stat_threshold_long', 1.5, 3.0),
    't_stat_threshold_short': hp.uniform('t_stat_threshold_short', -3.0, -1.5),
    
    # Market regime detection parameters
    'vol_window': hp.quniform('vol_window', 14, 30, 1),
    'hmm_states': hp.choice('hmm_states', [2, 3]),
    'regime_override': hp.choice('regime_override', [True, False]),
    
    # Technical indicator parameters
    'rsi_period': hp.quniform('rsi_period', 7, 21, 1),
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.03, 0.08),  # Indian markets can be more volatile
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),
    'trailing_stop_pct': hp.uniform('trailing_stop_pct', 0, 0.05)
}