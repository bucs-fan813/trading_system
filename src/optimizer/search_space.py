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