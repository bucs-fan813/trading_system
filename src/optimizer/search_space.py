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

cup_and_handle_strat_search_space = {
    # Cup detection parameters
    'min_cup_duration': hp.quniform('min_cup_duration', 20, 60, 1),  # Longer durations for daily timeframe
    'max_cup_duration': hp.quniform('max_cup_duration', 100, 250, 5),  # Indian markets may form longer cups
    'cup_depth_threshold': hp.uniform('cup_depth_threshold', 0.18, 0.40),  # Adjusted for Indian market volatility
    
    # Handle detection parameters
    'min_handle_duration': hp.quniform('min_handle_duration', 5, 20, 1),  # Slightly longer handles
    'max_handle_duration': hp.quniform('max_handle_duration', 25, 70, 2),  # Extended for Indian market
    'handle_depth_threshold': hp.uniform('handle_depth_threshold', 0.25, 0.65),  # Adjusted range
    
    # Breakout parameters
    'breakout_threshold': hp.loguniform('breakout_threshold', np.log(0.002), np.log(0.03)),  # Higher range for more volatile market
    'volume_confirm': hp.choice('volume_confirm', [True, False]),
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.03, 0.12),  # Wider stops for Indian market volatility
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.25),  # Higher profit targets
    'trailing_stop_pct': hp.uniform('trailing_stop_pct', 0, 0.05)
}

triangle_breakout_strat_search_space = {
    # Pattern detection parameters
    'min_points': hp.quniform('min_points', 4, 8, 1),  # Minimum points needed to fit trendlines
    'max_lookback': hp.quniform('max_lookback', 30, 120, 5),  # Window size for pattern detection
    
    # Pattern significance parameters
    'min_pattern_size': hp.uniform('min_pattern_size', 0.02, 0.08),  # Minimum triangle height as % of price
    
    # Breakout parameters
    'breakout_threshold': hp.loguniform('breakout_threshold', np.log(0.003), np.log(0.015)),  # Price movement required to confirm breakout
    'volume_confirm': hp.choice('volume_confirm', [True, False]),  # Whether to require volume confirmation
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.03, 0.08),  # Stop loss percentage
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.18),  # Take profit percentage
    'trailing_stop_pct': hp.uniform('trailing_stop_pct', 0, 0.05)  # Trailing stop percentage
}


volume_breakout_strat_search_space = {
    # Breakout detection parameters
    'lookback_period': hp.quniform('lookback_period', 10, 50, 1),
    'consecutive_bars': hp.quniform('consecutive_bars', 1, 5, 1),
    
    # Volume confirmation parameters
    'volume_threshold': hp.uniform('volume_threshold', 1.2, 3.0),
    'volume_avg_period': hp.quniform('volume_avg_period', 10, 40, 1),
    
    # Price movement parameters
    'price_threshold': hp.uniform('price_threshold', 0.01, 0.05),
    
    # Volatility filter parameters
    'use_atr_filter': hp.choice('use_atr_filter', [True, False]),
    'atr_period': hp.quniform('atr_period', 10, 30, 1),
    'atr_threshold': hp.uniform('atr_threshold', 0.5, 2.0),
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.02, 0.08),
    'take_profit_pct': hp.uniform('take_profit_pct', 0.04, 0.15),
    'trailing_stop_pct': hp.uniform('trailing_stop_pct', 0.0, 0.06),
}

cci_oscillator_strat_search_space = {
    # CCI calculation parameters
    'cci_period': hp.quniform('cci_period', 10, 30, 1),  # Period for CCI calculation
    'cci_upper_band': hp.quniform('cci_upper_band', 80, 200, 5),  # Upper threshold for sell signal
    'cci_lower_band': hp.quniform('cci_lower_band', -200, -80, 5),  # Lower threshold for buy signal
    
    # Strategy configuration
    'long_only': hp.choice('long_only', [True, False]),  # Whether to take short positions
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.08),  # Stop loss percentage
    'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.15),  # Take profit percentage
    'trailing_stop_pct': hp.uniform('trailing_stop_pct', 0.0, 0.06)
}

disparity_index_strat_search_space = {
    # Disparity Index parameters
    'di_lookback': hp.quniform('di_lookback', 8, 30, 1),  # Shorter to longer MA periods
    'consecutive_period': hp.quniform('consecutive_period', 2, 7, 1),  # More/less sensitive to reversals
    
    # Trading direction parameters
    'long_only': hp.choice('long_only', [True, False]),  # Indian markets have uptick rule considerations
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.02, 0.08),  # Lower due to higher volatility in Indian markets
    'take_profit_pct': hp.uniform('take_profit_pct', 0.04, 0.15),  # Realistic profit targets for Indian stocks
    'trailing_stop_pct': hp.uniform('trailing_stop_pct', 0.0, 0.06),  # Optional trailing stop
}

relative_strength_index_strat_search_space = {
    # RSI Parameters
    'rsi_period': hp.quniform('rsi_period', 10, 21, 1),  # Traditional RSI periods with focus on 14±7
    'oversold': hp.uniform('oversold', 25.0, 35.0),      # Oversold thresholds
    'overbought': hp.uniform('overbought', 65.0, 75.0),  # Overbought thresholds
    
    # Risk Management Parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.02, 0.08),  # Higher volatility in Indian markets
    'take_profit_pct': hp.uniform('take_profit_pct', 0.04, 0.15),  # Balanced risk-reward
    'trailing_stop_pct': hp.uniform('trailing_stop_pct', 0.0, 0.05),  # Optional trailing stop

    # Strategy Configuration
    'long_only': hp.choice('long_only', [True, False])  # Test both long-only and long-short
}

stochastic_oscillator_strat_search_space = {
    # Stochastic Oscillator Parameters
    'k_period': hp.quniform('k_period', 5, 21, 1),  # Lookback period for %K
    'd_period': hp.quniform('d_period', 2, 7, 1),   # Smoothing period for %D
    'overbought': hp.quniform('overbought', 70, 90, 1),  # Overbought threshold
    'oversold': hp.quniform('oversold', 10, 30, 1),      # Oversold threshold
    
    # Risk Management Parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.015, 0.08),    # Wider range for Indian markets volatility
    'take_profit_pct': hp.uniform('take_profit_pct', 0.025, 0.15), # Balanced for Indian markets
    'trailing_stop_pct': hp.uniform('trailing_stop_pct', 0.02, 0.07), # Trailing stop parameter
    
    # Strategy Configuration
    'long_only': hp.choice('long_only', [True, False])  # Indian markets allow shorting with restrictions
}

williams_percent_r_strat_search_space = {
    # Williams %R parameters
    'wr_period': hp.quniform('wr_period', 10, 30, 1),  # Lookback period range suitable for Indian markets
    'oversold_threshold': hp.uniform('oversold_threshold', -85, -75),  # Oversold threshold
    'overbought_threshold': hp.uniform('overbought_threshold', -25, -15),  # Overbought threshold
    
    # Strategy behavior
    'long_only': hp.choice('long_only', [True, False]),  # Test both long-only and long-short
    'data_lookback': hp.quniform('data_lookback', 200, 500, 25),  # Data history required
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.02, 0.08),  # Higher volatility in Indian markets
    'take_profit_pct': hp.uniform('take_profit_pct', 0.05, 0.15),  # Realistic profit targets
    'trailing_stop_pct': hp.uniform('trailing_stop_pct', 0.0, 0.08),  # Optional trailing stop
    
    # Market friction parameters
    'slippage_pct': hp.loguniform('slippage_pct', np.log(0.001), np.log(0.005)),  # Higher for mid/small caps
    'transaction_cost_pct': hp.loguniform('transaction_cost_pct', np.log(0.0005), np.log(0.003))  # Indian brokerages
}