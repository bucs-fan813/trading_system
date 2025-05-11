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
    # Cup Detection Parameters
    'min_cup_duration': hp.quniform('min_cup_duration', 20, 60, 1), # Approx 1-3 months
    'max_cup_duration': hp.quniform('max_cup_duration', 60, 200, 5), # Approx 3-10 months # Ensure min_cup_duration < max_cup_duration in your objective function if needed
    'cup_depth_threshold': hp.uniform('cup_depth_threshold', 0.15, 0.50), # Max relative depth (D/R)
    'peak_similarity_threshold': hp.uniform('peak_similarity_threshold', 0.05, 0.25), # Max relative diff between cup peaks |Pl-Pr|/R
    
    # Handle Detection Parameters
    'min_handle_duration': hp.quniform('min_handle_duration', 3, 15, 1), # Approx 3 days to 3 weeks
    'max_handle_duration': hp.quniform('max_handle_duration', 15, 50, 2), # Approx 3 weeks to 2.5 months # Ensure min_handle_duration < max_handle_duration in your objective function if needed
    'handle_depth_threshold': hp.uniform('handle_depth_threshold', 0.25, 0.65 ),# Max handle depth relative to cup depth (Dh/D)
    
    # Pattern Recognition Parameters
     'extrema_order': hp.quniform('extrema_order', 3, 15, 1), # Order for local extrema detection (scipy.signal.argrelextrema)
     
    # Breakout Parameters 
    'breakout_threshold': hp.loguniform('breakout_threshold', np.log(0.002), np.log(0.025)), # % above resistance for signal (0.2% to 2.5%)
    'volume_confirm': hp.choice('volume_confirm', [True, False]), # Whether to require volume surge on breakout
    
    # Risk Management Parameters (from RiskManager)
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.03, 0.15), # 3% to 15% stop loss
    'take_profit_pct': hp.uniform('take_profit_pct', 0.05, 0.30), # 5% to 30% take profit
    'trailing_stop_pct': hp.uniform('trailing_stop_pct', 0.02, 0.15) # 0% (disabled) 
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
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]) 
}

disparity_index_strat_search_space = {
    # Disparity Index parameters
    'di_lookback': hp.quniform('di_lookback', 8, 30, 1),  # Shorter to longer MA periods
    'consecutive_period': hp.quniform('consecutive_period', 2, 7, 1),  # More/less sensitive to reversals
    
    # Trading direction parameters
    'long_only': hp.choice('long_only', [True, False]),  # Indian markets have uptick rule considerations
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.08),  # Lower due to higher volatility in Indian markets
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),  # Realistic profit targets for Indian stocks
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]) 
}

relative_strength_index_strat_search_space = {
    # RSI Parameters
    'rsi_period': hp.quniform('rsi_period', 10, 21, 1),  # Traditional RSI periods with focus on 14±7
    'oversold': hp.uniform('oversold', 25.0, 35.0),      # Oversold thresholds
    'overbought': hp.uniform('overbought', 65.0, 75.0),  # Overbought thresholds
    
    # Risk Management Parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.02, 0.08),  # Higher volatility in Indian markets
    'take_profit_pct': hp.uniform('take_profit_pct', 0.04, 0.15),  # Balanced risk-reward
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]),

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
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]),

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
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]) 
}

awesome_oscillator_strat_search_space = {
    # Core AO parameters
    'short_period': hp.quniform('short_period', 3, 15, 1),  # Short SMA period
    'long_period': hp.quniform('long_period', 20, 60, 2),   # Long SMA period
    
    # Trading mode
    'long_only': hp.choice('long_only', [True, False]),     # Allow only long positions or both long/short
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.1),         # Stop loss percentage (smaller for Indian markets due to volatility)
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),      # Take profit percentage
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ])   # Trailing stop percentage
}

# coppock_curve_strat_search_space = {
#     # ROC parameters
#     'roc1_months': hp.quniform('roc1_months', 10, 18, 1),  # Traditional is 14
#     'roc2_months': hp.quniform('roc2_months', 8, 14, 1),   # Traditional is 11
    
#     # WMA lookback parameter
#     'wma_lookback': hp.quniform('wma_lookback', 6, 14, 1), # In months (default is 10)
    
#     # Signal generation parameters
#     'method': hp.choice('method', ['zero_crossing', 'directional']),
#     'long_only': hp.choice('long_only', [True, False]),
    
#     # For directional method
#     'sustain_days': hp.quniform('sustain_days', 3, 7, 1),
#     'trend_strength_threshold': hp.uniform('trend_strength_threshold', 0.6, 0.9),
    
#     # Strength calculation
#     'strength_window': hp.quniform('strength_window', 252, 756, 63),  # ~1-3 years in trading days
#     'normalize_strength': hp.choice('normalize_strength', [True, False]),
    
#     # Risk management parameters
#     'stop_loss_pct': hp.uniform('stop_loss_pct', 0.03, 0.12),         # Higher for more volatile markets
#     'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.25),     # Adjusted for Indian market volatility
#     'trailing_stop_pct': hp.uniform('trailing_stop_pct', 0.0, 0.05),   # Trailing stop percentage
# }


coppock_curve_strat_search_space = {
    'roc1_months': hp.quniform('roc1_months', 2, 8, 1),  # Shorter range (e.g., 2-8 months => ~42-168 days)
    'roc2_months': hp.quniform('roc2_months', 1, 6, 1),  # Shorter range (e.g., 1-6 months => ~21-126 days) ensuring roc2 < roc1 often

    'wma_lookback': hp.quniform('wma_lookback', 1, 5, 1), # Shorter range (e.g., 1-5 months => ~21-105 days)

    'method': hp.choice('method', ['zero_crossing', 'directional']),
    'long_only': hp.choice('long_only', [True, False]), # Test both long-only and long/short

    'zc_require_prior_state': hp.choice('zc_require_prior_state', [True, False]), # NEW: Test strict vs simple zero cross
    'zc_sustain_days': hp.quniform('zc_sustain_days', 2, 5, 1), # NEW: Lookback for prior state (if zc_require_prior_state=True)

    'sustain_days': hp.quniform('sustain_days', 2, 5, 1), # How many days of sustained CC direction needed (reduced upper bound slightly)

    'dir_require_prior_trend': hp.choice('dir_require_prior_trend', [True, False]), # NEW: Test if prior counter-trend is required

    'trend_strength_threshold': hp.uniform('trend_strength_threshold', 0.1, 0.7), # Widen range significantly downwards (if dir_require_prior_trend=True)

    'strength_window': hp.quniform('strength_window', 252, 504, 63),  # ~1-2 years in trading days (reduced upper bound slightly)
    'normalize_strength': hp.choice('normalize_strength', [True, False]),

    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.1), # Wider range for SL, acknowledging potential volatility
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15), # Wider range for TP
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]),
}

know_sure_thing_strat_search_space = {
    # KST ROC periods (typically shorter for Indian markets due to higher volatility)
    'roc_periods': hp.choice('roc_periods', [
        [8, 13, 21, 34],  # Fibonacci-based sequence
        [10, 15, 20, 30],  # Traditional setup
        [5, 10, 15, 20],   # Shorter timeframes for small caps
        [12, 18, 24, 36]   # Longer timeframes for large caps
    ]),
    
    # SMA smoothing periods
    'sma_periods': hp.choice('sma_periods', [
        [8, 8, 8, 13],     # More responsive
        [10, 10, 10, 15],  # Traditional
        [5, 5, 5, 10],     # Highly responsive for volatile markets
        [15, 15, 15, 20]   # Less noise, smoother signals
    ]),
    
    # Signal line period
    'signal_period': hp.quniform('signal_period', 5, 15, 1),
    
    # KST weights for different ROC components
    'kst_weights': hp.choice('kst_weights', [
        [1, 2, 3, 4],      # Traditional ascending
        [1, 1.5, 2, 2.5],  # More balanced
        [1, 3, 5, 9],      # Emphasizing longer-term momentum
        [4, 3, 2, 1]       # Emphasizing shorter-term momentum
    ]),
    
    # Trading direction
    'long_only': hp.choice('long_only', [True, False]),
    
    # Risk management parameters
    'risk_params': {
        'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.1),  # Tighter for Indian markets
        'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),
        'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ])
    }
}

mcad_strat_search_space = {
    # MACD Parameters
    'fast': hp.quniform('fast', 8, 20, 1),       # Fast EMA period
    'slow': hp.quniform('slow', 21, 50, 1),      # Slow EMA period
    'smooth': hp.quniform('smooth', 5, 15, 1),   # Signal line smoothing period
    
    # Trading Mode
    'long_only': hp.choice('long_only', [True, False]),
    
    # Risk Management Parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.1),           # Stop loss percentage
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),       # Take profit percentage
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]),    # Trailing stop percentage
}

relative_vigor_index_strat_search_space = {
    # RVI calculation parameters
    'lookback': hp.quniform('lookback', 7, 21, 1),  # SMA period for RVI smoothing
    'signal_strength_window': hp.quniform('signal_strength_window', 10, 40, 2),  # Window for signal strength normalization
    
    # Trading mode
    'long_only': hp.choice('long_only', [True, False]),  # Whether to allow short positions
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.1),  # Stop loss percentage
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),  # Take profit percentage
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ])}

sma_crossover_strat_search_space = {
    # SMA Parameters
    'short_window': hp.quniform('short_window', 5, 30, 1),  # Short-term SMA window
    'long_window': hp.quniform('long_window', 30, 200, 5),  # Long-term SMA window
    
    # Risk Management Parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.1),  # Stop loss percentage
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),  # Take profit percentage
        'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]),  # Trailing stop percentage
    
    # Strategy Mode
    'long_only': hp.choice('long_only', [True, False])  # Indian markets allow short selling with restrictions
}

true_strength_index_strat_search_space = {
    # TSI calculation parameters
    'long_period': hp.quniform('long_period', 15, 45, 1),        # First smoothing period
    'short_period': hp.quniform('short_period', 5, 20, 1),       # Second smoothing period
    'signal_period': hp.quniform('signal_period', 5, 20, 1),     # Signal line period
    
    # Trading mode
    'long_only': hp.choice('long_only', [True, False]),          # True for long-only, False for long-short
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.1),   # Stop loss percentage
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15), # Take profit percentage
        'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]),  # Trailing stop percentage
}

adx_strat_search_space = {
    # ADX parameters
    'adx_lookback': hp.quniform('adx_lookback', 10, 30, 1),  # Traditional range is 14-21, expanded for Indian markets
    'adx_threshold': hp.quniform('adx_threshold', 20, 30, 1),  # ADX threshold for trend strength
    
    # Trading mode
    'long_only': hp.choice('long_only', [True, False]),  # Consider both long and short for complete evaluation
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.08),  # Indian markets can be volatile, especially small caps
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),  # Balanced risk-reward ratio
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]) # Trailing stop percentage
}

aroon_strat_search_space = {
    # Aroon indicator parameters
    'lookback': hp.quniform('lookback', 10, 50, 1),  # Period for Aroon calculation
    'aroon_up_threshold': hp.quniform('aroon_up_threshold', 60, 80, 1),  # Threshold for Aroon Up (default 70)
    'aroon_down_threshold': hp.quniform('aroon_down_threshold', 20, 40, 1),  # Threshold for Aroon Down (default 30)

    'signal_smoothing': hp.quniform('signal_smoothing', 1, 5, 1),  # Optional smoothing for signal strength
    
    # Trading parameters
    'long_only': hp.choice('long_only', [True, False]),  # Whether to allow short positions

    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.08),  # Suitable for Indian markets' volatility
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),  # Conservative profits for Indian market
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]) # Trailing stop percentage
}

choppiness_index_strat_search_space = {
    # Choppiness Index parameters
    'ci_period': hp.quniform('ci_period', 10, 30, 1),  # Shorter periods for higher volatility in Indian markets
    
    # MACD parameters
    'macd_fast': hp.quniform('macd_fast', 8, 16, 1),  # Standard is 12, but testing range for market adaptability
    'macd_slow': hp.quniform('macd_slow', 20, 35, 1),  # Standard is 26, but testing range for market adaptability
    'macd_smooth': hp.quniform('macd_smooth', 7, 12, 1),  # Standard is 9, modified for potential higher volatility
    
    # Strategy configuration
    'long_only': hp.choice('long_only', [True, False]),  # Test both long-only and long-short approaches
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.08),  # Higher volatility in Indian markets
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),  # Potentially higher returns in growth markets
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]) # Trailing stop percentage
}   

donchian_channel_strat_search_space = {
    # Channel parameters
    'lookback_period': hp.quniform('lookback_period', 15, 60, 1),  # Overall lookback period
    'entry_lookback': hp.quniform('entry_lookback', 15, 60, 1),    # Entry channel lookback
    'exit_lookback': hp.quniform('exit_lookback', 5, 30, 1),       # Exit channel lookback (typically shorter)
    
    # ATR filter parameters
    'use_atr_filter': hp.choice('use_atr_filter', [True, False]),  # Whether to use ATR filter
    'atr_period': hp.quniform('atr_period', 7, 21, 1),             # ATR calculation period
    'atr_threshold': hp.uniform('atr_threshold', 0.5, 2.0),        # ATR threshold multiplier
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.08),      # Stop loss percentage (tighter for Indian markets)
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),  # Take profit percentage
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]), # Trailing stop percentage

    # Strategy type
    'long_only': hp.choice('long_only', [True, False])  # Whether to allow short positions
}

ichimoku_cloud_strat_search_space = {
    # Ichimoku components parameters
    'tenkan_period': hp.quniform('tenkan_period', 7, 15, 1),             # Shorter periods for Indian market volatility
    'kijun_period': hp.quniform('kijun_period', 20, 40, 1),              # Adjusted for Indian market cycles
    'senkou_b_period': hp.quniform('senkou_b_period', 40, 70, 1),        # Wider range for different cap sizes
    'displacement': hp.quniform('displacement', 20, 35, 1),              # Adjusted displacement for local market dynamics
    
    # Signal generation flags
    'use_cloud_breakouts': hp.choice('use_cloud_breakouts', [True, False]),
    'use_tk_cross': hp.choice('use_tk_cross', [True, False]),
    'use_price_cross': hp.choice('use_price_cross', [True, False]),
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.08),            # Indian markets can be volatile, especially mid/small caps
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),        # Reasonable profit targets for Indian stocks
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]), # Trailing stop percentage

    # Strategy type
    'long_only': hp.choice('long_only', [True, False])                   # Test both long-only and long-short approaches
}

parabolic_sar_strat_search_space = {
    # PSAR core parameters
    'initial_af': hp.uniform('initial_af', 0.01, 0.05),
    'max_af': hp.uniform('max_af', 0.1, 0.3),
    'af_step': hp.uniform('af_step', 0.01, 0.05),
    
    # ATR filter parameters
    'use_atr_filter': hp.choice('use_atr_filter', [True, False]),
    'atr_period': hp.quniform('atr_period', 10, 21, 1),
    'atr_threshold': hp.uniform('atr_threshold', 0.8, 1.5),
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.08),
    'take_profit_pct': hp.uniform('take_profit_pct', 0.05, 0.15),
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]), # Trailing stop percentage

    # Strategy behavior
    'long_only': hp.choice('long_only', [True, False])
}

atr_trailing_stops_strat_search_space = {
    # ATR calculation parameters
    'atr_period': hp.quniform('atr_period', 5, 30, 1),  # Shorter for more responsive, longer for more stability
    'atr_multiplier': hp.uniform('atr_multiplier', 1.5, 5.0),  # Controls stop distance
    
    # Trend detection parameters
    'trend_period': hp.quniform('trend_period', 10, 50, 1),  # SMA period for trend identification
    
    # Position control parameters
    'long_only': hp.choice('long_only', [True, False]),  # True for long-only, False for both long and short
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.08),  # Fixed stop-loss (adapted for Indian markets)
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),  # Take-profit levels
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]), # Trailing stop percentage
}

bollinger_bands_strat_search_space = {
    # Bollinger Bands parameters
    'window': hp.quniform('window', 15, 40, 1),  # SMA and STD calculation period
    'std_dev': hp.uniform('std_dev', 1.5, 3.0),  # Standard deviation multiplier
    
    # Trading mode
    'long_only': hp.choice('long_only', [True, False]),  # Whether to allow short positions
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.08),  # Stop loss percentage
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),  # Take profit percentage
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]),
}

garch_strat_search_space = {
    # GARCH model parameters
    'window_size': hp.quniform('window_size', 50, 200, 5),  # Calibration window
    'forecast_horizon': hp.choice('forecast_horizon', [1, 2, 3]),  # Days ahead to forecast
    'p': hp.choice('p', [1, 2]),  # ARCH order
    'q': hp.choice('q', [1, 2]),  # GARCH order
    'return_type': hp.choice('return_type', ['log', 'simple']),
    
    # Signal generation parameters
    'vol_threshold': hp.uniform('vol_threshold', 0.05, 0.3),  # Sensitivity to volatility changes
    
    # Trading parameters
    'long_only': hp.choice('long_only', [True, False]),  # Indian markets have uptick rule restrictions
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.02, 0.08),  # Lower range due to high volatility in Indian stocks
    'take_profit_pct': hp.uniform('take_profit_pct', 0.03, 0.15),  # Adjusted for Indian markets
    'trailing_stop_pct': hp.uniform('trailing_stop_pct', 0.0, 0.05),  # Optional trailing stop
}

keltner_channel_strat_search_space = {
    # Keltner Channel parameters
    'kc_span': hp.quniform('kc_span', 15, 35, 1),  # EMA period for channel middle
    'atr_span': hp.quniform('atr_span', 8, 20, 1),  # EMA period for ATR
    'multiplier': hp.uniform('multiplier', 1.5, 3.5),  # ATR multiplier for channel width
    
    # Trading direction
    'long_only': hp.choice('long_only', [True, False]),  # True for long-only, False for long-short
    
    # Risk management parameters
    'stop_loss': hp.uniform('stop_loss', 0.02, 0.08),  # Stop loss percentage
    'take_profit': hp.uniform('take_profit', 0.04, 0.15),  # Take profit percentage
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]),
}

supertrend_strat_search_space = {
    # Supertrend indicator parameters
    'lookback': hp.quniform('lookback', 7, 21, 1),  # ATR period (7-21 days)
    'multiplier': hp.uniform('multiplier', 1.5, 4.5),  # ATR multiplier (1.5-4.5)
    'long_only': hp.choice('long_only', [True, False]),  # Long-only or long-short
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.08),  # Stop-loss (2-7%)
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),  # Take-profit (4-15%) # 0.05-0.2%
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]),
}

volatality_squeeze_strat_search_space = {
    # Bollinger Bands parameters
    'bb_period': hp.quniform('bb_period', 15, 40, 1),
    'bb_std': hp.uniform('bb_std', 1.5, 3.0),
    
    # Keltner Channel parameters
    'kc_period': hp.quniform('kc_period', 15, 40, 1),
    'kc_atr_period': hp.quniform('kc_atr_period', 10, 30, 1),
    'kc_multiplier': hp.uniform('kc_multiplier', 1.2, 2.5),
    
    # Momentum parameters
    'momentum_period': hp.quniform('momentum_period', 8, 20, 1),
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.08),
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]),

    # Strategy options
    'long_only': hp.choice('long_only', [True, False])
}

volatality_breakout_strat_search_space = {
    # Volatility calculation parameters
    'lookback_period': hp.quniform('lookback_period', 10, 50, 1),
    'volatility_multiplier': hp.uniform('volatility_multiplier', 1.0, 3.0),
    'use_atr': hp.choice('use_atr', [True, False]),
    'atr_period': hp.quniform('atr_period', 7, 21, 1),
    
    # Trading constraints
    'long_only': hp.choice('long_only', [True, False]),
    
    # Risk management parameters
    'stop_loss_pct': hp.uniform('stop_loss_pct', 0.04, 0.08),
    'take_profit_pct': hp.uniform('take_profit_pct', 0.06, 0.15),
    'trailing_stop_pct': hp.choice('trailing_stop_pct', [
        0.0, # Explicitly include 0 (disabled)
        hp.uniform('trailing_stop_pct_val', 0.04, 0.08) # Enable TSL within a range if chosen
    ]),
}