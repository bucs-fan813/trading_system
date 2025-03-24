# trading_system/src/strategies/value_strategy/config.py

"""Configuration settings for value strategy implementation."""

# Strategy parameter presets
INDIAN_MARKET_PARAMS = {
    # Financial filters
    'min_market_cap': 500,  # 500 crores (~$60M)
    'max_market_cap': None,  # No upper limit
    'min_liquidity': 100000,  # Minimum daily volume
    
    # Sector exclusions
    'exclude_sectors': ['Financial Services', 'Banks'],
    
    # Strategy weights
    'weight_magic_formula': 1.0,
    'weight_acquirers_multiple': 1.0,
    'weight_deep_value': 1.0,
    'weight_graham': 1.0,
    'weight_piotroski': 1.0,
    'weight_dividend': 0.5,  # Lower weight as not all companies pay dividends in India
    
    # Lookback period for historical analysis
    'lookback_years': 5
}

# Small-cap focused strategy
SMALL_CAP_PARAMS = {
    'min_market_cap': 100,  # 100 crores (~$12M)
    'max_market_cap': 5000,  # 5000 crores (~$600M)
    'min_liquidity': 50000,  # Lower liquidity requirement
    'exclude_sectors': ['Financial Services', 'Banks'],
    'weight_magic_formula': 1.0,
    'weight_acquirers_multiple': 1.2,  # Higher weight for acquirer's multiple
    'weight_deep_value': 1.2,  # Higher weight for deep value
    'weight_graham': 0.8,  # Lower weight as small caps may not meet all Graham criteria
    'weight_piotroski': 1.2,  # Higher weight for operational efficiency
    'weight_dividend': 0.3,  # Lower weight as small caps often reinvest rather than pay dividends
    'lookback_years': 3  # Shorter lookback for high-growth small caps
}

# Dividend-focused strategy
DIVIDEND_INCOME_PARAMS = {
    'min_market_cap': 1000,  # 1000 crores (~$120M)
    'max_market_cap': None,
    'min_liquidity': 100000,
    'exclude_sectors': [],  # Include all sectors as financials often pay good dividends
    'weight_magic_formula': 0.8,
    'weight_acquirers_multiple': 0.8,
    'weight_deep_value': 1.0,
    'weight_graham': 1.2,  # Higher weight for stability
    'weight_piotroski': 1.0,
    'weight_dividend': 2.0,  # Much higher weight on dividend metrics
    'lookback_years': 7  # Longer lookback to find consistent dividend payers
}