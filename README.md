# Project Overview

This project implements a comprehensive trading system with modular components for data management, strategy execution, and strategy optimization. It features systematic trading strategies across multiple asset classes, with robust backtesting and optimization capabilities.

## Source Code Documentation

### Core Components:

1. **Data Collection (`src/collectors/`)**:
   - `base_collector.py`: Abstract base class defining the interface for all data collectors
   - `price_collector.py`: Handles collection of daily market price data from various sources
   - `info_collector.py`: Retrieves and stores company information and metadata
   - `statements_collector.py`: Manages collection of financial statements and fundamental data
   - `data_collector.py`: Orchestrates the overall data collection process with error handling

2. **Strategy Implementation (`src/strategies/`)**:
   - **Base Classes**:
     - `base_strat.py`: Core abstract class defining the strategy interface
     - `risk_management.py`: Risk control and position sizing functionality
   
   - **Strategy Categories**:
     - **Value Strategies**: Graham's Defensive, Acquirer's Multiple, Deep Value, Piotroski F-Score
     - **Momentum Strategies**: SMA Crossover, MACD, KST (Know Sure Thing)
     - **Mean Reversion**: Statistical arbitrage and mean reversion techniques
     - **Breakout Strategies**: Various breakout detection and trading methods
     - **Advanced Strategies**: Enhanced market pressure and ML-based approaches

3. **Optimization Framework (`src/optimizer/`)**:
   - `strategy_optimizer.py`: Main optimization engine with hyperparameter tuning
   - `performance_evaluator.py`: Strategy performance metrics calculation
   - `sensitivity_analyzer.py`: Parameter sensitivity analysis tools
   - `search_space.py`: Hyperparameter search space definitions
   - **Ticker-Level Optimization**: Specialized optimization at individual asset level

4. **Database Management (`src/database/`)**:
   - `config.py`: Database configuration and connection settings
   - `engine.py`: SQLAlchemy engine setup and management
   - `remove_duplicates.py`: Data cleaning and deduplication utilities
   - `utils.py`: General database helper functions

## Directory Structure

├── **src/**  
│   ├── **config/**  
│   │   ├── `settings.py` – (Empty) Placeholder for application-wide settings.  
│   │   ├── `logging_config.py` – Logging configuration utility function.  
│   │   └── `__init__.py` – (Empty) Config module initializer.  
│   │  
│   ├── **database/**  
│   │   ├── `config.py` – Database configuration dataclass and default settings.  
│   │   ├── `engine.py` – Creates and manages SQLAlchemy database engines.  
│   │   ├── `remove_duplicates.py` – Functions to identify and remove duplicate records.  
│   │   ├── `utils.py` – Utility for running SQL queries and returning pandas DataFrames.  
│   │   └── `__init__.py` – (Empty) Database module initializer.  
│   │  
│   ├── **collectors/**  
│   │   ├── `base_collector.py` – Abstract base class for all data collectors.  
│   │   ├── `data_collector.py` – Orchestrates the overall data collection process.  
│   │   ├── `info_collector.py` – Collects company information using yfinance.  
│   │   ├── `price_collector.py` – Collects and stores daily market price data.  
│   │   ├── `statements_collector.py` – Collects financial statements and fundamentals.  
│   │   └── `__init__.py` – (Empty) Collectors module initializer.  
│   │  
│   ├── **optimizer/**  
│   │   ├── `performance_evaluator.py` – Portfolio and single-asset performance metrics.  
│   │   ├── `search_space.py` – Hyperparameter search space definitions for strategies.  
│   │   ├── `sensitivity_analyzer.py` – Sensitivity analysis for strategy parameters.  
│   │   ├── `strategy_optimizer.py` – Main engine for optimizing trading strategies.  
│   │   ├── **ticker_level/**  
│   │   │   ├── `performance_evaluator.py` – Ticker-level performance metrics.  
│   │   │   ├── `strategy_optimizer.py` – Ticker-level strategy optimization.  
│   │   │   ├── `sensitivity_analyzer.py` – Ticker-level sensitivity analysis.  
│   │   │   └── `usage_example.py` – Example usage of ticker-level optimization modules.  
│   │  
│   ├── **strategies/**  
│   │   ├── `base_strat.py` – Abstract base class for all trading strategies.  
│   │   ├── `risk_management.py` – Risk management and position sizing logic.  
│   │   ├── `buy_and_hold_baseline.py` – Buy-and-hold benchmark strategy.  
│   │   ├── **advanced/**  
│   │   │   ├── `enhanced_market_pressure_strat.py` – Market pressure indicator strategy.  
│   │   │   ├── `garchx_strat.py` – GARCH-X volatility modeling and trading strategy.  
│   │   │   ├── `pairs_trading_strat.py` – Statistical arbitrage pairs trading.  
│   │   │   └── `prophet_momentum_strat.py` – Prophet-based momentum strategy.  
│   │   ├── **breakout/**  
│   │   │   ├── `cup_and_handle_strat.py` – Cup-and-handle breakout pattern strategy.  
│   │   │   └── `triangle_breakout_strat.py` – Triangle breakout trading strategy.  
│   │   ├── **mean_reversion/**  
│   │   │   ├── `cci_oscillator_strat.py` – Commodity Channel Index mean reversion.  
│   │   │   ├── `disparity_index_strat.py` – Disparity Index mean reversion strategy.  
│   │   │   ├── `relative_strength_index_strat.py` – RSI-based mean reversion.  
│   │   │   ├── `stochastic_oscillator_strat.py` – Stochastic Oscillator mean reversion.  
│   │   │   └── `williams_percent_r_start.py` – Williams %R mean reversion strategy.  
│   │   ├── **momentum/**  
│   │   │   ├── `awesome_oscillator_strat.py` – Awesome Oscillator momentum strategy.  
│   │   │   ├── `coppock_curve_strat.py` – Coppock Curve momentum strategy.  
│   │   │   ├── `know_sure_thing_strat.py` – Know Sure Thing (KST) momentum strategy.  
│   │   │   ├── `macd_strat.py` – MACD momentum strategy.  
│   │   │   ├── `relative_vigor_index_strat.py` – Relative Vigor Index momentum strategy.  
│   │   │   ├── `sma_crossover_strat.py` – SMA crossover momentum strategy.  
│   │   │   └── `true_strength_index_strat.py` – True Strength Index momentum strategy.  
│   │   ├── **trend_following/**  
│   │   │   ├── `adx_strat.py` – ADX trend following strategy.  
│   │   │   ├── `aroon_strat.py` – Aroon trend following strategy.  
│   │   │   ├── `choppiness_index.py` – Choppiness Index trend following strategy.  
│   │   │   ├── `donchian_channel_strat.py` – Donchian Channel breakout strategy.  
│   │   │   ├── `ichimoku_cloud_strat.py` – Ichimoku Cloud trend following strategy.  
│   │   │   └── `parabolic_sar_strat.py` – Parabolic SAR trend following strategy.  
│   │   ├── **value/**  
│   │   │   ├── `acquirer_multiple.py` – Acquirer’s Multiple value strategy.  
│   │   │   ├── `buffets_qarp_strat.py` – Buffett’s QARP value strategy.  
│   │   │   ├── `deep_value_strat.py` – Deep value investing strategy.  
│   │   │   ├── `dividend_yeild_start.py` – Dividend yield value strategy.  
│   │   │   ├── `ev_ebidta_private_equity_strat.py` – EV/EBITDA private equity screener.  
│   │   │   ├── `grahams_defensive_strat.py` – Graham’s defensive value strategy.  
│   │   │   ├── `greenblatts_earnings_yeild_strat.py` – Greenblatt’s earnings yield strategy.  
│   │   │   ├── `insider_activist_strat.py` – Insider/activist investor screener.  
│   │   │   ├── `low_price_sales_start.py` – Low price-to-sales value strategy.  
│   │   │   ├── `magic_formula_start.py` – Magic Formula value strategy.  
│   │   │   ├── `montiers_c_score_strat.py` – Montier’s C-Score value trap screener.  
│   │   │   ├── `net_net_start.py` – Graham’s Net-Net value strategy.  
│   │   │   ├── `oshaugnessy_tending_value_strat.py` – O'Shaughnessy trending value strategy.  
│   │   │   ├── `piotroski_f_score_strat.py` – Piotroski F-Score value strategy.  
│   │   │   ├── `scholls_pb_strat.py` – Schloss’ Low P/B value strategy.  
│   │   │   ├── `small_cap_value_strat.py` – Small-cap value screener strategy.  
│   │   │   └── **unified_class/components/**  
│   │   │       ├── `config.py` – Configurations for unified value strategy.  
│   │   │       ├── `data_manager.py` – Data management for unified value strategy.  
│   │   │       ├── `financial_metrics.py` – Financial metrics for value strategies.  
│   │   │       └── `value_strategy.py` – Unified value strategy logic and interface.  
│   │   │   └── `value_screener_strat.py` – Value screener using unified components.  
│   │   ├── **volatality/**  
│   │   │   ├── `atr_trailing_stops_strat.py` – ATR trailing stops volatility strategy.  
│   │   │   ├── `bollinger_bands_strat.py` – Bollinger Bands volatility strategy.  
│   │   │   ├── `garch_strat.py` – GARCH volatility modeling strategy.  
│   │   │   ├── `keltner_channel_strat.py` – Keltner Channel volatility strategy.  
│   │   │   ├── `supertrend_strat.py` – Supertrend volatility strategy.  
│   │   │   ├── `volatality_squeeze_strat.py` – Volatility squeeze strategy.  
│   │   │   └── `volatility_breakout_strat.py` – Volatility breakout strategy.
```

