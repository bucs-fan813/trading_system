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
│   │   ├── `settings.py` – Defines application-wide settings and configuration parameters.  
│   │   ├── `logging_config.py` – Sets up logging format, handlers, and log file management.  
│   │   └── `__init__.py` – Package initialization for config module.  
│   │  
│   ├── **database/**  
│   │   ├── `config.py` – Contains database connection and ORM configuration classes.  
│   │   ├── `engine.py` – Creates and manages SQLAlchemy database engines.  
│   │   ├── `remove_duplicates.py` – Provides functions to identify and remove duplicate records from the database.  
│   │   ├── `utils.py` – Utility functions for database operations and queries.  
│   │   └── `__init__.py` – Package initialization for database module.  
│   │  
│   ├── **collectors/**  
│   │   ├── `base_collector.py` – Abstract base class defining the interface for all data collectors.  
│   │   ├── `price_collector.py` – Collects and stores daily market price data from external sources.  
│   │   ├── `info_collector.py` – Retrieves and manages company information and metadata.  
│   │   ├── `statements_collector.py` – Collects and processes company financial statements and fundamentals.  
│   │   ├── `data_collector.py` – Orchestrates and manages the overall data collection workflow.  
│   │   └── `__init__.py` – Package initialization for collectors module.  
│   │  
│   ├── **optimizer/**  
│   │   ├── `performance_evaluator.py` – Computes and reports strategy performance metrics.  
│   │   ├── `search_space.py` – Defines hyperparameter search spaces for strategy optimization.  
│   │   ├── `sensitivity_analyzer.py` – Analyzes sensitivity of strategies to parameter changes.  
│   │   ├── `strategy_optimizer.py` – Main engine for optimizing trading strategies and hyperparameters.  
│   │   ├── **ticker_level/**  
│   │   │   ├── `performance_evaluator.py` – Evaluates performance metrics at the individual ticker level.  
│   │   │   ├── `strategy_optimizer.py` – Optimizes strategies for specific tickers.  
│   │   │   ├── `sensitivity_analyzer.py` – Performs sensitivity analysis for single-ticker strategies.  
│   │   │   └── `usage_example.py` – Example usage of ticker-level optimization modules.  
│   │   └── `__pycache__/` – Python bytecode cache.  
│   │  
│   ├── **strategies/**  
│   │   ├── `base_strat.py` – Abstract base class for all trading strategies.  
│   │   ├── `risk_management.py` – Implements risk management and position sizing logic.  
│   │   ├── `buy_and_hold_baseline.py` – Simple buy-and-hold baseline strategy implementation.  
│   │   ├── **advanced/**  
│   │   │   ├── `enhanced_market_pressure_strat.py` – Strategy based on market pressure indicators.  
│   │   │   ├── `garchx_strat.py` – GARCH-X volatility modeling and trading strategy.  
│   │   │   ├── `pairs_trading_strat.py` – Implements pairs trading using statistical arbitrage.  
│   │   │   └── `prophet_momentum_strat.py` – Momentum strategy using Prophet time series forecasting.  
│   │   ├── **breakout/**  
│   │   │   ├── `cup_and_handle_strat.py` – Detects and trades on cup-and-handle breakout patterns.  
│   │   │   └── `triangle_breakout_strat.py` – Implements triangle breakout trading strategy.  
│   │   ├── **mean_reversion/**  
│   │   │   └── ... (mean reversion strategy files)  
│   │   ├── **momentum/**  
│   │   │   ├── `awesome_oscillator_strat.py` – Momentum strategy using the Awesome Oscillator indicator.  
│   │   │   ├── `know_sure_thing_strat.py` – Implements the Know Sure Thing (KST) momentum strategy.  
│   │   │   └── `true_strength_index_strat.py` – Uses the True Strength Index for momentum trading.  
│   │   ├── **value/**  
│   │   │   ├── `acquirer_multiple.py` – Value strategy based on the Acquirer’s Multiple.  
│   │   │   ├── `deep_value_strat.py` – Deep value investing strategy implementation.  
│   │   │   ├── `grahams_defensive_strat.py` – Graham’s defensive value investing strategy.  
│   │   │   ├── `magic_formula_start.py` – Implements the Magic Formula value strategy.  
│   │   │   ├── `montiers_c_score_strat.py` – Montier’s C-Score value screening strategy.  
│   │   │   ├── `net_net_start.py` – Net-net value investing strategy.  
│   │   │   ├── `piotroski_f_score_strat.py` – Piotroski F-Score value strategy.  
│   │   │   ├── `scholls_pb_strat.py` – Scholl’s Price-to-Book value strategy.  
│   │   │   ├── **unified_class/components/**  
│   │   │   │   ├── `value_strategy.py` – Unified value strategy logic and interface.  
│   │   │   │   ├── `config.py` – Configuration for unified value strategy.  
│   │   │   │   ├── `data_manager.py` – Data management for unified value strategy.  
│   │   │   │   └── `financial_metrics.py` – Financial metrics calculations for value strategies.  
│   │   │   └── `value_screener_strat.py` – Value screener strategy using unified components.  
│   │   ├── **volatality/**  
│   │   │   └── `supertrend_strat.py` – Volatility-based Supertrend trading strategy.  
│   │   └── `__pycache__/` – Python bytecode cache.  
│   │  
│   └── **utils/**  
│       └── `file_utils.py` – Utility functions for file and directory management.  
│
├── **data/**  
│   ├── `ticker.xlsx` – Excel file containing ticker details and metadata.  
│   └── `trading_system.db` – Main SQLite database for storing market and analysis data.  
│
├── **notebooks/**  
│   ├── `hyperparameter_tuning_mean_reversion.ipynb` – Jupyter notebook for mean reversion strategy optimization.  
│   ├── `hyperparameter_tuning_momentum.ipynb` – Optimization notebook for momentum strategies.  
│   ├── `hyperparameter_tuning_trend.ipynb` – Trend strategy parameter tuning notebook.  
│   ├── `hyperparameter_tuning_volatality.ipynb` – Volatility strategy optimization notebook.  
│   ├── `test_output.ipynb` – Notebook for testing and result visualization.  
│   └── **mlruns/** – MLflow tracking directory for experiment logging.  
│
├── **optimization_artefact/**  
│   ├── `benchmark_bnh_metrics_*.csv` – Buy-and-hold benchmark performance metrics.  
│   ├── Multiple strategy optimization results, each including:  
│   │   ├── `*_bestparams.json` – Best parameters found during optimization.  
│   │   ├── `*_param_history.csv` – Parameter evolution during optimization.  
│   │   └── `*_portfolio_performance.csv` – Strategy performance metrics.  
│   └── (Covers ADX, Aroon, Awesome Oscillator, CCI, Choppiness Index, and other strategies)  
│
├── **tests/**  
│   ├── `__init__.py` – Test package initialization.  
│   ├── `Claude_testCase.py` – Test cases for core functionality.  
│   ├── `test_config.py` – Configuration tests.  
│   ├── `test_engine.py` – Database engine tests.  
│   ├── `test_base_collector.py` – Base collector class tests.  
│   ├── `test_statements_collector.py` – Financial statements collector tests.  
│   ├── `test_info_collector.py` – Company info collector tests.  
│   ├── `test_price_collector.py` – Price data collector tests.  
│   └── `e2e_test.py` – End-to-end system tests.  
│
├── Top-Level Files  
│   ├── `__init__.py` – Package initialization.  
│   ├── `.env` – Environment variable configurations.  
│   ├── `requirements.txt` – Python package dependencies.  
│   ├── `poetry.lock` & `pyproject.toml` – Poetry project configuration and lock files.  
│   ├── `README.md` – Project documentation.  
│   └── `main.py` – Application entry point.

## Additional Notes

- The project is managed using Poetry for dependency management and builds.  
- Configuration and logging are modularized under the **config** directory, offering flexibility in environment setups.  
- The **database** module handles connections and includes utilities for data consistency.  
- Data collectors fetch various market and financial data to populate the databases.  
- A wide array of trading strategies is provided, covering value, momentum, breakout, volatility, and mean reversion techniques.  
- The **optimizer** module facilitates strategy optimization through performance evaluation, parameter tuning, and sensitivity analysis.  
- **Jupyter notebooks** in the notebooks/ directory provide interactive environments for hyperparameter tuning across different strategy types.  
- **MLflow tracking** (mlruns/) is used to log and monitor optimization experiments.  
- The **optimization_artefact/** directory stores optimization results, including best parameters, parameter evolution history, and performance metrics for each strategy.  
- Extensive tests ensure the reliability of data collection, database operations, and strategy execution.

