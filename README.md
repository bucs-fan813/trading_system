# Project Overview

This project implements a trading system with comprehensive modules for data management, strategy execution, and strategy optimization. The following sections detail the project structure and key components.

## Directory Structure

├── **data/**  
│   ├── `ticker.xlsx` – Excel file with ticker details  
│   └── `trading_system.db` – Main SQLite database  
│
├── **src/**  
│   ├── **config/**  
│   │   ├── `settings.py` – Application settings and configurations  
│   │   ├── `logging_config.py` – Logging configuration  
│   │   └── `__init__.py`  
│   │  
│   ├── **database/**  
│   │   ├── `config.py` – Database configuration classes  
│   │   ├── `engine.py` – Functions to create database engines  
│   │   ├── `remove_duplicates.py` – Utilities to clean data  
│   │   ├── `utils.py` – Database helper functions  
│   │   └── `__init__.py`  
│   │  
│   ├── **collectors/**  
│   │   ├── `base_collector.py` – Base class for data collectors  
│   │   ├── `price_collector.py` – Collects daily price data  
│   │   ├── `info_collector.py` – Gathers company info  
│   │   ├── `data_collector.py` – Generic data collection utilities  
│   │   ├── `statements_collector.py` – Collects financial statements  
│   │   └── `__init__.py`  
│   │  
│   ├── **optimizer/**  
│   │   ├── `performance_evaluator.py` – Evaluates strategy performance  
│   │   ├── `search_space.py` – Defines search spaces for optimization parameters  
│   │   ├── `sensitivity_analyzer.py` – Analyzes strategy sensitivity to parameters  
│   │   ├── `strategy_optimizer.py` – Optimizes trading strategies  
│   │   ├── **ticker_level/** – (Submodules for ticker-level optimization)  
│   │   └── `__pycache__/` – Python bytecode cache  
│   │  
│   ├── **strategies/**  
│   │   ├── `base_strat.py` – Base class for strategies  
│   │   ├── `risk_management.py` – Handles risk control strategies  
│   │   ├── **advanced/** – Advanced strategies  
|   |   |   ├── `enhanced_market_Pressure_strat.py` – 
│   │   ├── **breakout/** – Breakout strategies  
│   │   ├── **mean_reversion/** – Mean reversion strategies  
│   │   ├── **momentum/** – Momentum strategies  
│   │   ├── **value/**  
│   │   │   ├── `grahams_defensive_strat.py` – Graham’s defensive strategy  
│   │   │   ├── `acquirer_multiple.py` – Acquirer’s multiple strategy  
│   │   │   ├── `deep_value_strat.py` – Deep value strategy  
│   │   │   ├── `piotroski_f_score_strat.py` – Piotroski F-Score strategy  
│   │   │   └── **unified_class/**  
│   │   │       ├── `value_strategy.py`, `config.py`, `data_manager.py`, `financial_metrics.py` – Unified value strategy components  
│   │   └── `__pycache__/` – Python bytecode cache  
│   │  
│   └── **utils/**  
│       └── `file_utils.py` – File management utilities  
│
├── **tests/**  
│   ├── `__init__.py`  
│   ├── `Claude_testCase.py`  
│   ├── **collectors/**  
│   ├── **database/**  
│   ├── `test_config.py`  
│   ├── `test_engine.py`  
│   ├── `test_base_collector.py`  
│   ├── `test_statements_collector.py`  
│   ├── `test_info_collector.py`  
│   ├── `test_price_collector.py`  
│   ├── `e2e_test.py`  
│   └── Additional tests  
│
├── **utils/**  
│   └── `file_utils.py` – File management utilities  
│
├── Top-Level Files  
│   ├── `__init__.py` – Package initialization  
│   ├── `.env` – Environment variable configurations  
│   ├── `requirements.txt` – Python dependencies  
│   ├── `.gitignore` – Git exclusion file  
│   ├── `README.md` – Project documentation  
│   ├── `main.py` – Application entry point  
│   ├── `poetry.lock` & `pyproject.toml` – Poetry project configuration  

## Additional Notes

- The project is managed using Poetry for dependency management and builds.  
- Configuration and logging are modularized under the **config** directory, offering flexibility in environment setups.  
- The **database** module not only handles connections but includes utilities for data consistency (e.g., removal of duplicates).  
- Data collectors fetch various market and financial data to populate the databases, enabling robust strategy execution.  
- A wide array of trading strategies is provided, covering value, momentum, breakout, volatility, and mean reversion techniques.  
- The **optimizer** module facilitates strategy optimization through performance evaluation, parameter tuning, and sensitivity analysis.  
- Extensive tests ensure the reliability of data collection, database operations, and the correct functioning of trading strategies.

