# Project Overview

This project implements a trading system with comprehensive modules for data management, collection, and strategy execution. The following sections detail the project structure and key components.

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
│   ├── **strategies/**  
│   │   ├── **value/**  
│   │   │   ├── `grahams_defensive_strat.py` – Graham’s defensive strategy  
│   │   │   ├── `acquirer_multiple.py`, `deep_value_strat.py`  
│   │   │   ├── `piotroski_f_score_strat.py` & other value strategies  
│   │   │   └── **unified_class/**  
│   │   │       ├── Components including `value_strategy.py`, `config.py`, `data_manager.py`, and `financial_metrics.py`  
│   │   │  
│   │   ├── **momentum/**  
│   │   │   ├── `coppock_curve_strat.py`, `know_sure_thing_strat.py`, `mcad_strat.py`  
│   │   │   ├── `true_strength_index_strat.py`, `relative_vigor_index_strat.py`  
│   │   │   ├── `awesome_oscillator_strat.py`, `sma_crossover_strat.py`  
│   │   │   └── `strats_explanation.md` – Documentation of momentum strategies  
│   │   │  
│   │   ├── **breakout/**  
│   │   │   ├── `cup_and_handle_strat.py`, `volume_breakout_strat.py`, `triangle_breakout_strat.py`  
│   │   │  
│   │   ├── **volatality/**  
│   │   │   ├── `bollinger_bands_strat.py`, `supertrend_strat.py`, `garch_strat.py`  
│   │   │   ├── `volatility_breakout_strat.py`, `volatality_squeeze_strat.py`  
│   │   │   ├── `atr_trailing_stops_strat.py`, `keltner_channel_strat.py`  
│   │   │  
│   │   ├── **mean_reversion/**  
│   │   │   ├── `disparity_index_strat.py`, `stochastic_oscillator_strat.py`  
│   │   │   ├── `relative_strength_index_strat.py`, `williams_percent_r_start.py`  
│   │   │   └── `cci_oscillator_strat.py`  
│   │   │  
│   │   ├── **advanced/**  
│   │   │   ├── `adx_rsi_strat.py`, `bollinger_stochastic_strat.py`, `bollinger_bands_kcr.py`  
│   │   │  
│   │   ├── **risk_management.py** – Handles risk control strategies  
│   │   └── **base_strat.py** – Base class for strategies  

│   └── **utils/** – Utility functions for general use  

├── **tests/**  
│   ├── Unit tests for **database/** and **collectors/** modules such as:  
│   │   ├── `test_config.py`  
│   │   ├── `test_engine.py`  
│   │   ├── `test_base_collector.py`  
│   │   ├── `test_statements_collector.py`  
│   │   ├── `test_info_collector.py`  
│   │   └── `test_price_collector.py`  
│   ├── End-to-end tests like `e2e_test.py`  
│   └── Additional tests such as `Claude_testCase.py`  

├── **utils/**  
│   └── `file_utils.py` – File management utilities  

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
- Extensive tests ensure the reliability of data collection, database operations, and the correct functioning of trading strategies.

This detailed structure should help understand and extend the functionality of the trading system.

