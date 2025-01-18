algo_trading/
│
├── data/
│   ├── tickers.csv                # File with all ticker names of interest
│   └── historical_data/           # Folder for storing raw data (optional if using a database)
│
├── src/
│   ├── database/
│   │   ├── __init__.py            # Package initialization
│   │   ├── config.py              # Database configuration
│   │   ├── engine.py              # Engine creation and connection utilities
│   │   └── utils.py               # Common database helper functions
│   │
│   ├── collectors/
│   │   ├── __init__.py            # Package initialization
│   │   ├── financial_data.py      # Financial data collection functions
│   │   └── price_data.py          # Price data collection functions
│   │
│   ├── strategies/
│   │   ├── __init__.py            # Package initialization
│   │   ├── base.py                # Base class for all strategies
│   │   ├── strategy_family_1.py   # Strategies under a specific family
│   │   └── strategy_family_2.py   # Another strategy family
│   │
│   ├── portfolio/
│   │   ├── __init__.py            # Package initialization
│   │   ├── optimizer.py           # Money allocation optimizer
│   │   ├── review.py              # Portfolio review and adjustment logic
│   │   └── utils.py               # Helper functions for portfolio management
│   │
│   ├── config.py                  # Global configurations
│   ├── logging_config.py          # Centralized logging configuration
│   └── main.py                    # Entry point of the application
│
├── tests/
│   ├── test_collectors.py         # Unit tests for data collection modules
│   ├── test_strategies.py         # Unit tests for strategy modules
│   ├── test_portfolio.py          # Unit tests for portfolio modules
│   └── conftest.py                # Test fixtures for the test suite
│
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Poetry configuration (includes dependencies and build info)
├── README.md                      # Project overview and usage instructions
└── .env                           # Environment variables (e.g., database credentials)
