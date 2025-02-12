trading_system/
│
├── data/
│   ├── tickers.csv                # File containing the list of tickers
│
├── src/
│   ├── database/
│   │   ├── __init__.py            # Package initialization
│   │   ├── config.py              # DatabaseConfig class
│   │   ├── engine.py              # Database engine creation utilities
│   │   └── utils.py               # Database helper functions
│   │
│   ├── collectors/
│   │   ├── __init__.py            # Package initialization
│   │   ├── financial_data.py      # Financial data collection logic
│   │   ├── price_data.py          # Price data collection logic
│   │   └── company_info.py        # Company info data collection logic
│   │
│   ├── config/
│   │   ├── __init__.py            # Package initialization
│   │   ├── logging_config.py      # Logging configuration
│   │   └── settings.py            # Global configurations
│   │
│   └── main.py                    # Entry point for the application
│
├── tests/
│   ├── test_financial_data.py     # Unit tests for financial data collection
│   ├── test_price_data.py         # Unit tests for price data collection
│   ├── test_company_info.py       # Unit tests for company info collection
│   └── conftest.py                # Shared test fixtures
│
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Poetry configuration
├── README.md                      # Project overview and usage instructions
└── .env                           # Environment variables
