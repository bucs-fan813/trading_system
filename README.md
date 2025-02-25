trading_system/
│
├── data/
│   ├── ticker.xlsx                     # Excel file containing ticker information
│   ├── trading_system.db               # SQLite main database
│   ├── trading_system.db-shm           # SQLite database shared memory file
│   ├── trading_system.db-wal           # SQLite database write-ahead log file
│   ├── trading_system.duckdb           # DuckDB database file only to be used in future
│
├── src/
│   ├── database/
│   │   ├── __init__.py                 # Package initialization
│   │   ├── config.py                   # DatabaseConfig class
│   │   ├── engine.py                   # Database engine creation utilities
│   │   └── utils.py                    # Database helper functions
│   │
│   ├── collectors/
│   │   ├── __init__.py                 # Package initialization
│   │   ├── base_collector.py           # Base Class for data colectors to fetch data from yfinance and store in DB
│   │   ├── price_collector.py          # Collects Daily Price data and stores in DB
│   │   ├── info_collector.py           # Collects company info data and stores in DB
│   │   ├── statements_collector.py     # Collects cashflow, income statements and baance sheet data and stores in DB
│   │   └── company_info.py             # Company info data collection logic
│   │
│   ├── config/                         # Configuration files
│   │
│   ├── portfolio/                      # Portfolio management logic
│   │
│   ├── strategies/                     # Trading strategies
│   │
│   └── utils/                          # Utility functions
│
├── tests/
│   ├── __init__.py                     # Package initialization
│   ├── collectors/                     # Tests for collectors
│   ├── database/                       # Tests for database
│
├── .env                                # Environment variables
├── .gitignore                          # Git ignore file
├── main.py                             # Main application script
├── poetry.lock                         # Poetry lock file
├── pyproject.toml                      # Poetry project file
├── README.md                           # Project README file
├── requirements.txt                    # Python dependencies
