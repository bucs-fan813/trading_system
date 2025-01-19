import math
from datetime import datetime
from src.config.logging_config import configure_logging
from src.database.config import DatabaseConfig
from src.database.engine import create_db_engine
from src.collectors.financial_data import FinancialDataCollector

def divide_tickers(tickers, cycle_days=30):
    """
    Divide tickers into batches based on a 30-day cycle.
    :param tickers: List of ticker symbols.
    :param cycle_days: Number of days in the refresh cycle.
    :return: List of tickers for today's batch.
    """
    day_of_cycle = (datetime.now().timetuple().tm_yday - 1) % cycle_days + 1
    batch_size = math.ceil(len(tickers) / cycle_days)
    start_idx = (day_of_cycle - 1) * batch_size
    end_idx = min(start_idx + batch_size, len(tickers))
    return tickers[start_idx:end_idx]

def main():
    """Entry point for financial data collection."""
    configure_logging()

    # Database configuration
    config = DatabaseConfig(url="sqlite:///trading_data.db")
    db_engine = create_db_engine(config)

    # Collector initialization
    collector = FinancialDataCollector(db_engine)

    # Ticker list (replace with your own)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]

    # Select today's batch
    tickers_today = divide_tickers(tickers)

    # Process data for tickers
    try:
        for ticker in tickers_today:
            # Refresh daily prices (important for splits/dividends)
            collector.refresh_data(
                ticker,
                "daily_prices",
                lambda stock: stock.history(period="1y")
            )
            
            # Refresh company info (to update shares outstanding, market cap, etc.)
            collector.refresh_data(
                ticker,
                "company_info",
                lambda stock: stock.info
            )
        
        print(f"Data refreshed for today's tickers: {tickers_today}")
    except Exception as e:
        print(f"Data collection failed: {e}")

if __name__ == "__main__":
    main()
