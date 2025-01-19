# trading_system/collectors/data_collector.py

import logging
from database.config import DatabaseConfig
from database.engine import create_db_engine
from collectors.price_collector import PriceCollector
from collectors.info_collector import InfoCollector
from collectors.statements_collector import StatementsCollector
from datetime import datetime
import random
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def select_tickers_for_refresh(all_tickers, day_of_month):
    """Select 1/30th of the tickers based on the current day of the month."""
    random.seed(day_of_month)  # Ensure consistent selection per day
    tickers_per_day = len(all_tickers) // 30 or 1
    selected_tickers = random.sample(all_tickers, tickers_per_day)
    return selected_tickers

def main(batch_size: int = 10):
    """Main function to orchestrate data collection."""

    dbengine = create_db_engine(DatabaseConfig())

    # Ticker list
    with open("tickers.txt", "r") as f:
        all_tickers = [line.strip() for line in f if line.strip()]

    # Select tickers to refresh based on day of the month
    today = datetime.now().day
    tickers_to_refresh = select_tickers_for_refresh(all_tickers, today)

    logger.info(f"Refreshing data for tickers: {tickers_to_refresh}")

    # Initialize collectors
    price_collector = PriceCollector(dbengine)
    info_collector = InfoCollector(dbengine)
    statements_collector = StatementsCollector(dbengine)

    try:
        # Refresh price data
        for ticker in tickers_to_refresh:
            price_collector.refresh_price_data(ticker)

        # Refresh company info
        for ticker in tickers_to_refresh:
            info_collector.refresh_company_info(ticker)

        logger.info("Data refresh completed successfully.")

    except Exception as e:
        logger.error(f"Error during data refresh: {str(e)}")
        raise

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # Process financial statements
        for statement_type, fetch_func in statements_collector.financial_statements:
            futures = []
            for ticker in all_tickers:
                future = executor.submit(
                    statements_collector.fetch_financial_statement,
                    ticker,
                    statement_type,
                    fetch_func
                )
                futures.append(future)
            
            # Wait for all futures to complete
            for future in futures:
                future.result()

        # Process daily prices
        futures = []
        for ticker in all_tickers:
            future = executor.submit(price_collector.fetch_price_data, ticker)
            futures.append(future)
        
        # Wait for price futures to complete
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
