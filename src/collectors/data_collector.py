# trading_system/collectors/data_collector.py

import logging
from collectors.price_collector import PriceCollector
from collectors.info_collector import InfoCollector
from collectors.statements_collector import StatementsCollector
from config.settings import DatabaseConfig
from datetime import datetime
import random

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

def main():
    """Main function to orchestrate data collection."""
    # Load database configuration
    config = DatabaseConfig(
        url="sqlite:///trading_data.db",
        max_retries=3,
        pool_size=5,
        max_overflow=10
    )

    # Ticker list
    with open("tickers.txt", "r") as f:
        all_tickers = [line.strip() for line in f if line.strip()]

    # Select tickers to refresh based on day of the month
    today = datetime.now().day
    tickers_to_refresh = select_tickers_for_refresh(all_tickers, today)

    logger.info(f"Refreshing data for tickers: {tickers_to_refresh}")

    # Initialize collectors
    price_collector = PriceCollector(config)
    info_collector = InfoCollector(config)
    statements_collector = StatementsCollector(config)

    try:
        # Refresh price data
        for ticker in tickers_to_refresh:
            price_collector.refresh_price_data(ticker)

        # Refresh company info
        for ticker in tickers_to_refresh:
            info_collector.refresh_company_info(ticker)

        # Refresh financial statements
        for ticker in tickers_to_refresh:
            statements_collector.refresh_statements(ticker)

        logger.info("Data refresh completed successfully.")

    except Exception as e:
        logger.error(f"Error during data refresh: {str(e)}")
        raise

if __name__ == "__main__":
    main()
