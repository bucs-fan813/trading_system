# trading_system/collectors/price_collector.py

import logging
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, exc
from datetime import datetime, timedelta
from config.settings import DatabaseConfig

logger = logging.getLogger(__name__)

class PriceCollector:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        try:
            self.engine = create_engine(
                config.url,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow
            )
            logger.info("Database connection established successfully for PriceCollector")
        except exc.SQLAlchemyError as e:
            logger.error(f"Failed to establish database connection in PriceCollector: {str(e)}")
            raise

    def _save_to_database(self, df: pd.DataFrame, table_name: str):
        try:
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists='replace',
                index=False,
                method='multi',
                chunksize=1000
            )
            logger.info(f"Successfully saved data to {table_name}")
        except exc.SQLAlchemyError as e:
            logger.error(f"Failed to save data to {table_name}: {str(e)}")
            raise

    def refresh_price_data(self, ticker: str):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="max")

            if hist.empty:
                logger.warning(f"No price data available for {ticker}")
                return

            hist.reset_index(inplace=True)
            hist['ticker'] = ticker
            hist['updated_at'] = datetime.now()
            hist.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)

            self._save_to_database(hist, "daily_prices")

        except Exception as e:
            logger.error(f"Error refreshing price data for {ticker}: {str(e)}")
            raise
