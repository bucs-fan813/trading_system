# trading_system/collectors/info_collector.py

import logging
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, exc
from datetime import datetime
from config.settings import DatabaseConfig

logger = logging.getLogger(__name__)

class InfoCollector:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        try:
            self.engine = create_engine(
                config.url,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow
            )
            logger.info("Database connection established successfully for InfoCollector")
        except exc.SQLAlchemyError as e:
            logger.error(f"Failed to establish database connection in InfoCollector: {str(e)}")
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

    def refresh_company_info(self, ticker: str):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                logger.warning(f"No company info available for {ticker}")
                return

            data = {
                "ticker": ticker,
                "long_name": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "country": info.get("country"),
                "website": info.get("website"),
                "market_cap": info.get("marketCap"),
                "updated_at": datetime.now()
            }

            df = pd.DataFrame([data])
            self._save_to_database(df, "company_info")

        except Exception as e:
            logger.error(f"Error refreshing company info for {ticker}: {str(e)}")
            raise
