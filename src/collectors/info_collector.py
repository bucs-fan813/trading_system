# trading_system/collectors/info_collector.py

import logging
import yfinance as yf
import pandas as pd
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict
# from concurrent.futures import ThreadPoolExecutor
# TODO use threading to speed up the process

logger = logging.getLogger(__name__)

class InfoCollector:
    def __init__(self, db_engine):
            """
            Initialize with a database engine.
            :param db_engine: SQLAlchemy engine object.
            """
            self.engine = db_engine

    def _flatten_nested_dict(self, nested_dict: Dict) -> Dict:
        """Flatten a nested dictionary structure."""
        flat_dict = {}
        def flatten(d, parent_key=''):
            for k, v in d.items():
                new_key = f"{parent_key}_{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten(v, new_key)
                else:
                    flat_dict[new_key] = v
        flatten(nested_dict)
        return flat_dict

    def _save_to_database(self, df, table_name):
        """
        Save a DataFrame to the database.
        :param df: Pandas DataFrame to save.
        :param table_name: Table name in the database.
        """
        try:
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )
            logger.info(f"Data saved to {table_name} successfully.")
        except SQLAlchemyError as e:
            logger.error(f"Error saving data to {table_name}: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def fetch_company_info(self, ticker: str) -> None:
        """Fetch and store company information."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                logger.warning(f"No company info available for {ticker}")
                return

            # Flatten nested info structure
            flat_info = self._flatten_nested_dict(info)
            
            # Convert to DataFrame
            info_df = pd.DataFrame([flat_info])
            info_df['ticker'] = ticker
            info_df['updated_at'] = datetime.now()
            info_df['data_source'] = 'yfinance'

            # Save to database
            self._save_to_database(info_df, 'company_info')
            logger.info(f"Successfully processed company info for {ticker}")

        except Exception as e:
            logger.error(f"Error processing company info for {ticker}: {str(e)}")
            raise
