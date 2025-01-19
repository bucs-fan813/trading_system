# trading_system/collectors/statements_collector.py

import logging
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, exc
from datetime import datetime
from config.settings import DatabaseConfig

logger = logging.getLogger(__name__)

class StatementsCollector:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        try:
            self.engine = create_engine(
                config.url,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow
            )
            logger.info("Database connection established successfully for StatementsCollector")
        except exc.SQLAlchemyError as e:
            logger.error(f"Failed to establish database connection in StatementsCollector: {str(e)}")
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

    def refresh_statements(self, ticker: str):
        try:
            stock = yf.Ticker(ticker)

            # Balance Sheet
            balance_sheet = stock.get_balancesheet(as_dict=False)
            if not balance_sheet.empty:
                balance_sheet = balance_sheet.T.reset_index()
                balance_sheet['ticker'] = ticker
                balance_sheet['statement_type'] = 'balance_sheet'
                balance_sheet['updated_at'] = datetime.now()
                self._save_to_database(balance_sheet, "financial_statements")

            # Income Statement
            income_statement = stock.get_incomestatement(as_dict=False)
            if not income_statement.empty:
                income_statement = income_statement.T.reset_index()
                income_statement['ticker'] = ticker
                income_statement['statement_type'] = 'income_statement'
                income_statement['updated_at'] = datetime.now()
                self._save_to_database(income_statement, "financial_statements")

            # Cash Flow Statement
            cash_flow = stock.get_cashflow(as_dict=False)
            if not cash_flow.empty:
                cash_flow = cash_flow.T.reset_index()
                cash_flow['ticker'] = ticker
                cash_flow['statement_type'] = 'cash_flow'
                cash_flow['updated_at'] = datetime.now()
                self._save_to_database(cash_flow, "financial_statements")

        except Exception as e:
            logger.error(f"Error refreshing financial statements for {ticker}: {str(e)}")
            raise
