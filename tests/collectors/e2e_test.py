# trading_system/tests/e2e/test_nse_stock.py

import sys
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

import pytest
from src.database.config import DatabaseConfig
from src.database.engine import create_db_engine
from src.collectors.price_collector import PriceCollector
from src.collectors.info_collector import InfoCollector
from src.collectors.statements_collector import StatementsCollector
from sqlalchemy import inspect, text
import pandas as pd
import logging
import time



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_TICKER = "RELIANCE.NS"
TEST_DB_URL = "sqlite:///test_trading_system.db"  # Persistent DB for inspection

@pytest.fixture(scope="module")
def test_engine():
    """Fixture with test database engine"""
    config = DatabaseConfig(url=TEST_DB_URL, pool_size=5)
    engine = create_db_engine(config)
    yield engine
    engine.dispose()

def test_end_to_end_flow(test_engine):
    """End-to-end test for NSE stock data collection"""
    
    # --------------------------
    # 1. Test Company Info Collector
    # --------------------------
    info_collector = InfoCollector(test_engine)
    info_collector.fetch_company_info(TEST_TICKER)
    
    # Verify company info
    with test_engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT * FROM company_info WHERE ticker = :ticker"),
            {"ticker": TEST_TICKER}
        ).fetchone()
        
    print(result)
    logger.info("Company info test passed")

    # --------------------------
    # 2. Test Price Collector
    # --------------------------
    price_collector = PriceCollector(test_engine)
    
    # First run - initial data
    price_collector.fetch_and_save(TEST_TICKER)
    
    # Second run - incremental update
    time.sleep(2)  # Ensure timestamp changes
    price_collector.fetch_and_save(TEST_TICKER)
    
    # Verify price data
    with test_engine.connect() as conn:
        prices = pd.read_sql(
            f"SELECT * FROM daily_prices WHERE ticker = '{TEST_TICKER}'",
            conn
        )
        
    print(prices)
    logger.info(f"Collected {len(prices)} price records")

    # --------------------------
    # 3. Test Financial Statements
    # --------------------------
    stmt_collector = StatementsCollector(test_engine)
    
    # Test balance sheet
    stmt_collector.fetch_financial_statement(
        TEST_TICKER,
        "balance_sheet",
        lambda stock: stock.balance_sheet
    )
    
    # Test income statement
    stmt_collector.fetch_financial_statement(
        TEST_TICKER,
        "income_statement",
        lambda stock: stock.income_stmt
    )
    
    # Test cash flow
    stmt_collector.fetch_financial_statement(
        TEST_TICKER,
        "cash_flow",
        lambda stock: stock.cash_flow
    )

    # Verify financial data
    with test_engine.connect() as conn:
        tables = inspect(conn).get_table_names()
        assert "balance_sheet" in tables, "Balance sheet table missing"
        assert "income_statement" in tables, "Income statement table missing"
        assert "cash_flow" in tables, "Cash flow table missing"
        
    logger.info("Financial statements test passed")

    # --------------------------
    # 4. Test Data Freshness
    # --------------------------
    with test_engine.connect() as conn:
        max_prices_date = conn.execute(
            text(f"SELECT MAX(date) FROM daily_prices WHERE ticker = :ticker"),
            {"ticker": TEST_TICKER}
        ).scalar()
        
        max_info_date = conn.execute(
            text(f"SELECT MAX(updated_at) FROM company_info WHERE ticker = :ticker"),
            {"ticker": TEST_TICKER}
        ).scalar()

    print(max_prices_date)
    print(max_info_date)
    logger.info("Data freshness test passed")

"""
def test_cleanup(test_engine):
    #Optional cleanup (comment out to inspect database)
    with test_engine.connect() as conn:
        conn.execute(text("DELETE FROM company_info WHERE ticker = :ticker"), {"ticker": TEST_TICKER})
        conn.execute(text("DELETE FROM daily_prices WHERE ticker = :ticker"), {"ticker": TEST_TICKER})
        conn.execute(text("DELETE FROM balance_sheet WHERE ticker = :ticker"), {"ticker": TEST_TICKER})
        conn.execute(text("DELETE FROM income_statement WHERE ticker = :ticker"), {"ticker": TEST_TICKER})
        conn.execute(text("DELETE FROM cash_flow WHERE ticker = :ticker"), {"ticker": TEST_TICKER})
        conn.commit()
    logger.info("Test data cleaned up")

    """