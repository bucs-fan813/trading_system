# trading_system/tests/collectors/test_statements_collector.py

import pytest
from unittest.mock import MagicMock, patch
from src.collectors.statements_collector import StatementsCollector

@pytest.fixture
def mock_collector(test_engine):
    return StatementsCollector(test_engine)

@patch('yfinance.Ticker')
def test_fetch_financial_statement(mock_yfinance):
    """Test financial statement fetching and processing."""
    mock_stock = MagicMock()
    mock_stock.balance_sheet = MagicMock()
    mock_yfinance.return_value = mock_stock
    
    collector = StatementsCollector(None)
    collector._get_latest_date = lambda *args: None
    collector._save_to_database = MagicMock()
    
    collector.fetch_financial_statement("TEST", "balance_sheet", lambda x: x.balance_sheet)
    assert collector._save_to_database.called