# trading_system/tests/collectors/test_price_collector.py

import pytest
from unittest.mock import MagicMock, patch
from src.collectors.price_collector import PriceCollector

@patch('yfinance.Ticker')
def test_fetch_price_data(mock_yfinance):
    """Test price data collection logic."""
    mock_stock = MagicMock()
    mock_stock.history.return_value = MagicMock()
    mock_yfinance.return_value = mock_stock
    
    collector = PriceCollector(None)
    collector._get_latest_date = lambda *args: None
    collector._save_to_database = MagicMock()
    
    collector.fetch_price_data("TEST")
    assert collector._save_to_database.called

def test_refresh_data():
    """Test full refresh of price data."""
    collector = PriceCollector(None)
    collector._delete_existing_data = MagicMock()
    collector._save_to_database = MagicMock()
    
    collector.refresh_data("TEST")
    assert collector._delete_existing_data.called