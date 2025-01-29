# trading_system/tests/collectors/test_info_collector.py

import pytest
from unittest.mock import MagicMock, patch
from src.collectors.info_collector import InfoCollector

@patch('yfinance.Ticker')
def test_fetch_company_info(mock_yfinance):
    """Test company info collection and flattening."""
    mock_stock = MagicMock()
    mock_stock.info = {"key": "value", "nested": {"subkey": "subvalue"}}
    mock_yfinance.return_value = mock_stock
    
    collector = InfoCollector(None)
    collector._save_to_database = MagicMock()
    
    collector.fetch_company_info("TEST")
    assert collector._save_to_database.called
    saved_df = collector._save_to_database.call_args[0][0]
    assert "nested_subkey" in saved_df.columns