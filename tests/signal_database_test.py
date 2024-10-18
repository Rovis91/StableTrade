import sys
import os
import pytest
from typing import Dict, List, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.signal_database import SignalDatabase

@pytest.fixture
def signal_database():
    return SignalDatabase()

def test_add_single_valid_signal(signal_database):
    signal: Dict[str, Any] = {
        'timestamp': 1620000000000,
        'asset_name': 'BTC/USD',
        'action': 'buy',
        'amount': 1.0,
        'price': 50000.0
    }
    signal_database.add_signals([signal])
    assert len(signal_database.signals) == 1
    assert signal_database.signals[0]['signal_id'] == 1
    assert signal_database.signals[0]['status'] == 'pending'
    assert signal_database.signals[0]['trade_id'] is None


def test_add_multiple_valid_signals(signal_database):
    signals: List[Dict[str, Any]] = [
        {'timestamp': 1620000000000, 'asset_name': 'BTC/USD', 'action': 'buy', 'amount': 1.0, 'price': 50000.0},
        {'timestamp': 1620000060000, 'asset_name': 'ETH/USD', 'action': 'sell', 'amount': 2.0, 'price': 3000.0}
    ]
    signal_database.add_signals(signals)
    assert len(signal_database.signals) == 2
    assert signal_database.signals[0]['signal_id'] == 1
    assert signal_database.signals[1]['signal_id'] == 2
    assert all(signal['trade_id'] is None for signal in signal_database.signals)

def test_add_signals_with_missing_fields(signal_database):
    signals: List[Dict[str, Any]] = [
        {'timestamp': 1620000000000, 'asset_name': 'BTC/USD', 'action': 'buy', 'amount': 1.0, 'price': 50000.0},
        {'timestamp': 1620000060000, 'asset_name': 'ETH/USD', 'action': 'sell', 'amount': 2.0}  # Missing price
    ]
    with pytest.raises(ValueError, match="Missing required field: price"):
        signal_database.add_signals(signals)

def test_add_signals_with_invalid_action(signal_database):
    signal: Dict[str, Any] = {
        'timestamp': 1620000000000,
        'asset_name': 'BTC/USD',
        'action': 'hold',  # Invalid action
        'amount': 1.0,
        'price': 50000.0
    }
    with pytest.raises(ValueError, match="Invalid action: hold"):
        signal_database.add_signals([signal])

def test_add_signals_with_invalid_amount(signal_database):
    signal: Dict[str, Any] = {
        'timestamp': 1620000000000,
        'asset_name': 'BTC/USD',
        'action': 'buy',
        'amount': -1.0,  # Invalid amount
        'price': 50000.0
    }
    with pytest.raises(ValueError, match="Invalid amount: -1.0"):
        signal_database.add_signals([signal])

def test_add_signals_with_invalid_price(signal_database):
    signal: Dict[str, Any] = {
        'timestamp': 1620000000000,
        'asset_name': 'BTC/USD',
        'action': 'buy',
        'amount': 1.0,
        'price': 0.0  # Invalid price
    }
    with pytest.raises(ValueError, match="Invalid price: 0.0"):
        signal_database.add_signals([signal])

def test_add_empty_signal_list(signal_database):
    signal_database.add_signals([])
    assert len(signal_database.signals) == 0

def test_add_signals_incremental_ids(signal_database):
    signals1 = [
        {'timestamp': 1620000000000, 'asset_name': 'BTC/USD', 'action': 'buy', 'amount': 1.0, 'price': 50000.0},
        {'timestamp': 1620000060000, 'asset_name': 'ETH/USD', 'action': 'sell', 'amount': 2.0, 'price': 3000.0}
    ]
    signals2 = [
        {'timestamp': 1620000120000, 'asset_name': 'LTC/USD', 'action': 'buy', 'amount': 3.0, 'price': 200.0},
        {'timestamp': 1620000180000, 'asset_name': 'XRP/USD', 'action': 'sell', 'amount': 1000.0, 'price': 1.0}
    ]
    signal_database.add_signals(signals1)
    signal_database.add_signals(signals2)
    assert len(signal_database.signals) == 4
    assert [s[signal_database.COLUMN_SIGNAL_ID] for s in signal_database.signals] == [1, 2, 3, 4]

def test_add_signals_type_conversion(signal_database):
    signal: Dict[str, Any] = {
        'timestamp': '1620000000000',  # String instead of int
        'asset_name': 'BTC/USD',
        'action': 'buy',
        'amount': '1.0',  # String instead of float
        'price': '50000'  # String instead of float
    }
    signal_database.add_signals([signal])
    assert isinstance(signal_database.signals[0]['timestamp'], int)
    assert isinstance(signal_database.signals[0]['amount'], float)
    assert isinstance(signal_database.signals[0]['price'], float)
    assert signal_database.signals[0]['timestamp'] == 1620000000000
    assert signal_database.signals[0]['amount'] == 1.0
    assert signal_database.signals[0]['price'] == 50000.0

def test_update_signal_status(signal_database):
    signal: Dict[str, Any] = {'timestamp': 1620000000000, 'asset_name': 'BTC/USD', 'action': 'buy', 'amount': 1.0, 'price': 50000.0}
    signal_database.add_signals([signal])
    signal_database.update_signal_status(1, 'executed', 'Test reason')
    assert signal_database.signals[0]['status'] == 'executed'
    assert signal_database.signals[0]['reason'] == 'Test reason'

def test_update_trade_id(signal_database):
    signal: Dict[str, Any] = {'timestamp': 1620000000000, 'asset_name': 'BTC/USD', 'action': 'buy', 'amount': 1.0, 'price': 50000.0}
    signal_database.add_signals([signal])
    signal_database.update_trade_id(1, 100)
    assert signal_database.signals[0]['trade_id'] == 100

def test_get_signals_with_filters(signal_database):
    signals: List[Dict[str, Any]] = [
        {'timestamp': 1620000000000, 'asset_name': 'BTC/USD', 'action': 'buy', 'amount': 1.0, 'price': 50000.0},
        {'timestamp': 1620000060000, 'asset_name': 'ETH/USD', 'action': 'sell', 'amount': 2.0, 'price': 3000.0}
    ]
    signal_database.add_signals(signals)
    filtered_signals = signal_database.get_signals(asset_name='BTC/USD')
    assert len(filtered_signals) == 1
    assert filtered_signals[0]['asset_name'] == 'BTC/USD'

