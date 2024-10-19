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
        {'timestamp': 1620000060000, 'asset_name': 'ETH/USD', 'action': 'sell', 'amount': 0.5, 'price': 3000.0}
    ]
    signal_database.add_signals(signals)
    assert len(signal_database.signals) == 2
    assert signal_database.signals[0]['signal_id'] == 1
    assert signal_database.signals[1]['signal_id'] == 2
    assert all(signal['trade_id'] is None for signal in signal_database.signals)  # trade_id should be None for new signals

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
        'action': 'invalid_action',  # Invalid action
        'amount': 1.0,
        'price': 50000.0
    }
    with pytest.raises(ValueError, match="Invalid action: invalid_action"):
        signal_database.add_signals([signal])

def test_add_signals_with_invalid_amount(signal_database):
    signal: Dict[str, Any] = {
        'timestamp': 1620000000000,
        'asset_name': 'BTC/USD',
        'action': 'buy',
        'amount': -1.0,  # Invalid amount
        'price': 50000.0
    }
    with pytest.raises(ValueError, match="Invalid amount in signal: -1.0"):
        signal_database.add_signals([signal])

def test_add_signals_with_invalid_price(signal_database):
    signal: Dict[str, Any] = {
        'timestamp': 1620000000000,
        'asset_name': 'BTC/USD',
        'action': 'buy',
        'amount': 1.0,
        'price': 0.0  # Invalid price
    }
    with pytest.raises(ValueError, match="Invalid price in signal: 0.0"):
        signal_database.add_signals([signal])

def test_add_empty_signal_list(signal_database):
    signal_database.add_signals([])
    assert len(signal_database.signals) == 0

def test_add_signals_incremental_ids(signal_database):
    signals1 = [
        {'timestamp': 1620000000000, 'asset_name': 'BTC/USD', 'action': 'buy', 'amount': 0.1, 'price': 50000.0},
        {'timestamp': 1620000060000, 'asset_name': 'ETH/USD', 'action': 'sell', 'amount': 0.2, 'price': 3000.0}
    ]
    signals2 = [
        {'timestamp': 1620000120000, 'asset_name': 'LTC/USD', 'action': 'buy', 'amount': 0.4, 'price': 200.0},
        {'timestamp': 1620000180000, 'asset_name': 'XRP/USD', 'action': 'sell', 'amount': 0.2, 'price': 1.0}
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
    signal_database.update_signal_field(signal_id=1, field_name='status', new_value='executed')
    signal_database.update_signal_field(signal_id=1, field_name='reason', new_value='Test reason')
    assert signal_database.signals[0]['status'] == 'executed'
    assert signal_database.signals[0]['reason'] == 'Test reason'

def test_get_signals_with_filters(signal_database):
    signals: List[Dict[str, Any]] = [
        {'timestamp': 1620000000000, 'asset_name': 'BTC/USD', 'action': 'buy', 'amount': 1.0, 'price': 50000.0},
        {'timestamp': 1620000060000, 'asset_name': 'ETH/USD', 'action': 'sell', 'amount': 0.5, 'price': 3000.0}
    ]
    signal_database.add_signals(signals)
    filtered_signals = signal_database.get_signals(asset_name='BTC/USD')
    assert len(filtered_signals) == 1
    assert filtered_signals[0]['asset_name'] == 'BTC/USD'

def test_add_duplicate_signals(signal_database):
    signal1: Dict[str, Any] = {
        'timestamp': 1620000000000,
        'asset_name': 'BTC/USD',
        'action': 'buy',
        'amount': 1.0,
        'price': 50000.0
    }
    signal2: Dict[str, Any] = signal1.copy()  #
    
    signal_database.add_signals([signal1, signal2])
    
    assert len(signal_database.signals) == 2
    assert signal_database.signals[0]['signal_id'] == 1
    assert signal_database.signals[1]['signal_id'] == 2

def test_update_invalid_field(signal_database):
    signal: Dict[str, Any] = {
        'timestamp': 1620000000000,
        'asset_name': 'BTC/USD',
        'action': 'buy',
        'amount': 1.0,
        'price': 50000.0
    }
    signal_database.add_signals([signal])

    # Try updating an invalid field, which should raise a ValueError
    with pytest.raises(ValueError, match="Field 'amount' cannot be updated"):
        signal_database.update_signal_field(signal_id=1, field_name='amount', new_value=0.5)

def test_update_invalid_type(signal_database):
    signal: Dict[str, Any] = {
        'timestamp': 1620000000000,
        'asset_name': 'BTC/USD',
        'action': 'buy',
        'amount': 1.0,
        'price': 50000.0
    }
    signal_database.add_signals([signal])

    # Try updating 'status' with an invalid type (int instead of str), which should raise a TypeError
    with pytest.raises(TypeError, match="Expected str for field 'status', got int"):
        signal_database.update_signal_field(signal_id=1, field_name='status', new_value=123)

def test_update_non_existent_signal(signal_database):
    # Try updating a signal ID that doesn't exist, which should raise a ValueError
    with pytest.raises(ValueError, match="Signal 999 not found"):
        signal_database.update_signal_field(signal_id=999, field_name='status', new_value='executed')

def test_add_empty_signal(signal_database):
    empty_signal: Dict[str, Any] = {}
    
    # Adding an empty signal should raise a ValueError for missing fields
    with pytest.raises(ValueError, match="Missing required field: timestamp"):
        signal_database.add_signals([empty_signal])

def test_add_close_signal_without_trade_id(signal_database):
    close_signal: Dict[str, Any] = {
        'timestamp': 1620000000000,
        'asset_name': 'BTC/USD',
        'action': 'close',
        'amount': 1.0,
        'price': 50000.0
    }
    
    # Close signal without 'trade_id' should raise a ValueError
    with pytest.raises(ValueError, match="Close signal is missing 'trade_id'"):
        signal_database.add_signals([close_signal])

def test_csv_export(signal_database, tmpdir):
    signals: List[Dict[str, Any]] = [
        {'timestamp': 1620000000000, 'asset_name': 'BTC/USD', 'action': 'buy', 'amount': 1.0, 'price': 50000.0},
        {'timestamp': 1620000060000, 'asset_name': 'ETH/USD', 'action': 'sell', 'amount': 0.5, 'price': 3000.0}
    ]
    
    signal_database.add_signals(signals)

    # Export to a temporary file
    csv_file = os.path.join(tmpdir, "signals.csv")
    signal_database.to_csv(csv_file)

    # Check if file was created and content matches the signals
    with open(csv_file, 'r') as f:
        content = f.readlines()
    
    assert len(content) > 1  # Ensure the file contains data
    assert "BTC/USD" in content[1]  # Check the content of the first signal

def test_add_signals_with_invalid_type_conversions(signal_database):
    signal: Dict[str, Any] = {
        'timestamp': 'not_a_number',  # Invalid timestamp
        'asset_name': 'BTC/USD',
        'action': 'buy',
        'amount': 'not_a_number',  # Invalid amount
        'price': 'not_a_number'  # Invalid price
    }
    
    # Adding a signal with non-convertible types should raise a ValueError
    with pytest.raises(ValueError):
        signal_database.add_signals([signal])

def test_invalid_stop_loss_take_profit_values(signal_database):
    signal: Dict[str, Any] = {
        'timestamp': 1620000000000,
        'asset_name': 'BTC/USD',
        'action': 'buy',
        'amount': 1.0,
        'price': 50000.0,
        'stop_loss': -1.0,  # Invalid stop loss
        'take_profit': -2.0  # Invalid take profit
    }
    
    # Stop loss and take profit should each raise a single ValueError
    with pytest.raises(ValueError, match="Invalid stop_loss: -1.0. Must be a valid number greater than 0."):
        signal_database.add_signals([signal])

    signal['stop_loss'] = 0.5  # Valid stop loss
    with pytest.raises(ValueError, match="Invalid take_profit: -2.0. Must be a valid number greater than 0."):
        signal_database.add_signals([signal])
