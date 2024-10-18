import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trade_manager import TradeManager

# Test data
TEST_ASSET = "BTC"
TEST_BASE_CURRENCY = "USD"
TEST_ASSET_AMOUNT = 1.0
TEST_BASE_AMOUNT = 10000.0
TEST_ENTRY_PRICE = 50000.0
TEST_ENTRY_TIMESTAMP = 1622520000
TEST_ENTRY_FEE = 50.0
TEST_EXIT_PRICE = 51000.0
TEST_EXIT_TIMESTAMP = 1622523600
TEST_EXIT_FEE = 50.0

@pytest.fixture
def trade_manager():
    """Fixture to create a fresh instance of TradeManager for each test."""
    return TradeManager()

# Test opening trades

def test_open_trade(trade_manager):
    trade = trade_manager.open_trade(
        asset_name=TEST_ASSET,
        base_currency=TEST_BASE_CURRENCY,
        asset_amount=TEST_ASSET_AMOUNT,
        base_amount=TEST_BASE_AMOUNT,
        entry_price=TEST_ENTRY_PRICE,
        entry_timestamp=TEST_ENTRY_TIMESTAMP,
        entry_fee=TEST_ENTRY_FEE
    )
    assert trade["id"] == 1
    assert trade["asset_name"] == TEST_ASSET
    assert trade["status"] == TradeManager.OPEN_STATUS

def test_open_trade_with_invalid_amount(trade_manager):
    with pytest.raises(ValueError):
        trade_manager.open_trade(
            asset_name=TEST_ASSET,
            base_currency=TEST_BASE_CURRENCY,
            asset_amount=-TEST_ASSET_AMOUNT,  # Invalid amount
            base_amount=TEST_BASE_AMOUNT,
            entry_price=TEST_ENTRY_PRICE,
            entry_timestamp=TEST_ENTRY_TIMESTAMP,
            entry_fee=TEST_ENTRY_FEE
        )

def test_open_trade_with_zero_asset_amount(trade_manager):
    with pytest.raises(ValueError):
        trade_manager.open_trade(
            asset_name=TEST_ASSET,
            base_currency=TEST_BASE_CURRENCY,
            asset_amount=0,  # Invalid amount
            base_amount=TEST_BASE_AMOUNT,
            entry_price=TEST_ENTRY_PRICE,
            entry_timestamp=TEST_ENTRY_TIMESTAMP,
            entry_fee=TEST_ENTRY_FEE
        )

def test_open_trade_with_invalid_direction(trade_manager):
    with pytest.raises(ValueError):
        trade_manager.open_trade(
            asset_name=TEST_ASSET,
            base_currency=TEST_BASE_CURRENCY,
            asset_amount=TEST_ASSET_AMOUNT,
            base_amount=TEST_BASE_AMOUNT,
            entry_price=TEST_ENTRY_PRICE,
            entry_timestamp=TEST_ENTRY_TIMESTAMP,
            entry_fee=TEST_ENTRY_FEE,
            direction="invalid_direction"  # Invalid direction
        )

# Test closing trades

def test_close_trade(trade_manager):
    trade = trade_manager.open_trade(
        asset_name=TEST_ASSET,
        base_currency=TEST_BASE_CURRENCY,
        asset_amount=TEST_ASSET_AMOUNT,
        base_amount=TEST_BASE_AMOUNT,
        entry_price=TEST_ENTRY_PRICE,
        entry_timestamp=TEST_ENTRY_TIMESTAMP,
        entry_fee=TEST_ENTRY_FEE
    )
    closed_trade = trade_manager.close_trade(
        trade_id=trade["id"],
        exit_price=TEST_EXIT_PRICE,
        exit_timestamp=TEST_EXIT_TIMESTAMP,
        exit_fee=TEST_EXIT_FEE
    )
    assert closed_trade["status"] == TradeManager.CLOSED_STATUS
    assert closed_trade["profit_loss"] is not None

def test_close_non_existent_trade(trade_manager):
    assert trade_manager.close_trade(trade_id=999, exit_price=TEST_EXIT_PRICE, exit_timestamp=TEST_EXIT_TIMESTAMP, exit_fee=TEST_EXIT_FEE) is None

def test_close_already_closed_trade(trade_manager):
    trade = trade_manager.open_trade(
        asset_name=TEST_ASSET,
        base_currency=TEST_BASE_CURRENCY,
        asset_amount=TEST_ASSET_AMOUNT,
        base_amount=TEST_BASE_AMOUNT,
        entry_price=TEST_ENTRY_PRICE,
        entry_timestamp=TEST_ENTRY_TIMESTAMP,
        entry_fee=TEST_ENTRY_FEE
    )
    trade_manager.close_trade(
        trade_id=trade["id"],
        exit_price=TEST_EXIT_PRICE,
        exit_timestamp=TEST_EXIT_TIMESTAMP,
        exit_fee=TEST_EXIT_FEE
    )
    assert trade_manager.close_trade(trade_id=trade["id"], exit_price=TEST_EXIT_PRICE, exit_timestamp=TEST_EXIT_TIMESTAMP, exit_fee=TEST_EXIT_FEE) is None

def test_close_trade_without_open_trade(trade_manager):
    result = trade_manager.close_trade(trade_id=1, exit_price=51000, exit_timestamp=TEST_EXIT_TIMESTAMP, exit_fee=TEST_EXIT_FEE)
    assert result is None

# Test modifying trades

def test_modify_trade_parameters(trade_manager):
    trade = trade_manager.open_trade(
        asset_name=TEST_ASSET,
        base_currency=TEST_BASE_CURRENCY,
        asset_amount=TEST_ASSET_AMOUNT,
        base_amount=TEST_BASE_AMOUNT,
        entry_price=TEST_ENTRY_PRICE,
        entry_timestamp=TEST_ENTRY_TIMESTAMP,
        entry_fee=TEST_ENTRY_FEE
    )
    modified_trade = trade_manager.modify_trade_parameters(
        trade_id=trade["id"],
        stop_loss=49000.0,
        take_profit=52000.0
    )
    assert modified_trade["stop_loss"] == 49000.0
    assert modified_trade["take_profit"] == 52000.0

def test_modify_closed_trade_parameters(trade_manager):
    trade = trade_manager.open_trade(
        asset_name=TEST_ASSET,
        base_currency=TEST_BASE_CURRENCY,
        asset_amount=TEST_ASSET_AMOUNT,
        base_amount=TEST_BASE_AMOUNT,
        entry_price=TEST_ENTRY_PRICE,
        entry_timestamp=TEST_ENTRY_TIMESTAMP,
        entry_fee=TEST_ENTRY_FEE
    )
    trade_manager.close_trade(trade_id=trade["id"], exit_price=TEST_EXIT_PRICE, exit_timestamp=TEST_EXIT_TIMESTAMP, exit_fee=TEST_EXIT_FEE)
    result = trade_manager.modify_trade_parameters(trade_id=trade["id"], stop_loss=49000.0, take_profit=52000.0)
    assert result is None

def test_modify_non_existent_trade(trade_manager):
    assert trade_manager.modify_trade_parameters(trade_id=999, stop_loss=49000.0, take_profit=52000.0) is None

# Test retrieving trades

def test_get_trade_by_id(trade_manager):
    trade = trade_manager.open_trade(
        asset_name=TEST_ASSET,
        base_currency=TEST_BASE_CURRENCY,
        asset_amount=TEST_ASSET_AMOUNT,
        base_amount=TEST_BASE_AMOUNT,
        entry_price=TEST_ENTRY_PRICE,
        entry_timestamp=TEST_ENTRY_TIMESTAMP,
        entry_fee=TEST_ENTRY_FEE
    )
    retrieved_trade = trade_manager.get_trade(trade_id=trade["id"])
    assert retrieved_trade["id"] == trade["id"]

def test_get_non_existent_trade(trade_manager):
    assert trade_manager.get_trade(trade_id=999) is None

def test_get_trades_by_status(trade_manager):
    trade_manager.open_trade(
        asset_name=TEST_ASSET,
        base_currency=TEST_BASE_CURRENCY,
        asset_amount=TEST_ASSET_AMOUNT,
        base_amount=TEST_BASE_AMOUNT,
        entry_price=TEST_ENTRY_PRICE,
        entry_timestamp=TEST_ENTRY_TIMESTAMP,
        entry_fee=TEST_ENTRY_FEE
    )
    open_trades = trade_manager.get_trade(status=TradeManager.OPEN_STATUS)
    assert len(open_trades) == 1
    
    trade_manager.close_trade(trade_id=1, exit_price=51000, exit_timestamp=TEST_EXIT_TIMESTAMP, exit_fee=TEST_EXIT_FEE)
    closed_trades = trade_manager.get_trade(status=TradeManager.CLOSED_STATUS)
    assert len(closed_trades) == 1

# Test unique IDs

def test_unique_trade_ids(trade_manager):
    trade1 = trade_manager.open_trade(
        asset_name=TEST_ASSET,
        base_currency=TEST_BASE_CURRENCY,
        asset_amount=TEST_ASSET_AMOUNT,
        base_amount=TEST_BASE_AMOUNT,
        entry_price=TEST_ENTRY_PRICE,
        entry_timestamp=TEST_ENTRY_TIMESTAMP,
        entry_fee=TEST_ENTRY_FEE
    )
    trade2 = trade_manager.open_trade(
        asset_name=TEST_ASSET,
        base_currency=TEST_BASE_CURRENCY,
        asset_amount=TEST_ASSET_AMOUNT,
        base_amount=TEST_BASE_AMOUNT,
        entry_price=TEST_ENTRY_PRICE,
        entry_timestamp=TEST_ENTRY_TIMESTAMP,
        entry_fee=TEST_ENTRY_FEE
    )
    assert trade1["id"] != trade2["id"]

# Test profit calculation

def test_profit_calculation(trade_manager):
    trade = trade_manager.open_trade(
        asset_name=TEST_ASSET,
        base_currency=TEST_BASE_CURRENCY,
        asset_amount=TEST_ASSET_AMOUNT,
        base_amount=TEST_BASE_AMOUNT,
        entry_price=TEST_ENTRY_PRICE,
        entry_timestamp=TEST_ENTRY_TIMESTAMP,
        entry_fee=TEST_ENTRY_FEE
    )
    closed_trade = trade_manager.close_trade(
        trade_id=trade["id"],
        exit_price=TEST_EXIT_PRICE,
        exit_timestamp=TEST_EXIT_TIMESTAMP,
        exit_fee=TEST_EXIT_FEE
    )
    expected_profit = (TEST_EXIT_PRICE - TEST_ENTRY_PRICE) * TEST_ASSET_AMOUNT - TEST_ENTRY_FEE - TEST_EXIT_FEE
    assert closed_trade["profit_loss"] == expected_profit

# Test empty trade list

def test_get_trade_when_no_trades_exist(trade_manager):
    assert trade_manager.get_trade() == []

def test_close_trade_when_no_trades_exist(trade_manager):
    assert trade_manager.close_trade(trade_id=1, exit_price=TEST_EXIT_PRICE, exit_timestamp=TEST_EXIT_TIMESTAMP, exit_fee=TEST_EXIT_FEE) is None
