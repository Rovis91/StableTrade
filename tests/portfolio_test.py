import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import Mock, call
from src.portfolio import Portfolio
from src.trade_manager import TradeManager
from src.signal_database import SignalDatabase


@pytest.fixture
def mock_trade_manager():
    return Mock(spec=TradeManager)

@pytest.fixture
def mock_signal_database():
    return Mock(spec=SignalDatabase)

@pytest.fixture
def valid_portfolio_config():
    return {
        'BTC': {
            'market_type': 'spot',
            'fees': {'entry': 0.001, 'exit': 0.001},
            'max_trades': 5,
            'max_exposure': 0.5
        },
        'ETH': {
            'market_type': 'futures',
            'fees': {'entry': 0.002, 'exit': 0.002},
            'max_trades': 3,
            'max_exposure': 0.3
        }
    }

@pytest.fixture
def portfolio(mock_trade_manager, mock_signal_database, valid_portfolio_config):
    return Portfolio(
        initial_cash=10000.0,
        portfolio_config=valid_portfolio_config,
        signal_database=mock_signal_database,
        trade_manager=mock_trade_manager,
        base_currency='USD'
    )

def test_invalid_initialization_cases(mock_trade_manager, mock_signal_database):
    initial_cash = 10000.0

    # Invalid portfolio_config type
    invalid_config = ['invalid_config']
    with pytest.raises(TypeError, match="Portfolio configuration must be a dictionary"):
        Portfolio(initial_cash=initial_cash, portfolio_config=invalid_config, 
                  signal_database=mock_signal_database, trade_manager=mock_trade_manager, base_currency='USD')

    # Invalid asset configuration (missing required keys)
    invalid_portfolio_config = {'BTC': {'market_type': 'spot', 'fees': {'entry': 0.001}}}  # Missing 'max_trades' and 'max_exposure'
    with pytest.raises(ValueError) as excinfo:
        Portfolio(initial_cash=initial_cash, portfolio_config=invalid_portfolio_config, 
                  signal_database=mock_signal_database, trade_manager=mock_trade_manager, base_currency='USD')
    
    error_message = str(excinfo.value)
    assert "Invalid configuration for asset BTC" in error_message
    assert "Missing required keys" in error_message
    assert "max_trades" in error_message
    assert "max_exposure" in error_message

def test_edge_cases_for_cash_and_config(mock_trade_manager, mock_signal_database):
    # Test zero initial cash
    portfolio_zero_cash = Portfolio(initial_cash=0, portfolio_config={}, 
                                    signal_database=mock_signal_database, trade_manager=mock_trade_manager, base_currency='USD')
    assert portfolio_zero_cash.holdings['USD'] == 0

    # Test negative initial cash
    portfolio_negative_cash = Portfolio(initial_cash=-5000.0, portfolio_config={}, 
                                        signal_database=mock_signal_database, trade_manager=mock_trade_manager, base_currency='USD')
    assert portfolio_negative_cash.holdings['USD'] == -5000.0

    # Test empty portfolio configuration
    portfolio_empty_config = Portfolio(initial_cash=10000.0, portfolio_config={}, 
                                       signal_database=mock_signal_database, trade_manager=mock_trade_manager, base_currency='USD')
    assert portfolio_empty_config.holdings['USD'] == 10000.0
    assert portfolio_empty_config.history == []
    initial_cash = 10000.0
    portfolio = Portfolio(initial_cash=initial_cash, portfolio_config={}, 
                          signal_database=mock_signal_database, trade_manager=mock_trade_manager, base_currency='USD')

    # Test that the logger is set up and logging the initialization
    assert portfolio.logger.name == 'portfolio'

def test_validate_portfolio_config_valid(valid_portfolio_config):
    # Test that valid config doesn't raise any exception
    portfolio = Portfolio(initial_cash=10000.0, portfolio_config=valid_portfolio_config, 
                          signal_database=None, trade_manager=None, base_currency='USD')
    portfolio._validate_portfolio_config(valid_portfolio_config)

def test_validate_portfolio_config_invalid_type():
    invalid_config = ['invalid_config']  # Non-dict type
    portfolio = Portfolio(initial_cash=10000.0, portfolio_config={}, 
                          signal_database=None, trade_manager=None, base_currency='USD')
    with pytest.raises(TypeError, match="Portfolio configuration must be a dictionary"):
        portfolio._validate_portfolio_config(invalid_config)

def test_validate_portfolio_config_asset_key_not_string():
    invalid_config = {123: {'market_type': 'spot', 'fees': {'entry': 0.001, 'exit': 0.001}, 'max_trades': 5, 'max_exposure': 0.5}}
    portfolio = Portfolio(initial_cash=10000.0, portfolio_config={}, 
                          signal_database=None, trade_manager=None, base_currency='USD')
    with pytest.raises(TypeError, match="Asset key must be a string"):
        portfolio._validate_portfolio_config(invalid_config)

def test_validate_portfolio_config_invalid_asset_config_type():
    invalid_config = {'BTC': ['invalid_config']}
    portfolio = Portfolio(initial_cash=10000.0, portfolio_config={}, 
                          signal_database=None, trade_manager=None, base_currency='USD')
    with pytest.raises(TypeError, match="Configuration for asset BTC must be a dictionary"):
        portfolio._validate_portfolio_config(invalid_config)

def test_validate_portfolio_config_missing_required_keys():
    invalid_config = {'BTC': {'market_type': 'spot', 'fees': {'entry': 0.001}}}  # Missing 'max_trades' and 'max_exposure'
    portfolio = Portfolio(initial_cash=10000.0, portfolio_config={}, 
                          signal_database=None, trade_manager=None, base_currency='USD')
    
    with pytest.raises(ValueError) as excinfo:
        portfolio._validate_portfolio_config(invalid_config)
    
    error_message = str(excinfo.value)
    assert "Invalid configuration for asset BTC" in error_message
    assert "Missing required keys" in error_message
    assert "max_trades" in error_message
    assert "max_exposure" in error_message

def test_validate_portfolio_config_invalid_market_type():
    invalid_config = {'BTC': {'market_type': 'invalid_type', 'fees': {'entry': 0.001, 'exit': 0.001}, 'max_trades': 5, 'max_exposure': 0.5}}
    portfolio = Portfolio(initial_cash=10000.0, portfolio_config={}, 
                          signal_database=None, trade_manager=None, base_currency='USD')
    with pytest.raises(ValueError, match="Invalid market type for asset BTC: invalid_type"):
        portfolio._validate_portfolio_config(invalid_config)

def test_validate_portfolio_config_invalid_fee_structure():
    # Missing 'entry' and 'exit' in fees
    invalid_config = {'BTC': {'market_type': 'spot', 'fees': {}, 'max_trades': 5, 'max_exposure': 0.5}}
    portfolio = Portfolio(initial_cash=10000.0, portfolio_config={}, 
                          signal_database=None, trade_manager=None, base_currency='USD')
    with pytest.raises(ValueError, match="Invalid fee structure for asset BTC"):
        portfolio._validate_portfolio_config(invalid_config)

def test_validate_portfolio_config_invalid_max_trades():
    # max_trades is negative
    invalid_config = {'BTC': {'market_type': 'spot', 'fees': {'entry': 0.001, 'exit': 0.001}, 'max_trades': -1, 'max_exposure': 0.5}}
    portfolio = Portfolio(initial_cash=10000.0, portfolio_config={}, 
                          signal_database=None, trade_manager=None, base_currency='USD')
    with pytest.raises(ValueError, match="Invalid max_trades for asset BTC: -1"):
        portfolio._validate_portfolio_config(invalid_config)

def test_validate_portfolio_config_invalid_max_exposure():
    # max_exposure greater than 1
    invalid_config = {'BTC': {'market_type': 'spot', 'fees': {'entry': 0.001, 'exit': 0.001}, 'max_trades': 5, 'max_exposure': 1.5}}
    portfolio = Portfolio(initial_cash=10000.0, portfolio_config={}, 
                          signal_database=None, trade_manager=None, base_currency='USD')
    with pytest.raises(ValueError, match="Invalid max_exposure for asset BTC: 1.5"):
        portfolio._validate_portfolio_config(invalid_config)

    # max_exposure is negative
    invalid_config_negative = {'BTC': {'market_type': 'spot', 'fees': {'entry': 0.001, 'exit': 0.001}, 'max_trades': 5, 'max_exposure': -0.2}}
    with pytest.raises(ValueError, match="Invalid max_exposure for asset BTC: -0.2"):
        portfolio._validate_portfolio_config(invalid_config_negative)

def test_process_valid_buy_signal(portfolio, mock_signal_database, mock_trade_manager):
    signals = [{'signal_id': '1', 'asset_name': 'BTC', 'action': 'buy', 'amount': '0.1'}]
    market_prices = {'BTC': 50000}
    timestamp = 1625097600

    mock_signal_database.validate_signal.return_value = True
    portfolio.validate_portfolio_constraints = Mock(return_value=True)
    portfolio._open_trade = Mock()
    mock_trade_manager.get_trade.return_value = []

    portfolio.process_signals(signals, market_prices, timestamp)

    mock_signal_database.validate_signal.assert_called_once()
    portfolio._open_trade.assert_called_once()
    assert len(portfolio.history) == 1

def test_process_valid_sell_signal(portfolio, mock_signal_database, mock_trade_manager):
    portfolio.holdings = {'USD': 10000, 'BTC': 1}
    signals = [{'signal_id': '1', 'asset_name': 'BTC', 'action': 'sell', 'amount': '0.5'}]
    market_prices = {'BTC': 50000}
    timestamp = 1625097600

    mock_signal_database.validate_signal.return_value = True
    portfolio.validate_portfolio_constraints = Mock(return_value=True)
    portfolio._open_trade = Mock()
    mock_trade_manager.get_trade.return_value = []

    portfolio.process_signals(signals, market_prices, timestamp)

    portfolio._open_trade.assert_called_once()
    assert len(portfolio.history) == 1

def test_process_valid_close_signal(portfolio, mock_signal_database, mock_trade_manager):
    signals = [{'signal_id': '1', 'asset_name': 'BTC', 'action': 'close', 'trade_id': 1}]
    market_prices = {'BTC': 50000}
    timestamp = 1625097600

    mock_signal_database.validate_signal.return_value = True
    portfolio._close_trade = Mock()
    mock_trade_manager.get_trade.return_value = []

    portfolio.process_signals(signals, market_prices, timestamp)

    portfolio._close_trade.assert_called_once()
    assert len(portfolio.history) == 1

def test_process_invalid_signal(portfolio, mock_signal_database, mock_trade_manager):
    signals = [{'signal_id': '1', 'asset_name': 'BTC', 'action': 'invalid'}]
    market_prices = {'BTC': 50000}
    timestamp = 1625097600

    mock_signal_database.validate_signal.return_value = False
    mock_trade_manager.get_trade.return_value = []

    portfolio.process_signals(signals, market_prices, timestamp)

    mock_signal_database.update_signal_field.assert_called_once_with(
        signal_id='1', field_name='status', new_value='rejected'
    )
    assert len(portfolio.history) == 1

def test_process_multiple_signals(portfolio, mock_signal_database, mock_trade_manager):
    signals = [
        {'signal_id': '1', 'asset_name': 'BTC', 'action': 'buy', 'amount': '0.1'},
        {'signal_id': '2', 'asset_name': 'ETH', 'action': 'sell', 'amount': '0.5'},
        {'signal_id': '3', 'asset_name': 'LTC', 'action': 'close', 'trade_id': 1}
    ]
    market_prices = {'BTC': 50000, 'ETH': 3000, 'LTC': 200}
    timestamp = 1625097600

    mock_signal_database.validate_signal.return_value = True
    portfolio.validate_portfolio_constraints = Mock(return_value=True)
    portfolio._open_trade = Mock()
    portfolio._close_trade = Mock()
    mock_trade_manager.get_trade.return_value = []

    portfolio.process_signals(signals, market_prices, timestamp)

    assert portfolio._open_trade.call_count == 2
    portfolio._close_trade.assert_called_once()
    assert len(portfolio.history) == 1

def test_process_signal_failing_portfolio_constraints(portfolio, mock_signal_database, mock_trade_manager):
    signals = [{'signal_id': '1', 'asset_name': 'BTC', 'action': 'buy', 'amount': '0.1'}]
    market_prices = {'BTC': 50000}
    timestamp = 1625097600

    mock_signal_database.validate_signal.return_value = True
    portfolio.validate_portfolio_constraints = Mock(return_value=False)
    mock_trade_manager.get_trade.return_value = []

    portfolio.process_signals(signals, market_prices, timestamp)

    mock_signal_database.update_signal_field.assert_called_once_with(
        signal_id='1', field_name='status', new_value='rejected'
    )
    assert len(portfolio.history) == 1

def test_process_signal_exception_handling(portfolio, mock_signal_database, mock_trade_manager):
    signals = [{'signal_id': '1', 'asset_name': 'BTC', 'action': 'buy', 'amount': '0.1'}]
    market_prices = {'BTC': 50000}
    timestamp = 1625097600

    mock_signal_database.validate_signal.side_effect = Exception("Test exception")
    mock_trade_manager.get_trade.return_value = []

    portfolio.process_signals(signals, market_prices, timestamp)

    # Ensure that the exception was caught and the function didn't crash
    assert len(portfolio.history) == 1

def test_max_exposure_limit_reached(portfolio, monkeypatch):
    # Arrange
    signal = {'asset_name': 'BTC', 'action': 'buy', 'amount': '0.1'}
    market_prices = {'BTC': 50000}
    
    # Mock methods
    monkeypatch.setattr(portfolio, 'get_total_value', lambda market_prices: 100000)
    monkeypatch.setattr(portfolio, 'get_asset_exposure', lambda asset, market_prices: 0.45)
    portfolio.trade_manager.get_trade.return_value = []

    # Print debug information
    print(f"Portfolio config: {portfolio.portfolio_config}")
    print(f"BTC max_exposure: {portfolio.portfolio_config['BTC']['max_exposure']}")
    
    # Act
    result = portfolio.validate_portfolio_constraints(signal, market_prices)

    # Print more debug information
    print(f"Validation result: {result}")
    print(f"Current exposure: 0.45")
    print(f"New exposure: {0.1 * 50000 / 100000}")
    print(f"Total exposure: {0.45 + 0.1 * 50000 / 100000}")

    # Assert
    assert result is True, "Expected True (max exposure limit exactly reached)"

    # Test with slightly higher amount to ensure it fails when exceeding max_exposure
    signal['amount'] = '0.11'  # This should push it over the limit
    result = portfolio.validate_portfolio_constraints(signal, market_prices)
    assert result is False, "Expected False (max exposure limit exceeded)"

def test_spot_market_buy_insufficient_funds(portfolio):
    # Arrange
    portfolio.holdings = {'USD': 1000}
    signal = {'asset_name': 'BTC', 'action': 'buy', 'amount': '0.1'}
    market_prices = {'BTC': 50000}
    portfolio.trade_manager.get_trade.return_value = []

    # Act
    result = portfolio.validate_portfolio_constraints(signal, market_prices)

    # Assert
    assert result is False

def test_spot_market_sell_insufficient_holdings(portfolio):
    # Arrange
    portfolio.holdings = {'USD': 1000, 'BTC': 0.05}
    signal = {'asset_name': 'BTC', 'action': 'sell', 'amount': '0.1'}
    market_prices = {'BTC': 50000}
    portfolio.trade_manager.get_trade.return_value = []

    # Act
    result = portfolio.validate_portfolio_constraints(signal, market_prices)

    # Assert
    assert result is False

def test_successful_spot_market_buy(portfolio, mocker):
    # Arrange
    portfolio.holdings = {'USD': 10000}
    signal = {'asset_name': 'BTC', 'action': 'buy', 'amount': '0.1'}
    market_prices = {'BTC': 50000}
    mocker.patch.object(portfolio, 'get_total_value', return_value=100000)
    mocker.patch.object(portfolio, 'get_asset_exposure', return_value=0)
    portfolio.trade_manager.get_trade.return_value = []

    # Act
    result = portfolio.validate_portfolio_constraints(signal, market_prices)

    # Assert
    assert result is True

def test_successful_spot_market_sell(portfolio, mocker):
    # Arrange
    portfolio.holdings = {'USD': 10000, 'BTC': 0.2}
    signal = {'asset_name': 'BTC', 'action': 'sell', 'amount': '0.1'}
    market_prices = {'BTC': 50000}
    mocker.patch.object(portfolio, 'get_total_value', return_value=100000)
    mocker.patch.object(portfolio, 'get_asset_exposure', return_value=0.1)
    portfolio.trade_manager.get_trade.return_value = []

    # Act
    result = portfolio.validate_portfolio_constraints(signal, market_prices)

    # Assert
    assert result is True

def test_invalid_asset(portfolio):
    # Arrange
    signal = {'asset_name': 'INVALID', 'action': 'buy', 'amount': '0.1'}
    market_prices = {'INVALID': 100}

    # Act
    result = portfolio.validate_portfolio_constraints(signal, market_prices)

    # Assert
    assert result is False

def test_error_handling(portfolio, mocker):
    # Arrange
    signal = {'asset_name': 'BTC', 'action': 'buy', 'amount': '0.1'}
    market_prices = {'BTC': 50000}
    mocker.patch.object(portfolio, 'get_fee', side_effect=Exception("Test error"))

    # Act
    result = portfolio.validate_portfolio_constraints(signal, market_prices)

    # Assert
    assert result is False

def test_open_trade_buy_spot_market(portfolio, mock_trade_manager, mock_signal_database):
    # Arrange
    signal = {
        'signal_id': 1,
        'action': 'buy',
        'asset_name': 'BTC',
        'amount': 0.5,
        'price': 50000.0,
        'stop_loss': 49000.0,
        'take_profit': 51000.0,
        'trailing_stop': 500.0
    }
    asset_quantity = 0.5
    base_amount = 25000.0
    asset_price = 50000.0
    timestamp = 1620000000000

    # Mock necessary functions
    mock_trade_manager.open_trade.return_value = {'id': 10}
    portfolio.holdings['USD'] = 50000.0
    portfolio.holdings['BTC'] = 0.0

    # Act
    portfolio._open_trade(signal, asset_quantity, base_amount, asset_price, timestamp)

    # Assert
    mock_trade_manager.open_trade.assert_called_once_with(
        asset_name='BTC',
        base_currency='USD',
        asset_amount=asset_quantity,
        base_amount=base_amount,
        entry_price=asset_price,
        entry_timestamp=timestamp,
        entry_fee=0.001 * base_amount,
        stop_loss=49000.0,
        take_profit=51000.0,
        trailing_stop=500.0,
        direction='buy',
        entry_reason='buy_signal'
    )

    # Check that holdings were updated
    assert portfolio.holdings['USD'] == 50000.0 - base_amount - (0.001 * base_amount)
    assert portfolio.holdings['BTC'] == 0.5

    # Check that signal was updated in the database
    mock_signal_database.update_signal_field.assert_has_calls([
        call(signal_id=1, field_name='status', new_value='executed'),
        call(signal_id=1, field_name='trade_id', new_value=10)
    ])

def test_open_trade_sell_spot_market(portfolio, mock_trade_manager, mock_signal_database):
    # Arrange
    signal = {
        'signal_id': 2,
        'action': 'sell',
        'asset_name': 'BTC',
        'amount': 0.5,
        'price': 50000.0
    }
    asset_quantity = 0.5
    base_amount = 25000.0
    asset_price = 50000.0
    timestamp = 1620000000000

    # Mock necessary functions
    mock_trade_manager.open_trade.return_value = {'id': 11}
    portfolio.holdings['USD'] = 10000.0
    portfolio.holdings['BTC'] = 0.5

    # Act
    portfolio._open_trade(signal, asset_quantity, base_amount, asset_price, timestamp)

    # Assert
    mock_trade_manager.open_trade.assert_called_once()

    # Check that holdings were updated, allowing for minor floating-point differences
    assert portfolio.holdings['USD'] == pytest.approx(10000.0 + base_amount - (0.001 * base_amount), rel=1e-5)

def test_open_trade_futures_market(portfolio, mock_trade_manager, mock_signal_database):
    # Arrange
    signal = {
        'signal_id': 3,
        'action': 'buy',
        'asset_name': 'ETH',
        'amount': 1.0,
        'price': 3000.0
    }
    asset_quantity = 1.0
    base_amount = 3000.0
    asset_price = 3000.0
    timestamp = 1620000000000

    # Mock necessary functions
    mock_trade_manager.open_trade.return_value = {'id': 12}

    # Act
    portfolio._open_trade(signal, asset_quantity, base_amount, asset_price, timestamp)

    # Assert
    mock_trade_manager.open_trade.assert_called_once()

    # No changes to holdings expected for futures market
    assert portfolio.holdings['USD'] == 10000.0
    assert 'ETH' not in portfolio.holdings

    # Check that signal was updated in the database
    mock_signal_database.update_signal_field.assert_has_calls([
        call(signal_id=3, field_name='status', new_value='executed'),
        call(signal_id=3, field_name='trade_id', new_value=12)
    ])

def test_open_trade_with_stop_loss_take_profit_trailing_stop(portfolio, mock_trade_manager, mock_signal_database):
    # Arrange
    signal = {
        'signal_id': 4,
        'action': 'buy',
        'asset_name': 'BTC',
        'amount': 0.5,
        'price': 50000.0,
        'stop_loss': 49000.0,
        'take_profit': 51000.0,
        'trailing_stop': 500.0
    }
    asset_quantity = 0.5
    base_amount = 25000.0
    asset_price = 50000.0
    timestamp = 1620000000000

    # Mock necessary functions
    mock_trade_manager.open_trade.return_value = {'id': 13}

    # Act
    portfolio._open_trade(signal, asset_quantity, base_amount, asset_price, timestamp)

    # Assert
    mock_trade_manager.open_trade.assert_called_once_with(
        asset_name='BTC',
        base_currency='USD',
        asset_amount=asset_quantity,
        base_amount=base_amount,
        entry_price=asset_price,
        entry_timestamp=timestamp,
        entry_fee=0.001 * base_amount,
        stop_loss=49000.0,
        take_profit=51000.0,
        trailing_stop=500.0,
        direction='buy',
        entry_reason='buy_signal'
    )

def test_open_trade_with_missing_fields(portfolio, mock_trade_manager, mock_signal_database):
    # Arrange
    signal = {
        'signal_id': 5,
        'action': 'buy',
        'asset_name': 'BTC',
        'amount': 0.5,
        'price': 50000.0
    }
    asset_quantity = 0.5
    base_amount = 25000.0
    asset_price = 50000.0
    timestamp = 1620000000000

    # Mock necessary functions
    mock_trade_manager.open_trade.return_value = {'id': 14}

    # Act
    portfolio._open_trade(signal, asset_quantity, base_amount, asset_price, timestamp)

    # Assert
    mock_trade_manager.open_trade.assert_called_once_with(
        asset_name='BTC',
        base_currency='USD',
        asset_amount=asset_quantity,
        base_amount=base_amount,
        entry_price=asset_price,
        entry_timestamp=timestamp,
        entry_fee=0.001 * base_amount,
        stop_loss=0,  # Default value
        take_profit=0,  # Default value
        trailing_stop=0,  # Default value
        direction='buy',
        entry_reason='buy_signal'  # Default reason
    )

def test_open_trade_raises_exception(portfolio, mock_trade_manager, mock_signal_database):
    # Arrange
    signal = {
        'signal_id': 6,
        'action': 'buy',
        'asset_name': 'BTC',
        'amount': 0.5,
        'price': 50000.0
    }
    asset_quantity = 0.5
    base_amount = 25000.0
    asset_price = 50000.0
    timestamp = 1620000000000

    # Mock necessary functions
    mock_trade_manager.open_trade.side_effect = Exception("Trade manager failed")

    # Act / Assert
    with pytest.raises(Exception, match="Trade manager failed"):
        portfolio._open_trade(signal, asset_quantity, base_amount, asset_price, timestamp)

    # Assert no updates to signal or portfolio were made
    mock_signal_database.update_signal_field.assert_not_called()

def test_close_trade_success(portfolio, mock_trade_manager, mock_signal_database):
    # Arrange
    signal = {
        'signal_id': 1,
        'trade_id': 10,
        'asset_name': 'BTC',
        'price': 50000.0,
        'reason': 'take_profit'
    }
    trade = {
        'trade_id': 10,
        'status': 'open',
        'asset_name': 'BTC',
        'base_amount': 10000.0
    }
    closed_trade = {
        'id': 10,
        'asset_name': 'BTC',
        'asset_amount': 0.5,
        'base_amount': 25000.0,
        'exit_fee': 10.0,
        'direction': 'sell'
    }

    # Ensure that BTC exists in the portfolio holdings
    portfolio.holdings['BTC'] = 0.5

    mock_trade_manager.get_trade.return_value = trade
    mock_trade_manager.close_trade.return_value = closed_trade

    # Act
    portfolio._close_trade(signal, timestamp=1620000000000)

    # Assert
    mock_trade_manager.close_trade.assert_called_once_with(
        trade_id=10,
        exit_price=50000.0,
        exit_timestamp=1620000000000,
        exit_fee=10.0,
        exit_reason='take_profit'
    )
    
    # Fix the call assertions by matching the keyword arguments
    mock_signal_database.update_signal_field.assert_has_calls([
        call(signal_id=1, field_name='status', new_value='executed'),
        call(signal_id=1, field_name='trade_id', new_value=10)
    ])

def test_close_trade_not_found(portfolio, mock_trade_manager, mock_signal_database):
    # Arrange
    signal = {
        'signal_id': 1,
        'trade_id': 10,
        'asset_name': 'BTC',
        'price': 50000.0,
        'reason': 'take_profit'
    }
    mock_trade_manager.get_trade.return_value = None  # Trade not found

    # Act
    portfolio._close_trade(signal, timestamp=1620000000000)
    
    # Assert
    mock_trade_manager.close_trade.assert_not_called()
    mock_signal_database.update_signal_field.assert_not_called()

def test_close_trade_already_closed(portfolio, mock_trade_manager, mock_signal_database):
    # Arrange
    signal = {
        'signal_id': 1,
        'trade_id': 10,
        'asset_name': 'BTC',
        'price': 50000.0,
        'reason': 'take_profit'
    }
    trade = {
        'trade_id': 10,
        'status': 'closed',  # Already closed
        'asset_name': 'BTC',
        'base_amount': 10000.0
    }
    mock_trade_manager.get_trade.return_value = trade

    # Act
    portfolio._close_trade(signal, timestamp=1620000000000)
    
    # Assert
    mock_trade_manager.close_trade.assert_not_called()
    mock_signal_database.update_signal_field.assert_not_called()

def test_close_trade_failure(portfolio, mock_trade_manager, mock_signal_database):
    # Arrange
    signal = {
        'signal_id': 1,
        'trade_id': 10,
        'asset_name': 'BTC',
        'price': 50000.0,
        'reason': 'take_profit'
    }
    trade = {
        'trade_id': 10,
        'status': 'open',
        'asset_name': 'BTC',
        'base_amount': 10000.0
    }
    mock_trade_manager.get_trade.return_value = trade
    mock_trade_manager.close_trade.return_value = None  # Failure to close trade

    # Act
    portfolio._close_trade(signal, timestamp=1620000000000)
    
    # Assert
    mock_signal_database.update_signal_field.assert_not_called()
    # Ensure no portfolio update

def test_close_trade_with_exception(portfolio, mock_trade_manager, mock_signal_database):
    # Arrange
    signal = {
        'signal_id': 1,
        'trade_id': 10,
        'asset_name': 'BTC',
        'price': 50000.0,
        'reason': 'take_profit'
    }
    trade = {
        'trade_id': 10,
        'status': 'open',
        'asset_name': 'BTC',
        'base_amount': 10000.0
    }
    mock_trade_manager.get_trade.return_value = trade
    mock_trade_manager.close_trade.side_effect = Exception("Close trade failure")

    # Act & Assert
    with pytest.raises(Exception, match="Close trade failure"):
        portfolio._close_trade(signal, timestamp=1620000000000)

    # Ensure no updates were called due to the exception
    mock_signal_database.update_signal_field.assert_not_called()

def test_update_holdings_sell_spot_trade(portfolio):
    # Arrange
    trade = {
        'asset_name': 'BTC',
        'asset_amount': 0.5,
        'base_amount': 25000.0,
        'exit_fee': 10.0,
        'direction': 'sell'
    }
    portfolio.holdings['BTC'] = 0.5
    portfolio.holdings['USD'] = 10000.0

    # Act
    portfolio.update_holdings(trade)

    # Assert
    assert portfolio.holdings['USD'] == pytest.approx(34990.0, rel=1e-5)  # USD increases by 25000 - 10
    assert 'BTC' not in portfolio.holdings  # BTC should be removed since it was sold entirely

def test_update_holdings_buy_spot_trade(portfolio):
    # Arrange
    trade = {
        'asset_name': 'BTC',
        'asset_amount': 0.5,
        'base_amount': 25000.0,
        'exit_fee': 10.0,
        'direction': 'buy'
    }
    portfolio.holdings['USD'] = 50000.0

    # Act
    portfolio.update_holdings(trade)

    # Assert
    assert portfolio.holdings['USD'] == pytest.approx(24990.0, rel=1e-5)  # USD decreases by 25000 + 10
    assert portfolio.holdings['BTC'] == 0.5  # BTC holdings should increase

def test_update_holdings_sell_insufficient_assets(portfolio):
    # Arrange
    trade = {
        'asset_name': 'BTC',
        'asset_amount': 1.0,  # Trying to sell more than available
        'base_amount': 25000.0,
        'exit_fee': 10.0,
        'direction': 'sell'
    }
    portfolio.holdings['BTC'] = 0.5

    # Act and Assert
    with pytest.raises(ValueError, match="Not enough BTC holdings to sell"):
        portfolio.update_holdings(trade)

def test_update_holdings_unsupported_market_type(portfolio):
    # Arrange
    trade = {
        'asset_name': 'BTC',
        'asset_amount': 0.5,
        'base_amount': 25000.0,
        'exit_fee': 10.0,
        'direction': 'sell'
    }
    portfolio.portfolio_config['BTC']['market_type'] = 'options'  # Unsupported market type

    # Act and Assert
    with pytest.raises(ValueError, match="Unsupported market type options for asset BTC"):
        portfolio.update_holdings(trade)

def test_update_holdings_futures_market_handling(portfolio):
    # Arrange
    trade = {
        'asset_name': 'BTC',
        'asset_amount': 0.5,
        'base_amount': 25000.0,
        'exit_fee': 10.0,
        'direction': 'sell'
    }
    portfolio.portfolio_config['BTC']['market_type'] = 'futures'

    # Act
    portfolio.update_holdings(trade)

    # Assert
    # In futures market, no changes to holdings should occur
    assert portfolio.holdings == {portfolio.base_currency: 10000.0}

def test_update_holdings_missing_trade_data(portfolio):
    # Arrange
    trade = {
        'base_amount': 25000.0,
        'exit_fee': 10.0,
        'direction': 'sell'
    }  # Missing 'asset_name' and 'asset_amount'

    # Act and Assert
    with pytest.raises(KeyError):  # Expecting an error due to missing trade data
        portfolio.update_holdings(trade)

def test_get_total_value_with_multiple_assets(portfolio):
    # Arrange
    portfolio.holdings = {
        'USD': 10000.0,
        'BTC': 0.5,
        'ETH': 2.0
    }
    market_prices = {
        'BTC': 50000.0,
        'ETH': 3000.0
    }

    # Act
    total_value = portfolio.get_total_value(market_prices)

    # Assert
    expected_value = 10000.0 + (0.5 * 50000.0) + (2.0 * 3000.0)
    assert total_value == pytest.approx(expected_value, rel=1e-5)

def test_get_total_value_with_only_cash(portfolio):
    # Arrange
    portfolio.holdings = {
        'USD': 10000.0
    }
    market_prices = {}

    # Act
    total_value = portfolio.get_total_value(market_prices)

    # Assert
    assert total_value == 10000.0

def test_get_total_value_with_zero_assets_and_zero_cash(portfolio):
    # Arrange
    portfolio.holdings = {
        'USD': 0.0
    }
    market_prices = {}

    # Act
    total_value = portfolio.get_total_value(market_prices)

    # Assert
    assert total_value == 0.0

def test_get_total_value_with_unpriced_assets(portfolio):
    # Arrange
    portfolio.holdings = {
        'USD': 10000.0,
        'BTC': 0.5,
        'ETH': 2.0
    }
    market_prices = {
        'BTC': 50000.0  # ETH is missing from market prices
    }

    # Act
    total_value = portfolio.get_total_value(market_prices)

    # Assert
    expected_value = 10000.0 + (0.5 * 50000.0)  # Only BTC is priced
    assert total_value == pytest.approx(expected_value, rel=1e-5)

def test_get_total_value_with_empty_market_prices(portfolio):
    # Arrange
    portfolio.holdings = {
        'USD': 10000.0,
        'BTC': 0.5,
        'ETH': 2.0
    }
    market_prices = {}  # No market prices available

    # Act
    total_value = portfolio.get_total_value(market_prices)

    # Assert
    assert total_value == 10000.0  # Only cash value is included

def test_get_total_value_with_non_floating_quantities(portfolio):
    # Arrange
    portfolio.holdings = {
        'USD': 10000.0,
        'BTC': 1,  # Integer value
        'ETH': 2  # Integer value
    }
    market_prices = {
        'BTC': 50000.0,
        'ETH': 3000.0
    }

    # Act
    total_value = portfolio.get_total_value(market_prices)

    # Assert
    expected_value = 10000.0 + (1 * 50000.0) + (2 * 3000.0)
    assert total_value == pytest.approx(expected_value, rel=1e-5)

def test_get_fee_entry_fee(portfolio):
    # Act
    fee = portfolio.get_fee('BTC', 'entry')

    # Assert
    assert fee == 0.001  # BTC entry fee as per valid_portfolio_config


def test_get_fee_exit_fee(portfolio):
    # Act
    fee = portfolio.get_fee('BTC', 'exit')

    # Assert
    assert fee == 0.001  # BTC exit fee as per valid_portfolio_config


def test_get_fee_asset_with_no_fee(portfolio):
    # Act
    fee = portfolio.get_fee('LTC', 'entry')  # LTC does not exist in config

    # Assert
    assert fee == 0.0  # Default fee when asset is not in the portfolio config


def test_get_fee_invalid_fee_type(portfolio):
    # Act
    fee = portfolio.get_fee('BTC', 'non_existent_fee_type')

    # Assert
    assert fee == 0.0  # Should return 0 if fee type is not found

def test_get_asset_exposure_valid_asset(portfolio):
    # Arrange
    portfolio.holdings = {'USD': 10000.0, 'BTC': 0.5}
    market_prices = {'BTC': 50000.0}

    # Act
    exposure = portfolio.get_asset_exposure('BTC', market_prices)

    # Assert
    total_value = 10000.0 + (0.5 * 50000.0)
    expected_exposure = (0.5 * 50000.0) / total_value
    assert exposure == pytest.approx(expected_exposure, rel=1e-5)


def test_get_asset_exposure_asset_not_in_holdings(portfolio):
    # Arrange
    portfolio.holdings = {'USD': 10000.0}
    market_prices = {'BTC': 50000.0}

    # Act
    exposure = portfolio.get_asset_exposure('BTC', market_prices)

    # Assert
    assert exposure == 0.0  # No BTC in holdings


def test_get_asset_exposure_asset_not_in_market_prices(portfolio):
    # Arrange
    portfolio.holdings = {'USD': 10000.0, 'BTC': 0.5}
    market_prices = {}  # No market prices for BTC

    # Act
    exposure = portfolio.get_asset_exposure('BTC', market_prices)

    # Assert
    assert exposure == 0.0  # No BTC price, exposure is 0


def test_get_asset_exposure_zero_portfolio_value(portfolio):
    # Arrange
    portfolio.holdings = {'BTC': 0.5}
    market_prices = {'BTC': 50000.0}

    # Mock get_total_value to return 0
    portfolio.get_total_value = Mock(return_value=0)

    # Act
    exposure = portfolio.get_asset_exposure('BTC', market_prices)

    # Assert
    assert exposure == 0.0  # Total portfolio value is 0, so exposure is 0

def test_store_history_with_empty_holdings(portfolio):
    # Arrange
    portfolio.holdings = {}
    timestamp = 1620000000000

    # Act
    portfolio.store_history(timestamp)

    # Assert
    assert len(portfolio.history) == 1
    assert portfolio.history[0]['timestamp'] == timestamp
    assert portfolio.history[0]['holdings'] == {}  # Empty holdings should be recorded


def test_store_history_with_non_empty_holdings(portfolio):
    # Arrange
    portfolio.holdings = {'USD': 10000.0, 'BTC': 0.5}
    timestamp = 1620000000000

    # Act
    portfolio.store_history(timestamp)

    # Assert
    assert len(portfolio.history) == 1
    assert portfolio.history[0]['timestamp'] == timestamp
    assert portfolio.history[0]['holdings'] == {'USD': 10000.0, 'BTC': 0.5}


def test_store_history_multiple_calls(portfolio):
    # Arrange
    portfolio.holdings = {'USD': 10000.0, 'BTC': 0.5}
    timestamp1 = 1620000000000
    timestamp2 = 1620000005000

    # Act
    portfolio.store_history(timestamp1)
    portfolio.holdings['BTC'] = 1.0  # Update holdings
    portfolio.store_history(timestamp2)

    # Assert
    assert len(portfolio.history) == 2
    assert portfolio.history[0]['timestamp'] == timestamp1
    assert portfolio.history[1]['timestamp'] == timestamp2
    assert portfolio.history[0]['holdings'] == {'USD': 10000.0, 'BTC': 0.5}
    assert portfolio.history[1]['holdings'] == {'USD': 10000.0, 'BTC': 1.0}
