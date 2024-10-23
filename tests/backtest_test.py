import os
import sys
import pytest
import logging
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest_engine import BacktestEngine
from src.strategy.depeg_strategy import DepegStrategy
from src.metrics import MetricsModule
from unittest.mock import Mock, call
from src.portfolio import Portfolio
from src.trade_manager import TradeManager
from src.signal_database import SignalDatabase
from src.data_preprocessor import DataPreprocessor

@pytest.fixture
def assets():
    """Fixture for asset CSV paths."""
    return {
        "BTCUSD": "path/to/btcusd.csv",
        "ETHUSD": "path/to/ethusd.csv"
    }

@pytest.fixture
def strategies():
    """Fixture for mock strategies."""
    mock_strategy_btc = Mock(spec=DepegStrategy)
    mock_strategy_btc.get_required_indicators.return_value = ["SMA_50", "SMA_200"]
    mock_strategy_btc.generate_signal.return_value = {"action": "buy", "amount": 1}
    mock_strategy_btc.trailing_stop_percent = 5.0

    mock_strategy_eth = Mock(spec=DepegStrategy)
    mock_strategy_eth.get_required_indicators.return_value = ["SMA_50", "SMA_200"]
    mock_strategy_eth.generate_signal.return_value = {"action": "buy", "amount": 1}
    mock_strategy_eth.trailing_stop_percent = 5.0

    return {
        "BTCUSD": mock_strategy_btc,
        "ETHUSD": mock_strategy_eth
    }

@pytest.fixture
def portfolio():
    """Fixture for a mock Portfolio instance."""
    mock_portfolio = Mock(spec=Portfolio)
    mock_portfolio.get_total_value.return_value = 100000
    mock_portfolio.holdings = {"USD": 50000, "BTC": 1.0, "ETH": 10.0}
    return mock_portfolio

@pytest.fixture
def trade_manager():
    """Fixture for a mock TradeManager instance."""
    mock_trade_manager = Mock(spec=TradeManager)
    mock_trade_manager.get_trade.return_value = [
        {"id": 1, "asset_name": "BTCUSD", "stop_loss": 30000, "take_profit": 40000, "direction": "buy", "asset_amount": 1.0, "trailing_stop": None},
        {"id": 2, "asset_name": "ETHUSD", "stop_loss": 1500, "take_profit": 2000, "direction": "buy", "asset_amount": 2.0, "trailing_stop": None}
    ]
    return mock_trade_manager

@pytest.fixture
def metrics():
    """Fixture for a mock MetricsModule instance."""
    return Mock(spec=MetricsModule)

@pytest.fixture
def signal_database():
    """Fixture for a mock SignalDatabase instance."""
    mock_signal_database = Mock(spec=SignalDatabase)
    return mock_signal_database

@pytest.fixture
def logger():
    """Fixture for a mock logger."""
    return Mock(spec=logging.Logger)

@pytest.fixture
def data_preprocessor():
    """Fixture for a mock DataPreprocessor instance."""
    mock_preprocessor = Mock(spec=DataPreprocessor)
    mock_preprocessor.preprocess_data.return_value = pd.DataFrame({
        "timestamp": [1, 2, 3, 4, 5],
        "close": [30000, 31000, 32000, 33000, 34000]
    }).set_index("timestamp")
    return mock_preprocessor

@pytest.fixture
def backtest_engine(assets, strategies, portfolio, trade_manager, metrics, signal_database, logger):
    """Fixture for initializing the BacktestEngine."""
    return BacktestEngine(
        assets=assets,
        strategies=strategies,
        portfolio=portfolio,
        trade_manager=trade_manager,
        metrics=metrics,
        signal_database=signal_database,
        logger=logger
    )

def test_run_backtest_successful(backtest_engine):
    # Arrange
    backtest_engine.unified_timestamps = [1, 2, 3]
    backtest_engine._process_timestamp = Mock()
    backtest_engine._close_all_open_trades = Mock()
    backtest_engine.metrics.run = Mock()

    logging.info("Starting test_run_backtest_successful")

    # Act
    backtest_engine.run_backtest()

    # Assert
    logging.info("Asserting calls for _process_timestamp and _close_all_open_trades")
    backtest_engine._process_timestamp.assert_has_calls([call(1), call(2), call(3)])
    backtest_engine._close_all_open_trades.assert_called_once_with(3)
    backtest_engine.metrics.run.assert_called_once()

def test_run_backtest_missing_timestamps(backtest_engine):
    # Arrange
    backtest_engine.unified_timestamps = []
    logging.info("Starting test_run_backtest_missing_timestamps with empty unified_timestamps")

    # Act & Assert
    with pytest.raises(ValueError, match="No timestamps available"):
        backtest_engine.run_backtest()

def test_run_backtest_signal_processing(backtest_engine):
    # Arrange
    backtest_engine.unified_timestamps = [1, 2]
    backtest_engine._process_timestamp = Mock()
    backtest_engine.market_data = {
        "BTCUSD": pd.DataFrame({"close": [30000, 31000]}, index=[1, 2]),
        "ETHUSD": pd.DataFrame({"close": [1500, 1600]}, index=[1, 2])
    }
    backtest_engine.trade_manager.get_trade.return_value = [
        {"id": 1, "asset_name": "BTCUSD", "stop_loss": 29000, "take_profit": 32000, "direction": "buy", "asset_amount": 1.0, "trailing_stop": None}
    ]
    logging.info("Starting test_run_backtest_signal_processing")

    # Act
    backtest_engine.run_backtest()

    # Assert
    logging.info("Asserting calls for _process_timestamp")
    backtest_engine._process_timestamp.assert_has_calls([call(1), call(2)])

def test_run_backtest_final_trade_closure(backtest_engine, trade_manager):
    # Arrange
    backtest_engine.unified_timestamps = [1, 2, 3]
    backtest_engine._process_timestamp = Mock()
    backtest_engine._close_all_open_trades = Mock()
    logging.info("Starting test_run_backtest_final_trade_closure")

    # Act
    backtest_engine.run_backtest()

    # Assert
    logging.info("Asserting call for _close_all_open_trades")
    backtest_engine._close_all_open_trades.assert_called_once_with(3)

def test_run_backtest_metrics_calculation(backtest_engine, metrics):
    # Arrange
    backtest_engine.unified_timestamps = [1, 2, 3]
    backtest_engine._process_timestamp = Mock()
    backtest_engine.metrics.run = Mock()
    backtest_engine.market_data = {
        "BTCUSD": pd.DataFrame({"close": [30000, 31000, 32000]}, index=[1, 2, 3]),
        "ETHUSD": pd.DataFrame({"close": [1500, 1600, 1700]}, index=[1, 2, 3])
    }
    logging.info("Starting test_run_backtest_metrics_calculation")

    # Act
    backtest_engine.run_backtest()

    # Assert
    logging.info("Asserting call for metrics.run")
    backtest_engine.metrics.run.assert_called_once()

def test_process_timestamp_generate_signals(backtest_engine, trade_manager):
    # Arrange
    timestamp = 1
    backtest_engine.market_data = {
        "BTCUSD": pd.DataFrame({"close": [30000]}, index=[timestamp])
    }
    backtest_engine._generate_strategy_signals = Mock(return_value=[{"action": "buy", "asset_name": "BTCUSD"}])
    backtest_engine._process_signals = Mock()
    backtest_engine.portfolio.store_history = Mock()
    logging.info("Starting test_process_timestamp_generate_signals")

    # Act
    backtest_engine._process_timestamp(timestamp)

    # Assert
    logging.info("Asserting calls for _generate_strategy_signals, _process_signals, and store_history")
    backtest_engine._generate_strategy_signals.assert_called()
    backtest_engine._process_signals.assert_called()
    backtest_engine.portfolio.store_history.assert_called_once_with(timestamp)

def test_process_timestamp_market_prices_and_signal_processing(backtest_engine):
    # Arrange
    timestamp = 1
    backtest_engine.market_data = {
        "BTCUSD": pd.DataFrame({"close": [30000]}, index=[timestamp])
    }
    backtest_engine._process_signals = Mock()
    backtest_engine.portfolio.store_history = Mock()
    logging.info("Starting test_process_timestamp_market_prices_and_signal_processing")

    # Act
    backtest_engine._process_timestamp(timestamp)

    # Assert
    logging.info("Asserting calls for _process_signals and store_history")
    backtest_engine._process_signals.assert_called()
    backtest_engine.portfolio.store_history.assert_called_once_with(timestamp)

def test_process_timestamp_store_portfolio_history(backtest_engine):
    # Arrange
    timestamp = 1
    backtest_engine.market_data = {
        "BTCUSD": pd.DataFrame({"close": [30000]}, index=[timestamp])
    }
    backtest_engine.portfolio.store_history = Mock()
    logging.info("Starting test_process_timestamp_store_portfolio_history")

    # Act
    backtest_engine._process_timestamp(timestamp)

    # Assert
    logging.info("Asserting call for store_history")
    backtest_engine.portfolio.store_history.assert_called_once_with(timestamp)

def test_check_stop_loss_take_profit_stop_loss_triggered(backtest_engine):
    # Arrange
    open_trades = [
        {"id": 1, "asset_name": "BTCUSD", "stop_loss": 29000, "take_profit": None, "direction": "buy", "asset_amount": 1.0}
    ]
    current_price = 28000
    timestamp = 1
    logging.info("Starting test_check_stop_loss_take_profit_stop_loss_triggered")

    # Act
    result = backtest_engine._check_stop_loss_take_profit(open_trades, current_price, timestamp)

    # Debugging output
    logging.info(f"Result from stop loss test: {result}")

    # Assert
    assert len(result) == 1
    assert result[0]["action"] == "close"
    assert result[0]["reason"] == "stop_loss", f"Expected 'stop_loss', but got {result[0]['reason']}"

def test_check_stop_loss_take_profit_take_profit_triggered(backtest_engine):
    # Arrange
    open_trades = [
        {"id": 1, "asset_name": "BTCUSD", "stop_loss": None, "take_profit": 32000, "direction": "buy", "asset_amount": 1.0}
    ]
    current_price = 33000
    timestamp = 1
    logging.info("Starting test_check_stop_loss_take_profit_take_profit_triggered")

    # Act
    result = backtest_engine._check_stop_loss_take_profit(open_trades, current_price, timestamp)

    # Debugging output
    logging.info(f"Result from take profit test: {result}")

    # Assert
    assert len(result) == 1
    assert result[0]["action"] == "close"
    assert result[0]["reason"] == "take_profit", f"Expected 'take_profit', but got {result[0]['reason']}"

def test_check_stop_loss_take_profit_no_trigger(backtest_engine):
    # Arrange
    open_trades = [
        {"id": 1, "asset_name": "BTCUSD", "stop_loss": 29000, "take_profit": 32000, "direction": "buy", "asset_amount": 1.0}
    ]
    current_price = 30000
    timestamp = 1
    logging.info("Starting test_check_stop_loss_take_profit_no_trigger")

    # Act
    result = backtest_engine._check_stop_loss_take_profit(open_trades, current_price, timestamp)

    # Assert
    assert len(result) == 0

def test_update_trailing_stops_buy_direction(backtest_engine, trade_manager):
    # Arrange
    open_trades = [
        {"id": 1, "asset_name": "BTCUSD", "stop_loss": 29000, "direction": "buy", "trailing_stop": 5.0}
    ]
    asset_name = "BTCUSD"
    current_price = 35000
    backtest_engine.strategies[asset_name].trailing_stop_percent = 5.0
    trade_manager.modify_trade_parameters = Mock()
    logging.info("Starting test_update_trailing_stops_buy_direction")

    # Act
    backtest_engine._update_trailing_stops(open_trades, asset_name, current_price)

    # Assert
    logging.info("Asserting calls for modify_trade_parameters in buy direction")
    trade_manager.modify_trade_parameters.assert_called_once_with(trade_id=1, stop_loss=35000 * (1 - 0.05))

def test_update_trailing_stops_sell_direction(backtest_engine, trade_manager):
    # Arrange
    open_trades = [
        {"id": 2, "asset_name": "BTCUSD", "stop_loss": 38000, "direction": "sell", "trailing_stop": 5.0}
    ]
    asset_name = "BTCUSD"
    current_price = 34000
    backtest_engine.strategies[asset_name].trailing_stop_percent = 5.0
    trade_manager.modify_trade_parameters = Mock()
    logging.info("Starting test_update_trailing_stops_sell_direction")

    # Act
    backtest_engine._update_trailing_stops(open_trades, asset_name, current_price)

    # Assert
    logging.info("Asserting calls for modify_trade_parameters in sell direction")
    trade_manager.modify_trade_parameters.assert_called_once_with(trade_id=2, stop_loss=34000 * (1 + 0.05))

def test_update_trailing_stops_no_update(backtest_engine, trade_manager):
    # Arrange
    open_trades = [
        {"id": 3, "asset_name": "BTCUSD", "stop_loss": 32000, "direction": "buy", "trailing_stop": 5.0}
    ]
    asset_name = "BTCUSD"
    current_price = 31000
    backtest_engine.strategies[asset_name].trailing_stop_percent = 5.0
    trade_manager.modify_trade_parameters = Mock()
    logging.info("Starting test_update_trailing_stops_no_update")

    # Act
    backtest_engine._update_trailing_stops(open_trades, asset_name, current_price)

    # Assert
    logging.info("Asserting no calls for modify_trade_parameters when stop loss is not updated")
    trade_manager.modify_trade_parameters.assert_not_called()

def test_generate_strategy_signals(backtest_engine, trade_manager):
    # Arrange
    asset_name = "BTCUSD"
    row_data = pd.Series({"SMA_50": 30000, "SMA_200": 31000})
    current_price = 32000
    timestamp = 1
    backtest_engine.strategies[asset_name].generate_signal = Mock(return_value={"action": "buy", "amount": 1})
    logging.info("Starting test_generate_strategy_signals")

    # Act
    signals = backtest_engine._generate_strategy_signals(asset_name, row_data, current_price, timestamp)

    # Assert
    logging.info("Asserting generated strategy signals")
    assert signals is not None
    assert len(signals) == 1
    assert signals[0]["action"] == "buy"
    assert signals[0]["timestamp"] == timestamp

def test_generate_strategy_signals_no_signal(backtest_engine, trade_manager):
    # Arrange
    asset_name = "BTCUSD"
    row_data = pd.Series({"SMA_50": 30000, "SMA_200": 31000})
    current_price = 32000
    timestamp = 1
    backtest_engine.strategies[asset_name].generate_signal = Mock(return_value=None)
    logging.info("Starting test_generate_strategy_signals_no_signal")

    # Act
    signals = backtest_engine._generate_strategy_signals(asset_name, row_data, current_price, timestamp)

    # Assert
    logging.info("Asserting no generated strategy signals")
    assert signals is None or len(signals) == 0

def test_generate_strategy_signals_multiple_signals(backtest_engine, trade_manager):
    # Arrange
    asset_name = "BTCUSD"
    row_data = pd.Series({"SMA_50": 30000, "SMA_200": 31000})
    current_price = 32000
    timestamp = 1
    backtest_engine.strategies[asset_name].generate_signal = Mock(return_value=[{"action": "buy", "amount": 1}, {"action": "sell", "amount": 0.5}])
    logging.info("Starting test_generate_strategy_signals_multiple_signals")

    # Act
    signals = backtest_engine._generate_strategy_signals(asset_name, row_data, current_price, timestamp)

    # Assert
    logging.info("Asserting multiple generated strategy signals")
    assert signals is not None
    assert len(signals) == 2
    assert signals[0]["action"] == "buy"
    assert signals[1]["action"] == "sell"
    assert signals[0]["timestamp"] == timestamp
    assert signals[1]["timestamp"] == timestamp

def test_process_signals_successful(backtest_engine, signal_database, portfolio):
    # Arrange
    signals = [{"action": "buy", "asset_name": "BTCUSD", "amount": 1, "price": 30000, "timestamp": 1}]
    market_prices = {"BTCUSD": 30000}
    timestamp = 1
    backtest_engine.signal_database.add_signals = Mock()
    backtest_engine.portfolio.process_signals = Mock()
    logging.info("Starting test_process_signals_successful")

    # Act
    backtest_engine._process_signals(signals, market_prices, timestamp)

    # Assert
    logging.info("Asserting successful signal processing")
    backtest_engine.signal_database.add_signals.assert_called_once_with(signals)
    backtest_engine.portfolio.process_signals.assert_called_once_with(signals=signals, market_prices=market_prices, timestamp=timestamp)

def test_process_signals_with_exception(backtest_engine, signal_database, portfolio):
    # Arrange
    signals = [{"action": "buy", "asset_name": "BTCUSD", "amount": 1, "price": 30000, "timestamp": 1}]
    market_prices = {"BTCUSD": 30000}
    timestamp = 1
    backtest_engine.signal_database.add_signals = Mock()
    backtest_engine.portfolio.process_signals = Mock(side_effect=Exception("Test exception"))
    logging.info("Starting test_process_signals_with_exception")

    # Act
    backtest_engine._process_signals(signals, market_prices, timestamp)

    # Assert
    logging.info("Asserting signal processing with exception handling")
    backtest_engine.signal_database.add_signals.assert_called_once_with(signals)
    backtest_engine.portfolio.process_signals.assert_called_once_with(signals=signals, market_prices=market_prices, timestamp=timestamp)
    backtest_engine.logger.error.assert_called()

def test_process_signals_empty_signal_list(backtest_engine, signal_database, portfolio):
    # Arrange
    signals = []
    market_prices = {"BTCUSD": 30000}
    timestamp = 1
    backtest_engine.signal_database.add_signals = Mock()
    backtest_engine.portfolio.process_signals = Mock()
    logging.info("Starting test_process_signals_empty_signal_list")

    # Act
    backtest_engine._process_signals(signals, market_prices, timestamp)

    # Assert
    logging.info("Asserting no processing for empty signal list")
    backtest_engine.signal_database.add_signals.assert_not_called()
    backtest_engine.portfolio.process_signals.assert_not_called()

def test_update_trailing_stops_for_long_position(backtest_engine, trade_manager):
    # Arrange
    asset_name = "BTCUSD"
    open_trades = [
        {"id": 1, "asset_name": asset_name, "direction": "buy", "stop_loss": 29000, "trailing_stop": None}
    ]
    current_price = 31000
    backtest_engine.strategies[asset_name].trailing_stop_percent = 5.0  # Trailing stop is 5%
    backtest_engine.trade_manager.modify_trade_parameters = Mock()  # Mock the modify_trade_parameters on backtest_engine
    logging.info("Starting test_update_trailing_stops_for_long_position")

    # Act
    backtest_engine._update_trailing_stops(open_trades, asset_name, current_price)

    # Assert
    logging.info("Asserting trailing stop update for long position")
    new_stop_loss = round(current_price * (1 - 0.05), 2)  # Ensure rounding to match implementation
    if backtest_engine.trade_manager.modify_trade_parameters.call_count == 0:
        logging.error("modify_trade_parameters was not called. Expected stop loss: %f", new_stop_loss)
    else:
        logging.info("modify_trade_parameters was called with stop loss: %f", new_stop_loss)
    backtest_engine.trade_manager.modify_trade_parameters.assert_called_once_with(trade_id=1, stop_loss=new_stop_loss)

def test_update_trailing_stops_for_short_position(backtest_engine, trade_manager):
    # Arrange
    asset_name = "BTCUSD"
    open_trades = [
        {"id": 1, "asset_name": asset_name, "direction": "sell", "stop_loss": 33000, "trailing_stop": None}
    ]
    current_price = 31000
    backtest_engine.strategies[asset_name].trailing_stop_percent = 5.0  # Trailing stop is 5%
    backtest_engine.trade_manager.modify_trade_parameters = Mock()  # Mock the modify_trade_parameters on backtest_engine
    logging.info("Starting test_update_trailing_stops_for_short_position")

    # Act
    backtest_engine._update_trailing_stops(open_trades, asset_name, current_price)

    # Assert
    logging.info("Asserting trailing stop update for short position")
    new_stop_loss = round(current_price * (1 + 0.05), 2)  # Ensure rounding to match implementation
    if backtest_engine.trade_manager.modify_trade_parameters.call_count == 0:
        logging.error("modify_trade_parameters was not called. Expected stop loss: %f", new_stop_loss)
    else:
        logging.info("modify_trade_parameters was called with stop loss: %f", new_stop_loss)
    backtest_engine.trade_manager.modify_trade_parameters.assert_called_once_with(trade_id=1, stop_loss=new_stop_loss)

def test_update_trailing_stops_no_update_needed(backtest_engine, trade_manager):
    # Arrange
    asset_name = "BTCUSD"
    open_trades = [
        {"id": 1, "asset_name": asset_name, "direction": "buy", "stop_loss": 29500, "trailing_stop": None}
    ]
    current_price = 31000
    backtest_engine.strategies[asset_name].trailing_stop_percent = 5.0  # Trailing stop is 5%
    backtest_engine.trade_manager.modify_trade_parameters = Mock()  # Mock the modify_trade_parameters on backtest_engine
    logging.info("Starting test_update_trailing_stops_no_update_needed")

    # Act
    backtest_engine._update_trailing_stops(open_trades, asset_name, current_price)

    # Assert
    logging.info("Asserting no trailing stop update needed")
    if backtest_engine.trade_manager.modify_trade_parameters.call_count > 0:
        logging.error("modify_trade_parameters was called unexpectedly.")
    else:
        logging.info("modify_trade_parameters was not called as expected.")
    backtest_engine.trade_manager.modify_trade_parameters.assert_not_called()

def test_generate_strategy_signals_no_signal(backtest_engine, trade_manager):
    # Arrange
    asset_name = "BTCUSD"
    row_data = pd.Series({"SMA_50": 30000, "SMA_200": 31000})
    current_price = 32000
    timestamp = 1
    backtest_engine.strategies[asset_name].generate_signal = Mock(return_value=[])
    logging.info("Starting test_generate_strategy_signals_no_signal")

    # Act
    signals = backtest_engine._generate_strategy_signals(asset_name, row_data, current_price, timestamp)

    # Assert
    logging.info("Asserting no generated strategy signals")
    assert signals == []  # Expecting no signals to be generated

def test_generate_strategy_signals_buy_signal(backtest_engine, trade_manager):
    # Arrange
    asset_name = "BTCUSD"
    row_data = pd.Series({"SMA_50": 30000, "SMA_200": 31000})
    current_price = 29000
    timestamp = 1
    signal = {"action": "buy", "asset_name": asset_name, "amount": 1.0}
    backtest_engine.strategies[asset_name].generate_signal = Mock(return_value=[signal])
    logging.info("Starting test_generate_strategy_signals_buy_signal")

    # Act
    signals = backtest_engine._generate_strategy_signals(asset_name, row_data, current_price, timestamp)

    # Assert
    logging.info("Asserting buy signal is generated")
    assert signals == [signal]  # Expecting a buy signal to be generated

def test_generate_strategy_signals_multiple_signals(backtest_engine, trade_manager):
    # Arrange
    asset_name = "BTCUSD"
    row_data = pd.Series({"SMA_50": 30000, "SMA_200": 31000})
    current_price = 29000
    timestamp = 1
    signals = [
        {"action": "buy", "asset_name": asset_name, "amount": 1.0},
        {"action": "sell", "asset_name": asset_name, "amount": 0.5}
    ]
    backtest_engine.strategies[asset_name].generate_signal = Mock(return_value=signals)
    logging.info("Starting test_generate_strategy_signals_multiple_signals")

    # Act
    generated_signals = backtest_engine._generate_strategy_signals(asset_name, row_data, current_price, timestamp)

    # Assert
    logging.info("Asserting multiple signals are generated")
    assert generated_signals == signals  # Expecting multiple signals to be generated

def test_close_all_open_trades(backtest_engine, trade_manager):
    # Arrange
    final_timestamp = 3
    backtest_engine.market_data = {
        "BTCUSD": pd.DataFrame({"close": [30000, 31000, 32000]}, index=[1, 2, 3]),
        "ETHUSD": pd.DataFrame({"close": [1500, 1600, 1700]}, index=[1, 2, 3])
    }
    open_trades = [
        {"id": 1, "asset_name": "BTCUSD", "direction": "buy", "stop_loss": 29000, "take_profit": 35000, "asset_amount": 1.0},
        {"id": 2, "asset_name": "ETHUSD", "direction": "buy", "stop_loss": 1400, "take_profit": 1800, "asset_amount": 2.0}
    ]
    trade_manager.get_trade.return_value = open_trades
    backtest_engine._generate_close_signal = Mock(side_effect=lambda trade, current_price, timestamp, reason: {
        'action': 'close',
        'trade_id': trade['id'],
        'asset_name': trade['asset_name'],
        'amount': trade['asset_amount'],
        'price': current_price,
        'timestamp': timestamp,
        'reason': reason,
    })
    backtest_engine._process_signals = Mock()
    logging.info("Starting test_close_all_open_trades")

    # Act
    backtest_engine._close_all_open_trades(final_timestamp)

    # Assert
    logging.info("Asserting close signals are generated and processed")
    expected_close_signals = [
        {
            'action': 'close',
            'trade_id': 1,
            'asset_name': 'BTCUSD',
            'amount': 1.0,
            'price': 32000,
            'timestamp': 3,
            'reason': 'end_of_backtest'
        },
        {
            'action': 'close',
            'trade_id': 2,
            'asset_name': 'ETHUSD',
            'amount': 2.0,
            'price': 1700,
            'timestamp': 3,
            'reason': 'end_of_backtest'
        }
    ]
    backtest_engine._process_signals.assert_called_once_with(expected_close_signals, {
        "BTCUSD": 32000,
        "ETHUSD": 1700
    }, final_timestamp)

def test_close_all_open_trades_no_open_trades(backtest_engine, trade_manager):
    # Arrange
    final_timestamp = 3
    backtest_engine.market_data = {
        "BTCUSD": pd.DataFrame({"close": [30000, 31000, 32000]}, index=[1, 2, 3]),
        "ETHUSD": pd.DataFrame({"close": [1500, 1600, 1700]}, index=[1, 2, 3])
    }
    trade_manager.get_trade.return_value = []  # No open trades
    backtest_engine._generate_close_signal = Mock()
    backtest_engine._process_signals = Mock()
    logging.info("Starting test_close_all_open_trades_no_open_trades")

    # Act
    backtest_engine._close_all_open_trades(final_timestamp)

    # Assert
    logging.info("Asserting no close signals are generated when there are no open trades")
    backtest_engine._generate_close_signal.assert_not_called()
    backtest_engine._process_signals.assert_not_called()

def test_close_all_open_trades_partial_close(backtest_engine, trade_manager):
    # Arrange
    final_timestamp = 3
    backtest_engine.market_data = {
        "BTCUSD": pd.DataFrame({"close": [30000, 31000, 32000]}, index=[1, 2, 3]),
        "ETHUSD": pd.DataFrame({"close": [1500, 1600, 1700]}, index=[1, 2, 3])
    }
    open_trades = [
        {"id": 1, "asset_name": "BTCUSD", "direction": "buy", "stop_loss": 29000, "take_profit": 35000, "asset_amount": 1.0}
    ]
    trade_manager.get_trade.return_value = open_trades
    backtest_engine._generate_close_signal = Mock(side_effect=lambda trade, current_price, timestamp, reason: {
        'action': 'close',
        'trade_id': trade['id'],
        'asset_name': trade['asset_name'],
        'amount': trade['asset_amount'],
        'price': current_price,
        'timestamp': timestamp,
        'reason': reason,
    })
    backtest_engine._process_signals = Mock()
    logging.info("Starting test_close_all_open_trades_partial_close")

    # Act
    backtest_engine._close_all_open_trades(final_timestamp)

    # Assert
    logging.info("Asserting close signals are generated and processed for partial close")
    expected_close_signals = [
        {
            'action': 'close',
            'trade_id': 1,
            'asset_name': 'BTCUSD',
            'amount': 1.0,
            'price': 32000,
            'timestamp': 3,
            'reason': 'end_of_backtest'
        }
    ]
    backtest_engine._process_signals.assert_called_once_with(expected_close_signals, {
        "BTCUSD": 32000,
        "ETHUSD": 1700
    }, final_timestamp)

def test_close_all_open_trades_no_market_data(backtest_engine, trade_manager):
    # Arrange
    final_timestamp = 3
    backtest_engine.market_data = {}  # No market data available
    open_trades = [
        {"id": 1, "asset_name": "BTCUSD", "direction": "buy", "stop_loss": 29000, "take_profit": 35000, "asset_amount": 1.0}
    ]
    trade_manager.get_trade.return_value = open_trades
    backtest_engine._generate_close_signal = Mock()
    backtest_engine._process_signals = Mock()
    logging.info("Starting test_close_all_open_trades_no_market_data")

    # Act
    with pytest.raises(KeyError, match="'BTCUSD'"):
        backtest_engine._close_all_open_trades(final_timestamp)

    # Assert
    logging.info("Asserting KeyError is raised when there is no market data")
    backtest_engine._generate_close_signal.assert_not_called()
    backtest_engine._process_signals.assert_not_called()
