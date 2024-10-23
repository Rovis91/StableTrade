import os
import sys 
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics import MetricsModule

@pytest.fixture
def sample_market_data():
    dates = pd.date_range(start='2023-08-31 22:59:00', end='2023-09-01 00:30:00', freq='min')
    data = {
        'open': np.random.uniform(1.0001, 1.0011, len(dates)),
        'close': np.random.uniform(1.0001, 1.0011, len(dates)),
        'high': np.random.uniform(1.0001, 1.0011, len(dates)),
        'low': np.random.uniform(1.0001, 1.0011, len(dates)),
        'volume': np.random.uniform(3.0, 16.0, len(dates))
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_signals():
    return pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2023-11-01 12:34:00'),
            pd.Timestamp('2023-11-01 12:37:00'),
            pd.Timestamp('2023-11-01 12:42:00'),
            pd.Timestamp('2023-11-01 12:46:00'),
            pd.Timestamp('2023-11-01 12:47:00')
        ],
        'asset_name': ['EUTEUR'] * 5,
        'action': ['buy', 'close', 'buy', 'buy', 'close'],
        'amount': [0.1, 103.225806, 0.1, 0.1, 82.560852],
        'price': [0.96875, 0.9844, 0.96874, 0.96973, 0.96873],
        'stop_loss': [0.920312, np.nan, 0.920303, 0.921243, np.nan],
        'take_profit': [1.065625, np.nan, 1.065614, 1.066703, np.nan],
        'trailing_stop': [0.02, np.nan, 0.02, 0.02, np.nan],
        'signal_id': range(1, 6),
        'trade_id': [1.0, 1.0, 2.0, 3.0, 2.0],
        'status': ['executed'] * 5,
        'reason': [np.nan, 'reason', np.nan, np.nan, 'reason']
    })

@pytest.fixture
def sample_trades():
    return pd.DataFrame({
        'id': range(1, 6),
        'asset_name': ['EUTEUR'] * 5,
        'base_currency': ['EUR'] * 5,
        'asset_amount': [103.225806, 82.560852, 74.220661, 58.595694, 42.438639],
        'base_amount': [100.0, 79.98, 71.974002, 56.763406, 38.194775],
        'entry_price': [0.96875, 0.96874, 0.96973, 0.96873, 0.90000],
        'entry_timestamp': [
            pd.Timestamp('2023-11-01 12:34:00'),
            pd.Timestamp('2023-11-01 12:42:00'),
            pd.Timestamp('2023-11-01 12:46:00'),
            pd.Timestamp('2023-11-01 12:47:00'),
            pd.Timestamp('2023-11-01 12:49:00')
        ],
        'entry_fee': [0.1, 0.07998, 0.071974, 0.056763, 0.038195],
        'exit_price': [0.9844, 0.96873, 0.9, 0.9, 0.999],
        'exit_timestamp': [
            pd.Timestamp('2023-11-01 12:37:00'),
            pd.Timestamp('2023-11-01 12:47:00'),
            pd.Timestamp('2023-11-01 12:49:00'),
            pd.Timestamp('2023-11-01 12:49:00'),
            pd.Timestamp('2023-11-01 12:51:00')
        ],
        'exit_fee': [0.1, 0.07998, 0.071974, 0.056763, 0.038195],
        'status': ['closed'] * 5,
        'direction': ['buy'] * 5,
        'stop_loss': [0.999570, 0.969536, 0.968536, 0.920293, 0.998800],
        'take_profit': [1.065625, 1.065614, 1.066703, 1.065603, 0.990000],
        'trailing_stop': [0.02] * 5,
        'entry_reason': ['buy_signal'] * 5,
        'exit_reason': ['reason'] * 5,
        'profit_loss': [1.415484, -0.160786, -5.319355, -4.140809, 4.125036]
    })

@pytest.fixture
def sample_portfolio_history():
    dates = pd.date_range(start='2023-08-31 22:59:00', end='2023-09-01 00:30:00', freq='min')
    holdings = [{'EUR': 1000} for _ in range(len(dates))]
    return pd.DataFrame({
        'timestamp': dates,
        'holdings': holdings
    })

def test_calculate_monthly_returns(sample_market_data, sample_portfolio_history):
    metrics_module = MetricsModule('dummy_path.csv', base_currency='EUR')
    
    monthly_returns = metrics_module.calculate_monthly_returns(sample_portfolio_history, sample_market_data)
    
    assert isinstance(monthly_returns, pd.Series)
    # Our data spans from Aug 31 to Sept 1, so we expect 1 month of returns
    assert len(monthly_returns) == 1  
    assert monthly_returns.index.freq == 'ME'
    assert not monthly_returns.isnull().any()
    assert (monthly_returns >= -1).all() and (monthly_returns <= 1).all()

def test_calculate_monthly_returns_missing_data(sample_market_data, sample_portfolio_history):
    metrics_module = MetricsModule('dummy_path.csv', base_currency='EUR')
    
    # Remove some market data to test handling of missing data
    sample_market_data = sample_market_data.iloc[::2]  # Take every other row
    
    monthly_returns = metrics_module.calculate_monthly_returns(sample_portfolio_history, sample_market_data)
    
    assert isinstance(monthly_returns, pd.Series)
    assert len(monthly_returns) == 1  # Still expect 1 month of returns
    assert not monthly_returns.isnull().any()

def test_calculate_monthly_returns_error_handling():
    metrics_module = MetricsModule('dummy_path.csv', base_currency='EUR')
    
    with pytest.raises(KeyError):
        metrics_module.calculate_monthly_returns(
            pd.DataFrame({'wrong_column': [1, 2, 3]}),
            pd.DataFrame({'EUTEUR_close': [1, 2, 3]})
        )

def test_calculate_sharpe_ratio():
    metrics_module = MetricsModule('dummy_path.csv', base_currency='EUR', risk_free_rate=0.01)
    
    # Create a series of monthly returns
    monthly_returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.01, -0.01, 0.02, 0.01, 0.03, 0.01])
    
    sharpe_ratio = metrics_module.calculate_sharpe_ratio(monthly_returns)
    
    assert isinstance(sharpe_ratio, float)
    assert not np.isnan(sharpe_ratio)
    assert sharpe_ratio > -10 and sharpe_ratio < 10  # Reasonable range for Sharpe ratio

def test_calculate_sharpe_ratio_zero_volatility():
    metrics_module = MetricsModule('dummy_path.csv', base_currency='EUR', risk_free_rate=0.01)
    
    # Test case 1: Constant returns higher than risk-free rate
    monthly_returns = pd.Series([0.02] * 12)
    sharpe_ratio = metrics_module.calculate_sharpe_ratio(monthly_returns)
    assert np.isinf(sharpe_ratio) and sharpe_ratio > 0

    # Test case 2: Constant returns lower than risk-free rate
    monthly_returns = pd.Series([0.0005] * 12)
    sharpe_ratio = metrics_module.calculate_sharpe_ratio(monthly_returns)
    assert np.isinf(sharpe_ratio) and sharpe_ratio < 0

    # Test case 3: Constant returns equal to risk-free rate
    monthly_returns = pd.Series([0.01/12] * 12)  # Monthly risk-free rate
    sharpe_ratio = metrics_module.calculate_sharpe_ratio(monthly_returns)
    assert np.isclose(sharpe_ratio, 0, atol=1e-8)

def test_calculate_sharpe_ratio_insufficient_data():
    metrics_module = MetricsModule('dummy_path.csv', base_currency='EUR', risk_free_rate=0.01)
    
    # Create a series with only one month of returns
    monthly_returns = pd.Series([0.01])
    
    with pytest.raises(ValueError, match="At least two months of returns are required"):
        metrics_module.calculate_sharpe_ratio(monthly_returns)

def test_calculate_sharpe_ratio_negative():
    metrics_module = MetricsModule('dummy_path.csv', base_currency='EUR', risk_free_rate=0.05)
    
    # Create a series of monthly returns lower than the risk-free rate
    monthly_returns = pd.Series([0.001, 0.002, 0.003, 0.001, 0.002, 0.003] * 2)
    
    sharpe_ratio = metrics_module.calculate_sharpe_ratio(monthly_returns)
    
    assert sharpe_ratio < 0

def test_calculate_max_drawdown():
    metrics_module = MetricsModule('dummy_path.csv', base_currency='EUR')
    
    # Create a sample portfolio history
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    portfolio_values = [100, 110, 105, 95, 100, 90, 95, 100, 110, 105]  # Example portfolio values
    holdings = [{'EUR': value} for value in portfolio_values]
    portfolio_history = pd.DataFrame({'timestamp': dates[:len(portfolio_values)], 'holdings': holdings})

    max_drawdown = metrics_module.calculate_max_drawdown(portfolio_history)
    
    assert isinstance(max_drawdown, float)
    assert 0 <= max_drawdown <= 1
    assert np.isclose(max_drawdown, 0.1818, atol=1e-4)  # Expected max drawdown is about 18.18%

def test_calculate_max_drawdown_no_drawdown():
    metrics_module = MetricsModule('dummy_path.csv', base_currency='EUR')
    
    # Create a sample portfolio history with only increasing values
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    portfolio_values = [100, 110, 120, 130, 140]
    holdings = [{'EUR': value} for value in portfolio_values]
    portfolio_history = pd.DataFrame({'timestamp': dates[:len(portfolio_values)], 'holdings': holdings})

    max_drawdown = metrics_module.calculate_max_drawdown(portfolio_history)
    
    assert max_drawdown == 0.0

def test_calculate_win_rate_normal(sample_trades):
    metrics = MetricsModule("path/to/market/data", "USD")
    win_rate = metrics.calculate_win_rate(sample_trades)
    # From our sample data: 2 winning trades out of 5
    assert win_rate == 40.0  

def test_calculate_win_rate_all_winning():
    all_winning = pd.DataFrame({
        'id': range(1, 6),
        'status': ['closed'] * 5,
        'profit_loss': [100, 200, 150, 300, 250]
    })
    metrics = MetricsModule("path/to/market/data", "USD")
    win_rate = metrics.calculate_win_rate(all_winning)
    assert win_rate == 100.0

def test_calculate_win_rate_all_losing():
    all_losing = pd.DataFrame({
        'id': range(1, 6),
        'status': ['closed'] * 5,
        'profit_loss': [-100, -200, -150, -300, -250]
    })
    metrics = MetricsModule("path/to/market/data", "USD")
    win_rate = metrics.calculate_win_rate(all_losing)
    assert win_rate == 0.0

def test_calculate_win_rate_empty_trades():
    empty_trades = pd.DataFrame(columns=['id', 'status', 'profit_loss'])
    metrics = MetricsModule("path/to/market/data", "USD")
    win_rate = metrics.calculate_win_rate(empty_trades)
    assert win_rate == 0.0

def test_calculate_win_rate_no_closed_trades():
    open_trades = pd.DataFrame({
        'id': range(1, 6),
        'status': ['open'] * 5,
        'profit_loss': [100, -50, 75, -25, 50]
    })
    metrics = MetricsModule("path/to/market/data", "USD")
    win_rate = metrics.calculate_win_rate(open_trades)
    assert win_rate == 0.0

def test_calculate_win_rate_mixed_status():
    mixed_trades = pd.DataFrame({
        'id': range(1, 11),
        'status': ['closed', 'open', 'closed', 'closed', 'open', 'closed', 'open', 'closed', 'closed', 'closed'],
        'profit_loss': [100, -50, 75, 200, -25, 50, -100, 150, -75, 80]
    })
    metrics = MetricsModule("path/to/market/data", "USD")
    win_rate = metrics.calculate_win_rate(mixed_trades)
    assert win_rate == pytest.approx(85.71, 0.01)  # 6 winning trades out of 7 closed trades

def test_calculate_profit_factor_normal(sample_trades):
    metrics = MetricsModule("path/to/market/data", "USD")
    profit_factor = metrics.calculate_profit_factor(sample_trades)
    # Winning trades: 1.415484 + 4.125036 = 5.54052
    # Losing trades: 0.160786 + 5.319355 + 4.140809 = 9.62095
    expected_profit_factor = 5.54052 / 9.62095
    assert profit_factor == pytest.approx(expected_profit_factor, rel=1e-6)

def test_calculate_profit_factor_all_winning():
    all_winning = pd.DataFrame({
        'id': range(1, 6),
        'status': ['closed'] * 5,
        'profit_loss': [100, 200, 150, 300, 250]
    })
    metrics = MetricsModule("path/to/market/data", "USD")
    profit_factor = metrics.calculate_profit_factor(all_winning)
    assert profit_factor == 0.0  # No losing trades, profit factor is undefined

def test_calculate_profit_factor_all_losing():
    all_losing = pd.DataFrame({
        'id': range(1, 6),
        'status': ['closed'] * 5,
        'profit_loss': [-100, -200, -150, -300, -250]
    })
    metrics = MetricsModule("path/to/market/data", "USD")
    profit_factor = metrics.calculate_profit_factor(all_losing)
    assert profit_factor == 0.0  # No winning trades, profit factor is 0

def test_calculate_profit_factor_empty_trades():
    empty_trades = pd.DataFrame(columns=['id', 'status', 'profit_loss'])
    metrics = MetricsModule("path/to/market/data", "USD")
    profit_factor = metrics.calculate_profit_factor(empty_trades)
    assert profit_factor == 0.0

def test_calculate_profit_factor_no_closed_trades():
    open_trades = pd.DataFrame({
        'id': range(1, 6),
        'status': ['open'] * 5,
        'profit_loss': [100, -50, 75, -25, 50]
    })
    metrics = MetricsModule("path/to/market/data", "USD")
    profit_factor = metrics.calculate_profit_factor(open_trades)
    assert profit_factor == 0.0

def test_calculate_profit_factor_mixed_status():
    mixed_trades = pd.DataFrame({
        'id': range(1, 11),
        'status': ['closed', 'open', 'closed', 'closed', 'open', 'closed', 'open', 'closed', 'closed', 'closed'],
        'profit_loss': [100, -50, 75, 200, -25, 50, -100, 150, -75, 80]
    })
    metrics = MetricsModule("path/to/market/data", "USD")
    profit_factor = metrics.calculate_profit_factor(mixed_trades)
    expected_profit_factor = (100 + 75 + 200 + 50 + 150 + 80) / 75
    assert profit_factor == pytest.approx(expected_profit_factor, rel=1e-6)

def test_calculate_profit_factor_zero_profit():
    zero_profit_trades = pd.DataFrame({
        'id': range(1, 6),
        'status': ['closed'] * 5,
        'profit_loss': [0, 0, 0, 0, 0]
    })
    metrics = MetricsModule("path/to/market/data", "USD")
    profit_factor = metrics.calculate_profit_factor(zero_profit_trades)
    assert profit_factor == 0.0

def test_calculate_average_trade_duration_normal(sample_trades):
    metrics = MetricsModule("path/to/market/data", "USD")
    avg_duration = metrics.calculate_average_trade_duration(sample_trades)
    # Average duration from our sample trades (in seconds)
    durations = [
        (pd.Timestamp('2023-11-01 12:37:00') - pd.Timestamp('2023-11-01 12:34:00')).total_seconds(),
        (pd.Timestamp('2023-11-01 12:47:00') - pd.Timestamp('2023-11-01 12:42:00')).total_seconds(),
        (pd.Timestamp('2023-11-01 12:49:00') - pd.Timestamp('2023-11-01 12:46:00')).total_seconds(),
        (pd.Timestamp('2023-11-01 12:49:00') - pd.Timestamp('2023-11-01 12:47:00')).total_seconds(),
        (pd.Timestamp('2023-11-01 12:51:00') - pd.Timestamp('2023-11-01 12:49:00')).total_seconds()
    ]
    expected_average = sum(durations) / len(durations)
    assert avg_duration == pytest.approx(expected_average, rel=1e-6)
    
def test_calculate_average_trade_duration_empty_trades():
    empty_trades = pd.DataFrame(columns=['id', 'status', 'profit_loss', 'entry_timestamp', 'exit_timestamp'])
    metrics = MetricsModule("path/to/market/data", "USD")
    avg_duration = metrics.calculate_average_trade_duration(empty_trades)
    assert avg_duration == 0.0

def test_calculate_average_trade_duration_no_closed_trades(sample_trades):
    open_trades = sample_trades.copy()
    open_trades['status'] = 'open'
    open_trades['exit_timestamp'] = pd.NaT
    metrics = MetricsModule("path/to/market/data", "USD")
    avg_duration = metrics.calculate_average_trade_duration(open_trades)
    assert avg_duration == 0.0

def test_calculate_average_trade_duration_mixed_status(sample_trades):
    mixed_trades = sample_trades.copy()
    mixed_trades.loc[1:3, 'status'] = 'open'
    mixed_trades.loc[1:3, 'exit_timestamp'] = pd.NaT
    metrics = MetricsModule("path/to/market/data", "USD")
    avg_duration = metrics.calculate_average_trade_duration(mixed_trades)
    
    # Calculate durations for remaining closed trades (first and last trade)
    durations = [
        (pd.Timestamp('2023-11-01 12:37:00') - pd.Timestamp('2023-11-01 12:34:00')).total_seconds(),
        (pd.Timestamp('2023-11-01 12:51:00') - pd.Timestamp('2023-11-01 12:49:00')).total_seconds()
    ]
    expected_average = sum(durations) / len(durations)
    assert avg_duration == pytest.approx(expected_average, rel=1e-6)
    
    # Ensure the original DataFrame was not modified
    assert (mixed_trades.loc[1:3, 'status'] == 'open').all()
    assert mixed_trades.loc[1:3, 'exit_timestamp'].isna().all()

def test_calculate_average_trade_duration_single_trade(sample_trades):
    single_trade = sample_trades.iloc[[0]].copy()
    metrics = MetricsModule("path/to/market/data", "USD")
    avg_duration = metrics.calculate_average_trade_duration(single_trade)
    # Duration of first trade
    expected_duration = (pd.Timestamp('2023-11-01 12:37:00') - 
                        pd.Timestamp('2023-11-01 12:34:00')).total_seconds()
    assert avg_duration == pytest.approx(expected_duration, rel=1e-6)

def test_calculate_total_fees_normal(sample_trades):
    metrics = MetricsModule("path/to/market/data", "USD")
    total_fees = metrics.calculate_total_fees(sample_trades)
    expected_total_fees = sum(sample_trades['entry_fee']) + sum(sample_trades['exit_fee'])
    assert total_fees == pytest.approx(expected_total_fees, rel=1e-6)

def test_calculate_total_fees_empty_trades():
    empty_trades = pd.DataFrame(columns=['id', 'status', 'profit_loss', 'entry_timestamp', 'exit_timestamp', 'entry_fee', 'exit_fee'])
    metrics = MetricsModule("path/to/market/data", "USD")
    total_fees = metrics.calculate_total_fees(empty_trades)
    assert total_fees == 0.0

def test_calculate_total_fees_zero_fees(sample_trades):
    zero_fee_trades = sample_trades.copy()
    zero_fee_trades['entry_fee'] = 0.0
    zero_fee_trades['exit_fee'] = 0.0
    metrics = MetricsModule("path/to/market/data", "USD")
    total_fees = metrics.calculate_total_fees(zero_fee_trades)
    assert total_fees == 0.0

def test_calculate_total_fees_missing_fee_column(sample_trades):
    incomplete_trades = sample_trades.drop('exit_fee', axis=1)
    metrics = MetricsModule("path/to/market/data", "USD")
    with pytest.raises(KeyError):
        metrics.calculate_total_fees(incomplete_trades)

def test_calculate_total_fees_large_numbers(sample_trades):
    large_fee_trades = sample_trades.copy()
    large_fee_trades['entry_fee'] = large_fee_trades['entry_fee'] * 1e6
    large_fee_trades['exit_fee'] = large_fee_trades['exit_fee'] * 1e6
    metrics = MetricsModule("path/to/market/data", "USD")
    total_fees = metrics.calculate_total_fees(large_fee_trades)
    expected_total_fees = sum(large_fee_trades['entry_fee']) + sum(large_fee_trades['exit_fee'])
    assert total_fees == pytest.approx(expected_total_fees, rel=1e-6)

def test_calculate_total_return_normal(sample_portfolio_history):
    metrics = MetricsModule("path/to/market/data", "USD")
    total_return = metrics.calculate_total_return(sample_portfolio_history)
    
    # From our sample data: initial value is 1000, final value is 1000
    expected_return = 0.0  # (1000 - 1000) / 1000 * 100
    assert total_return == pytest.approx(expected_return, rel=1e-6)

def test_calculate_total_return_empty_portfolio():
    metrics = MetricsModule("path/to/market/data", "USD")
    empty_portfolio = pd.DataFrame(columns=['timestamp', 'holdings'])
    total_return = metrics.calculate_total_return(empty_portfolio)
    assert total_return == 0.0

def test_calculate_total_return_single_entry(sample_portfolio_history):
    metrics = MetricsModule("path/to/market/data", "USD")
    single_entry = sample_portfolio_history.iloc[[0]].copy()
    total_return = metrics.calculate_total_return(single_entry)
    assert total_return == 0.0

def test_calculate_total_return_zero_initial_value():
    metrics = MetricsModule("path/to/market/data", "USD")
    portfolio = pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2023-08-31 22:59:00'),
            pd.Timestamp('2023-09-01 00:30:00')
        ],
        'holdings': [
            {'EUR': 0},
            {'EUR': 1000}
        ]
    })
    total_return = metrics.calculate_total_return(portfolio)
    assert total_return == 0.0

def test_calculate_total_return_positive(sample_portfolio_history):
    metrics = MetricsModule("path/to/market/data", "USD")
    portfolio = pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2023-08-31 22:59:00'),
            pd.Timestamp('2023-09-01 00:30:00')
        ],
        'holdings': [
            {'EUR': 1000},
            {'EUR': 1500}
        ]
    })
    total_return = metrics.calculate_total_return(portfolio)
    expected_return = 50.0  # (1500 - 1000) / 1000 * 100
    assert total_return == pytest.approx(expected_return, rel=1e-6)

def test_calculate_total_return_negative(sample_portfolio_history):
    metrics = MetricsModule("path/to/market/data", "USD")
    portfolio = pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2023-08-31 22:59:00'),
            pd.Timestamp('2023-09-01 00:30:00')
        ],
        'holdings': [
            {'EUR': 1000},
            {'EUR': 800}
        ]
    })
    total_return = metrics.calculate_total_return(portfolio)
    expected_return = -20.0  # (800 - 1000) / 1000 * 100
    assert total_return == pytest.approx(expected_return, rel=1e-6)

def test_calculate_total_return_large_numbers(sample_portfolio_history):
    metrics = MetricsModule("path/to/market/data", "USD")
    portfolio = pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2023-08-31 22:59:00'),
            pd.Timestamp('2023-09-01 00:30:00')
        ],
        'holdings': [
            {'EUR': 1_000_000},
            {'EUR': 1_500_000}
        ]
    })
    total_return = metrics.calculate_total_return(portfolio)
    expected_return = 50.0  # (1.5e6 - 1e6) / 1e6 * 100
    assert total_return == pytest.approx(expected_return, rel=1e-6)

def test_calculate_total_return_multiple_assets():
    metrics = MetricsModule("path/to/market/data", "USD")
    portfolio = pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2023-08-31 22:59:00'),
            pd.Timestamp('2023-09-01 00:30:00')
        ],
        'holdings': [
            {'EUR': 1000, 'EUTEUR': 500},
            {'EUR': 1200, 'EUTEUR': 800}
        ]
    })
    total_return = metrics.calculate_total_return(portfolio)
    # Initial value = 1500, Final value = 2000
    expected_return = 33.33333  # (2000 - 1500) / 1500 * 100
    assert total_return == pytest.approx(expected_return, rel=1e-6)