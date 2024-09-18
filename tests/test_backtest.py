import pytest
import pandas as pd
from src.backtest_engine import backtest

def test_backtest():
    # Simulate test data
    df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'signal': [0, 1, 0, -1, 0]
    })

    final_balance, trades, trade_log = backtest(df, initial_balance=1000)

    assert final_balance > 1000  # Basic test: final balance should be greater than initial
    assert trades == 2           # Test: ensure we have 2 trades
