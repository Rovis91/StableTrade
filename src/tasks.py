from celery import Celery
from src.backtest import backtest
from src.metrics import (
    calculate_sharpe_ratio,
    calculate_cumulative_return,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_total_trades
)
import pandas as pd
from src.strategy import sma_strategy, Strategy
import numpy as np

# Initialize Celery with Redis as broker and result backend
celery_app = Celery('tasks')
celery_app.config_from_object('celeryconfig')

@celery_app.task
def run_backtest_task(data, initial_balance, commission, params):
    """
    Celery task to run backtest in a distributed manner.
    :param data: Historical data (as dictionary)
    :param initial_balance: Starting capital
    :param commission: Commission per trade
    :param params: Strategy parameters (dict)
    :return: Results of the backtest (final balance, trade count, etc.)
    """
    # Rebuild DataFrame from the dictionary
    df = pd.DataFrame.from_dict(data)
    
    # Apply strategy
    strategy = Strategy(sma_strategy, params)
    df = strategy.apply(df)
    
    # Run backtest
    final_balance, trades, trade_log = backtest(df, initial_balance, commission)

    # Create a balance series for calculating max drawdown and Sharpe ratio
    balance_series = [initial_balance]  # This would be built in the backtest to track balances over time

    # Calculate performance metrics
    cumulative_return = calculate_cumulative_return(initial_balance, final_balance)
    max_drawdown = calculate_max_drawdown(balance_series)
    sharpe_ratio = calculate_sharpe_ratio(np.array(balance_series) / initial_balance - 1)
    profit_factor = calculate_profit_factor(trade_log)
    total_trades = calculate_total_trades(trade_log)

    # Return all metrics as a dictionary
    return {
        'final_balance': final_balance,
        'cumulative_return': cumulative_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'total_trades': total_trades,
        'trade_log': trade_log
    }
