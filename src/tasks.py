from celery import Celery
from src.backtest import backtest
import pandas as pd
from src.strategy import sma_strategy, Strategy

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
    
    return {
        'final_balance': final_balance,
        'number_of_trades': trades,
        'trade_log': trade_log
    }
