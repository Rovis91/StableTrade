import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """
    Calculates the Sharpe Ratio for a given strategy's returns.
    
    :param returns: Series of returns (can be daily, monthly, etc.)
    :param risk_free_rate: Risk-free rate (defaults to 0)
    :return: Sharpe Ratio
    """
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0

def calculate_cumulative_return(initial_balance, final_balance):
    """
    Calculates the cumulative return over the entire backtest period.
    
    :param initial_balance: Initial account balance at the start of the backtest.
    :param final_balance: Final account balance at the end of the backtest.
    :return: Cumulative return as a percentage.
    """
    return (final_balance - initial_balance) / initial_balance

def calculate_max_drawdown(balance_series):
    """
    Calculates the maximum drawdown of the account balance during the backtest.
    
    :param balance_series: Series of account balances over time.
    :return: Maximum drawdown as a percentage.
    """
    running_max = np.maximum.accumulate(balance_series)
    drawdown = (running_max - balance_series) / running_max
    return np.max(drawdown) if len(drawdown) > 0 else 0

def calculate_profit_factor(trade_log):
    """
    Calculates the Profit Factor, which is the ratio of total profits to total losses.
    
    :param trade_log: A list of tuples representing trades, where each tuple contains ('BUY' or 'SELL', price).
    :return: Profit factor (ratio of total profits to total losses).
    """
    total_profit = 0
    total_loss = 0
    
    for trade in trade_log:
        if trade[0] == 'SELL':  # We calculate profit/loss on sells (closing trades)
            profit_or_loss = trade[2]  # This assumes profit or loss is tracked in the 3rd element of the tuple
            if profit_or_loss > 0:
                total_profit += profit_or_loss
            else:
                total_loss += abs(profit_or_loss)

    if total_loss == 0:
        return np.inf  # If no losses, profit factor is infinite.
    
    return total_profit / total_loss if total_loss > 0 else 0

def calculate_total_trades(trade_log):
    """
    Calculates the total number of trades executed during the backtest.
    
    :param trade_log: A list of tuples representing trades, where each tuple contains ('BUY' or 'SELL', price).
    :return: Total number of trades executed.
    """
    return len([trade for trade in trade_log if trade[0] in ('BUY', 'SELL')])

