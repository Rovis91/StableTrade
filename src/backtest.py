def backtest(df, initial_balance=1000, commission=0.001, position_size=1):
    """
    Backtests the strategy based on trading signals.
    
    :param df: DataFrame with signals
    :param initial_balance: Starting capital
    :param commission: Commission per trade
    :param position_size: Fraction of capital to allocate per trade
    :return: Final balance after backtest, total number of trades, and trade log
    """
    balance = initial_balance
    position = 0  # No initial position
    trades = 0
    trade_log = []

    for i in range(1, len(df)):
        signal = df.iloc[i]['signal']
        price = df.iloc[i]['close']
        
        if signal == 1 and position == 0:
            # Open a long position
            position = position_size * balance / price
            trades += 1
            trade_log.append(('BUY', price))
        elif signal == -1 and position > 0:
            # Close the long position
            balance += position * price * (1 - commission)
            position = 0
            trades += 1
            trade_log.append(('SELL', price))

    # Final balance after closing any remaining positions
    if position > 0:
        balance += position * df.iloc[-1]['close'] * (1 - commission)

    return balance, trades, trade_log
