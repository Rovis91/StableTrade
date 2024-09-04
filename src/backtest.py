def backtest(df, initial_balance=1000, commission=0.001, position_size=1):
    """
    Backtests the strategy based on trading signals with commission applied on both buy and sell.

    :param df: DataFrame with signals
    :param initial_balance: Starting capital
    :param commission: Commission per trade (e.g., 0.001 for 0.1%)
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

        # Buy (enter long position)
        if signal == 1 and position == 0:
            # Calculate the amount to invest, and apply the buy commission
            position = (position_size * balance) / price
            cost = position_size * balance * (1 + commission)  # Buying costs more due to commission
            balance -= cost
            trades += 1
            trade_log.append(('BUY', price))

        # Sell (exit long position)
        elif signal == -1 and position > 0:
            # Calculate the amount received from selling and apply the sell commission
            proceeds = position * price * (1 - commission)  # Selling gives less due to commission
            balance += proceeds
            position = 0
            trades += 1
            trade_log.append(('SELL', price))

    # Final balance after closing any remaining positions
    if position > 0:
        # Apply sell commission when closing the last open position
        proceeds = position * df.iloc[-1]['close'] * (1 - commission)
        balance += proceeds
        position = 0  # Position is closed

    return balance, trades, trade_log
