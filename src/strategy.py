class Strategy:
    def __init__(self, strategy_func, params):
        """
        Initialize a strategy with a function and its parameters.
        
        :param strategy_func: Function that defines the strategy logic
        :param params: Dictionary of parameters for the strategy
        """
        self.strategy_func = strategy_func
        self.params = params

    def apply(self, df):
        """
        Apply the strategy function to a DataFrame.
        
        :param df: DataFrame containing historical data
        :return: DataFrame with signals
        """
        return self.strategy_func(df, **self.params)


# Example strategy function (Simple Moving Average strategy)
def sma_strategy(df, offset, neutral_period, day_period):
    """
    Example of a simple SMA-based strategy.
    
    :param df: DataFrame with price data
    :param offset: Offset for long/short entry
    :param neutral_period: Period for the neutral SMA
    :param day_period: Period for the daily SMA
    :return: DataFrame with strategy signals
    """
    df['neutral_price'] = df['close'].rolling(window=neutral_period).mean()
    df['day_price'] = df['close'].rolling(window=day_period).mean()
    
    df['long_price'] = df['day_price'] * (1 - offset)
    df['short_price'] = df['day_price'] * (1 + offset)
    
    df['signal'] = 0
    df.loc[df['close'] <= df['long_price'], 'signal'] = 1
    df.loc[df['close'] >= df['short_price'], 'signal'] = -1
    
    return df
