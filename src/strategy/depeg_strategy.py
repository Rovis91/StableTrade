import pandas as pd
from src.strategy.base_strategy import Strategy

class DepegStrategy(Strategy):
    def __init__(self, depeg_threshold, trade_amount, stop_loss=None, take_profit=None, trailing_stop=None):
        """
        Initialize the DepegStrategy.

        Args:
            depeg_threshold (float): The percentage deviation to trigger a trade.
            trade_amount (float): The amount to trade as a percentage of the portfolio.
            stop_loss (float): Optional stop-loss level as a percentage of the entry price.
            take_profit (float): Optional take-profit level as a percentage of the entry price.
            trailing_stop (float): Optional trailing stop amount as a percentage of the entry price.
        """
        # Input validation
        if depeg_threshold <= 0:
            raise ValueError("depeg_threshold must be a positive value.")
        if trade_amount <= 0:
            raise ValueError("trade_amount must be a positive value.")
        
        self.depeg_threshold = depeg_threshold
        self.trade_amount = trade_amount
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop

    def get_required_indicators(self) -> dict:
        return {'SMA': [20, 50]}  

    def generate_signals(self, market_data: pd.Series) -> dict:
        
        # Use precomputed indicators (e.g., SMA_20 and SMA_50)
        sma_20 = market_data['SMA_20']
        sma_50 = market_data['SMA_50']
        
        # Compute the deviation
        deviation = (market_data['close'] - sma_20) / sma_20 * 100
        
        # Buy condition: price significantly below SMA_20 by the threshold (depeg event)
        if deviation <= -self.depeg_threshold:
            return {
                'action': 'buy',
                'amount': self.trade_amount,
                'stop_loss': self.stop_loss if self.stop_loss is not None else None,
                'take_profit': self.take_profit if self.take_profit is not None else None,
                'trailing_stop': self.trailing_stop if self.trailing_stop is not None else None
            }

        # Sell condition: price significantly above SMA_50 by the threshold (repeg event)
        elif deviation >= self.depeg_threshold:
            return {
                'action': 'sell',
                'amount': self.trade_amount,
                'stop_loss': self.stop_loss if self.stop_loss is not None else None,
                'take_profit': self.take_profit if self.take_profit is not None else None,
                'trailing_stop': self.trailing_stop if self.trailing_stop is not None else None
            }

        # Hold condition
        return {'action': 'hold'}

    def on_order_execution(self, order, portfolio):
    
        pass
