import pandas as pd
from src.strategy.base_strategy import Strategy
import logging

class DepegStrategy(Strategy):
    def __init__(self, market, trade_manager, depeg_threshold, trade_amount, stop_loss=None, take_profit=None, trailing_stop=None):
        """
        Initialize the DepegStrategy.

        Args:
            market (str): The market this strategy operates on (e.g., BTC/USD).
            trade_manager (TradeManager): The trade manager responsible for executing and managing trades.
            depeg_threshold (float): The percentage deviation to trigger a trade.
            trade_amount (float): The amount to trade as a percentage of the portfolio.
            stop_loss (float): Optional stop-loss level as a percentage of the entry price.
            take_profit (float): Optional take-profit level as a percentage of the entry price.
            trailing_stop (float): Optional trailing stop amount as a percentage of the entry price.
        """
        super().__init__(market, trade_manager)

        if depeg_threshold <= 0:
            raise ValueError("depeg_threshold must be a positive value.")
        if trade_amount <= 0:
            raise ValueError("trade_amount must be a positive value.")
        
        self.depeg_threshold = depeg_threshold
        self.trade_amount = trade_amount
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.logger = logging.getLogger(__name__)

    def get_required_indicators(self) -> dict:
        """
        Return the indicators required by this strategy.

        Returns:
            dict: A dictionary specifying indicators (e.g., {'SMA': [20, 50]}).
        """
        return {'SMA': [20, 50]}  # Example: Use two Simple Moving Averages (SMA_20 and SMA_50)

    def generate_signals(self, market_data: pd.Series, active_trades: list) -> dict:
        """
        Generate trading signals based on the current market data and active trades.

        Args:
            market_data (pd.Series): A row of market data (e.g., open, high, low, close, volume).
            active_trades (list): A list of active trades to check before generating new signals.

        Returns:
            dict: A dictionary containing the trading signal with action and trade details.
        """
        # Use precomputed indicators (e.g., SMA_20 and SMA_50)
        sma_20 = market_data['SMA_20']
        sma_50 = market_data['SMA_50']

        # Compute the deviation from SMA_20
        deviation = (market_data['close'] - sma_20) / sma_20 * 100

        # Buy condition: price significantly below SMA_20 by the threshold (depeg event)
        if deviation <= -self.depeg_threshold:
            self.logger.info(f"buy condition met ")
            return {
                'action': 'buy',
                'amount': self.trade_amount,
                'asset_name': self.market,
                'price': market_data['close'],
                'stop_loss': self.stop_loss if self.stop_loss is not None else None,
                'take_profit': self.take_profit if self.take_profit is not None else None,
                'trailing_stop': self.trailing_stop if self.trailing_stop is not None else None
            }

        # Sell condition: price significantly above SMA_50 by the threshold (repeg event)
        elif deviation >= self.depeg_threshold:
            return {
                'action': 'sell',
                'amount': self.trade_amount,
                'asset_name': self.market,
                'price': market_data['close'],
                'stop_loss': self.stop_loss if self.stop_loss is not None else None,
                'take_profit': self.take_profit if self.take_profit is not None else None,
                'trailing_stop': self.trailing_stop if self.trailing_stop is not None else None
            }

        # If no buy or sell condition is met, return an empty dictionary (no "hold")
        return {}


    def on_order_execution(self, order, portfolio):
        """
        Handle post-order execution logic, if needed.

        Args:
            order (dict): The order that was executed.
            portfolio (Portfolio): The portfolio managing trades.
        """
        pass  # No specific action required after order execution for this strategy.
