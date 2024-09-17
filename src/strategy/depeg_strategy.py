import pandas as pd
from src.strategy.base_strategy import Strategy
import logging

class DepegStrategy(Strategy):
    def __init__(self, depeg_threshold: float, trade_amount: float):
        """
        Initialize the DepegStrategy.

        Args:
            depeg_threshold (float): The percentage deviation to trigger a trade.
            trade_amount (float): The amount to trade as a percentage of the portfolio.
        """
        self.depeg_threshold = depeg_threshold
        self.trade_amount = trade_amount

    def get_required_indicators(self) -> dict:
        """
        Specify the indicators required for this strategy.

        Returns:
            dict: A dictionary specifying indicators and their parameters.
                  Example: {'SMA': [20, 50]}
        """
        return {'SMA': [20, 50]}  # Example: requires SMA for periods 20 and 50

    def generate_signals(self, market_data: pd.Series) -> dict:
        """
        Generate trading signals based on market data.

        Args:
            market_data (pd.Series): A row of market data (e.g., open, high, low, close, volume, SMA_20, SMA_50).

        Returns:
            dict: A dictionary containing the signals (e.g., {'action': 'buy', 'amount': 0.1}).
        """
        try:
            # Use precomputed indicators (e.g., SMA_20 and SMA_50)
            sma_20 = market_data['SMA_20']
            sma_50 = market_data['SMA_50']

            # Calculate deviations
            deviation_sma_20 = (market_data['close'] - sma_20) / sma_20 * 100
            deviation_sma_50 = (market_data['close'] - sma_50) / sma_50 * 100

            # Signal logic
            if deviation_sma_20 <= -self.depeg_threshold:
                logging.info(f"Buy signal generated: Deviation = {deviation_sma_20:.2f}%")
                return {'action': 'buy', 'amount': self.trade_amount}
            elif deviation_sma_50 >= self.depeg_threshold:
                logging.info(f"Sell signal generated: Deviation = {deviation_sma_50:.2f}%")
                return {'action': 'sell', 'amount': self.trade_amount}

        except KeyError as e:
            logging.error(f"Missing required market data for indicator: {e}")
            return {'action': 'hold'}

        # Default action if no condition is met
        return {'action': 'hold'}

    def on_order_execution(self, order, portfolio):
        """
        Handle actions to perform upon order execution.

        Args:
            order (Order): The order that was executed.
            portfolio (Portfolio): The portfolio object to update after execution.
        """
        # Placeholder for any updates on order execution
        pass
