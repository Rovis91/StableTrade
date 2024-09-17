from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    """

    @abstractmethod
    def get_required_indicators(self) -> dict:
        """
        Specify the indicators required for this strategy.

        Returns:
            dict: A dictionary specifying indicators and their parameters.
                  Example: {'SMA': [20, 50]}
        """
        pass

    @abstractmethod
    def generate_signals(self, market_data: pd.Series) -> dict:
        """
        Generate trading signals based on market data.
        
        Args:
            market_data (pd.Series): A row of market data (e.g., open, high, low, close, volume).
        
        Returns:
            dict: A dictionary containing the signals (e.g., {'action': 'buy', 'amount': 0.1}).
        """
        pass

    @abstractmethod
    def on_order_execution(self, order, portfolio):
        """
        Handle actions to perform upon order execution (e.g., updating state, logging).

        Args:
            order (Order): The order that was executed.
            portfolio (Portfolio): The portfolio object to update after execution.
        """
        pass
