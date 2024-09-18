from abc import ABC, abstractmethod
import pandas as pd

from src.portfolio import Portfolio

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
                  Example:
                  {
                      'SMA': [20, 50],  # Simple Moving Averages for 20 and 50 periods
                      'EMA': [14],      # Exponential Moving Average for 14 periods
                      ...
                  }
                  The keys should be the names of the indicators, and the values should be lists of required parameters.
        """
        pass

    @abstractmethod
    def generate_signals(self, market_data: pd.Series) -> dict:
        """
        Generate trading signals based on market data.
        
        Args:
            market_data (pd.Series): A row of market data (e.g., open, high, low, close, volume).
                                     The series is expected to contain any required indicators as columns.
        
        Returns:
            dict: A dictionary containing the signals.
                  Example:
                  {
                      'action': 'buy',  # 'buy', 'sell', or 'hold'
                      'amount': 0.1,    # Amount to buy/sell
                      'stop_loss': 0.95, # Optional stop loss level
                      'take_profit': 1.05 # Optional take profit level
                  }
                  This dictionary must contain at least 'action' and 'amount' keys.
        """
        pass

    @abstractmethod
    def on_order_execution(self, order: dict, portfolio: 'Portfolio'):
        """
        Handle actions to perform upon order execution (e.g., updating state, logging).

        Args:
            order (dict): The order that was executed. 
                          Expected keys:
                          {
                              'type': 'market',  # Order type
                              'amount': 0.1,     # Executed amount
                              'executed_price': 100.0, # Price at which the order was executed
                              'timestamp': 1234567890.0 # Timestamp of the execution
                          }
            portfolio (Portfolio): The portfolio object to update after execution.

        Notes:
            This method is called after every successful order execution.
            Implementations can use this hook to update strategy-specific state or metrics.
        """
        pass
