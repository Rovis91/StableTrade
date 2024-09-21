from abc import ABC, abstractmethod
import pandas as pd
import logging

class Strategy(ABC):
    """
    Abstract base class for all trading strategies. Each strategy operates on a specific market (e.g., BTC/USD).
    """

    def __init__(self, market: str, trade_manager):
        """
        Initialize the strategy.

        Args:
            market (str): The market this strategy operates on (e.g., BTC/USD).
            trade_manager (TradeManager): The trade manager responsible for executing and managing trades.
        """
        self.market = market
        self.trade_manager = trade_manager
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def get_required_indicators(self) -> dict:
        """
        Specify the indicators required for this strategy.

        Returns:
            dict: A dictionary specifying indicators and their parameters.
        """
        pass

    @abstractmethod
    def generate_signals(self, market_data: pd.Series, active_trades: list) -> dict:
        """
        Generate trading signals based on market data and current active trades.
        
        Args:
            market_data (pd.Series): A row of market data (e.g., open, high, low, close, volume).
            active_trades (list): A list of active trades to check before generating new signals.
        
        Returns:
            dict: A dictionary containing the trading signal with details such as:
                  {'action': 'buy', 'amount': 0.1, 'asset_name': 'BTC/USD', 'price': 50000, 'stop_loss': 45000, 'take_profit': 55000}
        """
        pass

    def log_signal(self, signal: dict):
        """
        Helper function to log the generated signal.
        
        Args:
            signal (dict): The signal dictionary to log.
        """
        if signal:
            self.logger.info(f"Generated signal: {signal}")
        else:
            self.logger.info(f"No signal generated for {self.market} at this time.")

    def log_indicator_error(self, indicator_name: str):
        """
        Helper function to log an error if a required indicator is missing or incorrect.
        
        Args:
            indicator_name (str): The name of the missing or incorrect indicator.
        """
        self.logger.error(f"Required indicator '{indicator_name}' is missing or has invalid data for {self.market}.")
