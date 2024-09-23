import pandas as pd
from src.strategy.base_strategy import Strategy
import logging

class DepegStrategy(Strategy):
    def __init__(self, market, trade_manager, depeg_threshold, trade_amount, stop_loss=None, take_profit=None, trailing_stop=None):
        """
        Initialize the DepegStrategy with market-specific configurations.

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

        # Market-specific configurations
        self.config = {
            'market_type': 'spot',  # Defines this as a spot market
            'fees': {
                'entry': 0.001,  # 0.1% entry fee
                'exit': 0.001   # 0.1% exit fee
            }
        }

        if depeg_threshold <= 0:
            raise ValueError("depeg_threshold must be a positive value.")
        if trade_amount <= 0:
            raise ValueError("trade_amount must be a positive value.")
        
        self.depeg_threshold = depeg_threshold
        self.trade_amount = trade_amount
        self.stop_loss_percent = stop_loss  # Store percentages
        self.take_profit_percent = take_profit  # Store percentages
        self.trailing_stop_percent = trailing_stop  # Store percentages
        self.logger = logging.getLogger(__name__)

    def get_required_indicators(self) -> dict:
        """
        Return the indicators required by this strategy.

        Returns:
            dict: A dictionary specifying indicators (e.g., {'SMA': [20, 50]}).
        """
        return {'SMA': [20, 50]}  # Example: Use two Simple Moving Averages (SMA_20 and SMA_50)

    def calculate_stop_loss(self, current_price):
        """
        Calculate the stop loss based on the current price and stop loss percentage.

        Args:
            current_price (float): The current market price of the asset.

        Returns:
            float or None: The stop loss price or None if stop_loss_percent is not set.
        """
        if self.stop_loss_percent is not None:
            stop_loss_price = current_price * (1 - self.stop_loss_percent)
            self.logger.debug(f"Calculated stop loss at {stop_loss_price} based on {current_price} and stop_loss_percent={self.stop_loss_percent}")
            return stop_loss_price
        return None

    def calculate_take_profit(self, current_price, sma_20):
        """
        Calculate the take profit based on either a percentage or the SMA_20 value.

        Args:
            current_price (float): The current market price of the asset.
            sma_20 (float): The 20-period Simple Moving Average for the asset.

        Returns:
            float or None: The take profit price or None if not set.
        """
        if self.take_profit_percent is not None:
            take_profit_price = current_price * (1 + self.take_profit_percent)
            self.logger.debug(f"Calculated take profit at {take_profit_price} based on {current_price} and take_profit_percent={self.take_profit_percent}")
            return take_profit_price
        else:
            self.logger.debug(f"Using SMA_20 as take profit: {sma_20}")
            return sma_20  # Default to using SMA_20 as the take profit level

    def generate_signal(self, market_data: pd.Series, active_trades: list) -> dict:
        """
        Generate trading signals based on the current market data and active trades.

        Args:
            market_data (pd.Series): A row of market data (e.g., open, high, low, close, volume).
            active_trades (list): A list of active trades to check before generating new signals.

        Returns:
            dict: A dictionary containing the trading signal with action and trade details.
        """
        # Use precomputed indicators (e.g., SMA_20)
        sma_20 = market_data['SMA_20']
        current_price = market_data['close']

        if pd.isna(sma_20):
            self.logger.error("SMA_20 is NaN, skipping signal generation.")
            return {}

        # Compute the deviation from SMA_20
        deviation = (current_price - sma_20) / sma_20 * 100

        # Buy condition: price significantly below SMA_20 by the threshold (depeg event)
        if deviation <= -self.depeg_threshold:
            self.logger.info(f"Buy condition met. Deviation: {deviation:.2f}% below SMA_20.")

            # Calculate stop loss and take profit prices based on the current price
            stop_loss_price = self.calculate_stop_loss(current_price)
            take_profit_price = self.calculate_take_profit(current_price, sma_20)

            signal = {
                'action': 'buy',
                'amount': self.trade_amount,
                'asset_name': self.market,
                'price': current_price,
                'stop_loss': stop_loss_price,  # Calculated stop loss
                'take_profit': take_profit_price,  # Calculated take profit
                'trailing_stop': self.trailing_stop_percent  # Trailing stop percentage
            }
            self.logger.info(f"Generated buy signal: {signal}")
            return signal

        self.logger.debug(f"No buy signal generated. Current deviation: {deviation:.2f}%")
        return {}  # No signal generated
