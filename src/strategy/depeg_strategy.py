import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from src.strategy.base_strategy import Strategy

class DepegStrategy(Strategy):
    def __init__(self, market: str, trade_manager: Any, depeg_threshold: float, trade_amount: float, 
                 stop_loss: Optional[float] = None, take_profit: Optional[float] = None, 
                 trailing_stop: Optional[float] = None):
        """
        Initialize the DepegStrategy with market-specific configurations.

        Args:
            market (str): The market this strategy operates on (e.g., BTC/USD).
            trade_manager (Any): The trade manager responsible for executing and managing trades.
            depeg_threshold (float): The percentage deviation to trigger a trade.
            trade_amount (float): The amount to trade as a percentage of the portfolio's available cash.
            stop_loss (Optional[float]): Stop-loss level as a percentage of the entry price.
            take_profit (Optional[float]): Take-profit level as a percentage of the entry price.
            trailing_stop (Optional[float]): Trailing stop amount as a percentage of the entry price.
        """
        super().__init__(market, trade_manager)

        self.config = {
            'market_type': 'spot',
            'fees': {'entry': 0.001, 'exit': 0.001},
            'max_trades': 10,  
            'max_exposure': 0.99
        }

        if depeg_threshold <= 0:
            raise ValueError("depeg_threshold must be a positive value.")
        if trade_amount <= 0:
            raise ValueError("trade_amount must be a positive value.")
        
        self.depeg_threshold = depeg_threshold
        self.trade_amount = trade_amount
        self.stop_loss_percent = stop_loss
        self.take_profit_percent = take_profit
        self.trailing_stop_percent = trailing_stop
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"DepegStrategy initialized for {market} with threshold {depeg_threshold}%")

    def get_required_indicators(self) -> dict:
        """Return the indicators required by this strategy."""
        return {'SMA': [20, 50]}

    def calculate_stop_loss(self, current_price: float) -> Optional[float]:
        """Calculate the stop loss based on the current price and stop loss percentage."""
        if self.stop_loss_percent is not None:
            return current_price * (1 - self.stop_loss_percent)
        return None

    def calculate_take_profit(self, current_price: float, sma_20: float) -> float:
        """Calculate the take profit based on either a percentage or the SMA_20 value."""
        if self.take_profit_percent is not None:
            return current_price * (1 + self.take_profit_percent)
        return sma_20

    def generate_signal(self, market_data: pd.Series, active_trades: list, portfolio_value: float, portfolio_cash: float) -> dict:
        """Generate trading signals based on the current market data and active trades."""
        sma_20 = market_data['SMA_20']
        current_price = market_data['close']

        if pd.isna(sma_20):
            return {}

        deviation = (current_price - 1) * 100
        if deviation <= -self.depeg_threshold:
            self.logger.info(f"Buy condition met. Deviation: {deviation:.2f}% below SMA_20.")
            return {
                'action': 'buy',
                'amount': self.trade_amount,
                'asset_name': self.market,
                'price': current_price,
                'stop_loss': self.calculate_stop_loss(current_price),
                'take_profit': self.calculate_take_profit(current_price, sma_20),
                'trailing_stop': self.trailing_stop_percent,
            }

        return {}  # No signal generated
