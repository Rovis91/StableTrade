import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from src.strategy.base_strategy import Strategy

class DepegStrategy(Strategy):
    def __init__(self, market: str, trade_manager: Any, depeg_threshold: float, trade_amount: float, 
                 stop_loss: Optional[float] = None, take_profit: Optional[float] = None, 
                 trailing_stop: Optional[float] = None, verbose: bool = False):
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
            verbose (bool): If True, enables verbose DEBUG logging.
        """
        super().__init__(market, trade_manager)

        self.config = {
            'market_type': 'spot',
            'fees': {'entry': 0.001, 'exit': 0.001},
            'max_trades': 5,  
            'max_exposure': 0.70
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
        self.verbose = verbose

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        self.logger.info(f"DepegStrategy initialized for {market} with threshold {depeg_threshold}%")
        if self.verbose:
            self.logger.debug(f"Strategy parameters: trade_amount={trade_amount}, "
                              f"stop_loss={stop_loss}, take_profit={take_profit}, "
                              f"trailing_stop={trailing_stop}")

    def get_required_indicators(self) -> dict:
        """Return the indicators required by this strategy."""
        indicators = {'SMA': [20, 50]}
        self.logger.debug(f"Required indicators: {indicators}")
        return indicators

    def calculate_stop_loss(self, current_price: float) -> Optional[float]:
        """Calculate the stop loss based on the current price and stop loss percentage."""
        if self.stop_loss_percent is not None:
            stop_loss = current_price * (1 - self.stop_loss_percent)
            self.logger.debug(f"Calculated stop loss: {stop_loss} for current price: {current_price}")
            return stop_loss
        return None

    def calculate_take_profit(self, current_price: float, sma_20: float) -> float:
        """Calculate the take profit based on either a percentage or the SMA_20 value."""
        if self.take_profit_percent is not None:
            take_profit = current_price * (1 + self.take_profit_percent)
            self.logger.debug(f"Calculated take profit: {take_profit} for current price: {current_price}")
            return take_profit
        self.logger.debug(f"Using SMA_20 as take profit: {sma_20}")
        return sma_20

    def generate_signal(self, market_data: pd.Series, active_trades: list, portfolio_value: float, portfolio_cash: float) -> dict:
        """Generate trading signals based on the current market data and active trades."""
        sma_20 = market_data['SMA_20']
        current_price = market_data['close']

        if pd.isna(sma_20):
            self.logger.warning("SMA_20 is NaN, skipping signal generation.")
            return {}

        # Check for sell signals first (risk management)
        for trade in active_trades:
            if trade['asset_name'] == self.market:
                if self.check_stop_loss(trade, current_price):
                    self.logger.info(f"Stop loss triggered for trade {trade['id']}")
                    return self.create_sell_signal(trade, current_price, "stop_loss")
                if self.check_take_profit(trade, current_price):
                    self.logger.info(f"Take profit triggered for trade {trade['id']}")
                    return self.create_sell_signal(trade, current_price, "take_profit")

        # Only check for buy signal if no sell signals were generated
        deviation = (current_price - sma_20) / sma_20 * 100
        if deviation <= -self.depeg_threshold:
            self.logger.info(f"Buy condition met. Deviation: {deviation:.2f}% below SMA_20.")
            signal = {
                'action': 'buy',
                'amount': self.trade_amount,
                'asset_name': self.market,
                'price': current_price,
                'stop_loss': self.calculate_stop_loss(current_price),
                'take_profit': self.calculate_take_profit(current_price, sma_20),
                'trailing_stop': self.trailing_stop_percent,
            }
            if self.verbose:
                self.logger.debug(f"Generated buy signal: {signal}")
            return signal

        self.logger.debug(f"No signal generated. Current deviation: {deviation:.2f}%")
        return {}  # No signal generated

    def check_stop_loss(self, trade: dict, current_price: float) -> bool:
        """Check if the stop loss condition is met for a trade."""
        is_triggered = trade['stop_loss'] is not None and current_price <= trade['stop_loss']
        if self.verbose:
            self.logger.debug(f"Stop loss check for trade {trade['id']}: "
                              f"current_price={current_price}, stop_loss={trade['stop_loss']}, "
                              f"triggered={is_triggered}")
        return is_triggered

    def check_take_profit(self, trade: dict, current_price: float) -> bool:
        """Check if the take profit condition is met for a trade."""
        is_triggered = trade['take_profit'] is not None and current_price >= trade['take_profit']
        if self.verbose:
            self.logger.debug(f"Take profit check for trade {trade['id']}: "
                              f"current_price={current_price}, take_profit={trade['take_profit']}, "
                              f"triggered={is_triggered}")
        return is_triggered

    def create_sell_signal(self, trade: dict, current_price: float, reason: str) -> dict:
        """Create a sell signal for a trade."""
        signal = {
            'action': 'sell',
            'amount': trade['asset_amount'],
            'asset_name': self.market,
            'price': current_price,
            'trade_id': trade['id'],
            'reason': reason
        }
        self.logger.info(f"Created sell signal for trade {trade['id']}: {signal}")
        return signal

    def update_trailing_stop(self, active_trades: List[Dict[str, Any]], asset_name: str, market_prices: Dict[str, float]) -> bool:
        """Update the trailing stop for active trades of a specific asset."""
        updated = False
        current_price = market_prices.get(asset_name)
        
        if current_price is None:
            self.logger.warning(f"No current price found for {asset_name}. Skipping trailing stop update.")
            return updated

        for trade in active_trades:
            if trade['asset_name'] == asset_name and trade['trailing_stop'] is not None:
                direction = trade['direction']
                trailing_stop_pct = trade['trailing_stop'] / 100  # Convert percentage to decimal
                current_stop_loss = trade['stop_loss']

                if direction in ['buy', 'long']:
                    new_stop_loss = current_price * (1 - trailing_stop_pct)
                    if new_stop_loss > current_stop_loss:
                        updated = self.trade_manager.modify_trade_parameters(trade_id=trade['id'], stop_loss=new_stop_loss)
                elif direction in ['sell', 'short']:
                    new_stop_loss = current_price * (1 + trailing_stop_pct)
                    if new_stop_loss < current_stop_loss:
                        updated = self.trade_manager.modify_trade_parameters(trade_id=trade['id'], stop_loss=new_stop_loss)

                if updated:
                    self.logger.info(f"Updated trailing stop for trade {trade['id']} ({asset_name}). "
                                     f"New stop loss: {new_stop_loss:.8f}")
                    if self.verbose:
                        self.logger.debug(f"Trailing stop update details: direction={direction}, "
                                          f"current_price={current_price}, trailing_stop_pct={trailing_stop_pct}, "
                                          f"old_stop_loss={current_stop_loss}, new_stop_loss={new_stop_loss}")

        return updated

    def log_strategy_state(self) -> None:
        """Log the current state of the strategy."""
        self.logger.info(f"DepegStrategy state for {self.market}:")
        self.logger.info(f"Depeg threshold: {self.depeg_threshold}%")
        self.logger.info(f"Trade amount: {self.trade_amount}")
        self.logger.info(f"Stop loss: {self.stop_loss_percent}")
        self.logger.info(f"Take profit: {self.take_profit_percent}")
        self.logger.info(f"Trailing stop: {self.trailing_stop_percent}")
        if self.verbose:
            self.logger.debug(f"Full strategy configuration: {self.config}")

