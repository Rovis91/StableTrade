import logging
from typing import Dict, List, Optional, Any

class TradeManager:
    """
    TradeManager handles the opening, closing, and management of trades.
    
    It supports both spot and futures markets by tracking the trade direction (buy/sell for spot, long/short for futures).
    Each trade includes stop-loss, take-profit, and trailing stop values (all in market prices, except trailing stop).
    """

    def __init__(self):
        self.trades: List[Dict[str, Any]] = []  # Stores all trades, both open and closed
        self.trade_counter: int = 0  # Counter for generating unique trade IDs
        self.logger = logging.getLogger(__name__)

    def open_trade(self, asset_name: str, base_currency: str, asset_amount: float, 
                   base_amount: float, entry_price: float, entry_timestamp: int, 
                   entry_fee: float, stop_loss: Optional[float] = None, 
                   take_profit: Optional[float] = None, trailing_stop: Optional[float] = None, 
                   direction: str = "buy", entry_reason: str = "buy_signal") -> Dict[str, Any]:
        """
        Open a new trade, tracking its entry details and market type (via direction).
        
        Args:
            asset_name (str): The asset being traded (e.g., 'BTC').
            base_currency (str): The base currency (e.g., 'USD').
            asset_amount (float): The amount of the asset being traded.
            base_amount (float): The amount in base currency.
            entry_price (float): The price at which the trade is opened.
            entry_timestamp (int): Timestamp of when the trade was opened.
            entry_fee (float): The fee applied to the trade upon entry (in base currency).
            stop_loss (float, optional): The stop loss price for the trade.
            take_profit (float, optional): The take profit price for the trade.
            trailing_stop (float, optional): The trailing stop percentage.
            direction (str): The trade direction - 'buy', 'sell', 'long', 'short'.
            entry_reason (str, optional): The reason for the trade entry.
        
        Returns:
            Dict[str, Any]: A dictionary representing the opened trade.
        """
        self.trade_counter += 1
        trade = {
            'id': self.trade_counter,
            'asset_name': asset_name,
            'base_currency': base_currency,
            'asset_amount': asset_amount,
            'base_amount': base_amount,
            'entry_price': entry_price,
            'entry_timestamp': entry_timestamp,
            'entry_fee': entry_fee,
            'exit_price': None,
            'exit_timestamp': None,
            'exit_fee': None,
            'status': 'open',
            'direction': direction,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': trailing_stop,
            'entry_reason': entry_reason,
            'exit_reason': None,
        }
        self.trades.append(trade)
        self.logger.info(f"Trade {self.trade_counter} opened for {asset_amount} {asset_name} "
                         f"({base_amount} {base_currency}) as {direction} at price {entry_price}.")
        return trade

    def close_trade(self, trade_id: int, exit_price: float, exit_timestamp: int, 
                    exit_fee: float, exit_reason: str = "sell_signal") -> Optional[Dict[str, Any]]:
        """
        Close an existing trade and update the trade database.
        
        Args:
            trade_id (int): The ID of the trade to close.
            exit_price (float): The exit price at which the trade was closed.
            exit_timestamp (int): The timestamp when the trade was closed.
            exit_fee (float): The fee applied to the trade upon exit (in base currency).
            exit_reason (str, optional): The reason for closing the trade.
        
        Returns:
            Optional[Dict[str, Any]]: The closed trade or None if the trade is not found or already closed.
        """
        for trade in self.trades:
            if trade['id'] == trade_id and trade['status'] == 'open':
                trade['exit_price'] = exit_price
                trade['exit_timestamp'] = exit_timestamp
                trade['exit_fee'] = exit_fee
                trade['status'] = 'closed'
                trade['exit_reason'] = exit_reason
                
                # Calculate profit/loss
                if trade['direction'] in ['buy', 'long']:
                    pl_base = (exit_price - trade['entry_price']) * trade['asset_amount'] - trade['entry_fee'] - exit_fee
                else:  # sell or short
                    pl_base = (trade['entry_price'] - exit_price) * trade['asset_amount'] - trade['entry_fee'] - exit_fee
                
                trade['profit_loss'] = pl_base
                
                self.logger.info(f"Trade {trade_id} closed at {exit_price}, reason: {exit_reason}, "
                                 f"exit_fee: {exit_fee}, profit/loss: {pl_base} {trade['base_currency']}")
                return trade

        self.logger.error(f"Attempt to close trade {trade_id} failed: trade not found or already closed.")
        return None

    def get_trade(self, trade_id: Optional[int] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve trades by their ID or status (open or closed).

        Args:
            trade_id (int, optional): The ID of the trade to retrieve.
            status (str, optional): The status of the trades to retrieve ('open' or 'closed').

        Returns:
            List[Dict[str, Any]]: A list of trades that match the search criteria.
        """
        # If trade_id is specified, return the trade with that specific ID.
        if trade_id is not None:
            trade = [trade for trade in self.trades if trade['id'] == trade_id]
            if not trade:
                self.logger.warning(f"Trade with ID {trade_id} not found.")
            return trade

        # If status is specified, return trades with the given status (open or closed).
        if status is not None:
            filtered_trades = [trade for trade in self.trades if trade['status'] == status]
            return filtered_trades

        # If neither trade_id nor status is provided, return all trades.
        return self.trades

    def modify_trade_parameters(self, trade_id: int, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Modify the stop-loss or take-profit of an existing trade.

        Args:
            trade_id (int): The unique ID of the trade.
            stop_loss (float, optional): New stop loss price.
            take_profit (float, optional): New take profit price.

        Returns:
            Optional[Dict[str, Any]]: The modified trade or None if the trade is not found.
        """
        for trade in self.trades:
            if trade['id'] == trade_id and trade['status'] == 'open':
                if stop_loss is not None:
                    trade['stop_loss'] = stop_loss
                if take_profit is not None:
                    trade['take_profit'] = take_profit
                self.logger.info(f"Modified trade {trade_id}: stop_loss={stop_loss}, take_profit={take_profit}")
                return trade

        self.logger.error(f"Failed to modify trade {trade_id}: trade not found or already closed.")
        return None