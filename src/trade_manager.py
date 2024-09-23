import logging

class TradeManager:
    """
    TradeManager handles the opening, closing, and management of trades.
    
    It supports both spot and futures markets by tracking the trade direction (buy/sell for spot, long/short for futures).
    Each trade includes stop-loss, take-profit, and trailing stop values (all in market prices, except trailing stop).
    """

    def __init__(self):
        self.trades = []  # Stores all trades, both open and closed
        self.trade_counter = 0  # Counter for generating unique trade IDs

    def open_trade(self, asset_name: str, amount: float, entry_price: float, entry_timestamp: int, 
                   entry_fee: float, stop_loss: float = None, take_profit: float = None, 
                   trailing_stop: float = None, direction: str = "buy", 
                   entry_reason: str = "buy_signal") -> dict:
        """
        Open a new trade, tracking its entry details and market type (via direction).
        
        Args:
            asset_name (str): The asset being traded (e.g., 'BTC/USD').
            amount (float): The size of the trade.
            entry_price (float): The price at which the trade is opened.
            entry_timestamp (int): Timestamp of when the trade was opened.
            entry_fee (float): The fee applied to the trade upon entry.
            stop_loss (float, optional): The stop loss price for the trade.
            take_profit (float, optional): The take profit price for the trade.
            trailing_stop (float, optional): The trailing stop percentage (still in percentage form).
            direction (str): The trade direction - 'buy', 'sell', 'long', 'short'.
            entry_reason (str, optional): The reason for the trade entry (e.g., buy signal).
        
        Returns:
            dict: A dictionary representing the opened trade.
        """
        self.trade_counter += 1
        trade = {
            'id': self.trade_counter,
            'asset_name': asset_name,
            'amount': amount,
            'entry_price': entry_price,
            'entry_timestamp': entry_timestamp,
            'entry_fee': entry_fee,
            'exit_price': None,
            'exit_timestamp': None,
            'exit_fee': None,
            'status': 'open',
            'direction': direction,  # e.g., 'buy', 'sell', 'long', 'short'
            'stop_loss': stop_loss,  # Market price for stop-loss
            'take_profit': take_profit,  # Market price for take-profit
            'trailing_stop': trailing_stop,  # Percentage for trailing stop
            'entry_reason': entry_reason,  # Reason for trade entry (optional)
            'exit_reason': None,  # Will be set when the trade is closed
        }
        self.trades.append(trade)
        logging.info(f"Trade {self.trade_counter} opened for {asset_name} as {direction} at price {entry_price}.")
        return trade

    def close_trade(self, trade_id: int, exit_price: float, exit_timestamp: int, 
                    exit_fee: float, exit_reason: str = "sell_signal") -> dict:
        """
        Close an existing trade and update the trade database.
        
        Args:
            trade_id (int): The ID of the trade to close.
            exit_price (float): The exit price at which the trade was closed.
            exit_timestamp (int): The timestamp when the trade was closed.
            exit_fee (float): The fee applied to the trade upon exit.
            exit_reason (str, optional): The reason for closing the trade.
        
        Returns:
            dict: The closed trade or None if the trade is not found or already closed.
        """
        for trade in self.trades:
            if trade['id'] == trade_id and trade['status'] == 'open':
                trade['exit_price'] = exit_price
                trade['exit_timestamp'] = exit_timestamp
                trade['exit_fee'] = exit_fee
                trade['status'] = 'closed'
                trade['exit_reason'] = exit_reason  # Reason for closing the trade
                logging.info(f"Trade {trade_id} closed at {exit_price}, reason: {exit_reason}, exit_fee: {exit_fee}")
                return trade

        logging.error(f"Attempt to close trade {trade_id} failed: trade not found or already closed.")
        return None

    def get_trade(self, trade_id=None, status=None):
        """
        Retrieve trades by their ID or status (open or closed).

        Args:
            trade_id (int, optional): The ID of the trade to retrieve.
            status (str, optional): The status of the trades to retrieve ('open' or 'closed').

        Returns:
            list: A list of trades that match the search criteria.
        """
        # If trade_id is specified, return the trade with that specific ID.
        if trade_id is not None:
            trade = [trade for trade in self.trades if trade['id'] == trade_id]
            if not trade:
                logging.warning(f"Trade with ID {trade_id} not found.")
            return trade

        # If status is specified, return trades with the given status (open or closed).
        if status is not None:
            filtered_trades = [trade for trade in self.trades if trade['status'] == status]
            return filtered_trades

        # If neither trade_id nor status is provided, return all trades.
        return self.trades

    def modify_trade_parameters(self, trade_id, stop_loss=None, take_profit=None):
        """
        Modify the stop-loss or take-profit of an existing trade.

        Args:
            trade_id (int): The unique ID of the trade.
            stop_loss (float, optional): New stop loss price.
            take_profit (float, optional): New take profit price.

        Returns:
            dict: The modified trade or None if the trade is not found.
        """
        for trade in self.trades:
            if trade['id'] == trade_id and trade['status'] == 'open':
                if stop_loss is not None:
                    trade['stop_loss'] = stop_loss
                if take_profit is not None:
                    trade['take_profit'] = take_profit
                logging.info(f"Modified trade {trade_id}: stop_loss={stop_loss}, take_profit={take_profit}")
                return trade

        logging.error(f"Failed to modify trade {trade_id}: trade not found or already closed.")
        return None
