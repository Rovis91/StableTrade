import logging

class TradeManager:
    def __init__(self):
        self.trades = []  # Single unified list for both open and closed trades
        self.trade_counter = 0  # Unique trade ID counter

    def open_trade(self, asset_name, amount, entry_price, entry_timestamp, entry_fee, stop_loss=None, take_profit=None, trailing_stop=None, entry_reason="buy_signal"):
        """
        Open a new trade and add it to the trade database.

        Args:
            asset_name (str): The market/pair (e.g., 'BTC/USD').
            amount (float): The size of the trade.
            entry_price (float): The price at which the trade is opened.
            entry_timestamp (float): The timestamp when the trade is opened.
            entry_fee (float): The fee applied when opening the trade.
            stop_loss (float, optional): Stop loss price.
            take_profit (float, optional): Take profit price.
            trailing_stop (float, optional): Trailing stop percentage.
            entry_reason (str, optional): Reason for entering the trade.

        Returns:
            dict: The opened trade.
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
            'entry_reason': entry_reason,
            'exit_reason': None,
            'stop_loss': stop_loss,  # Already market price
            'take_profit': take_profit,  # Already market price
            'trailing_stop': trailing_stop,  # Percentage
        }
        self.trades.append(trade)
        logging.info(f"Trade {self.trade_counter} opened for {asset_name} at {entry_price}, amount: {amount}, stop_loss: {stop_loss}, take_profit: {take_profit}")
        return trade

    def close_trade(self, trade_id, exit_price, exit_timestamp, exit_fee, exit_reason="sell_signal"):
        """
        Close an existing trade and update the trade database.

        Args:
            trade_id (int): The unique ID of the trade.
            exit_price (float): The price at which the trade is closed.
            exit_timestamp (float): The timestamp when the trade is closed.
            exit_fee (float): The fee applied when closing the trade.
            exit_reason (str, optional): Reason for exiting the trade.

        Returns:
            dict: The closed trade or None if the trade is not found.
        """
        for trade in self.trades:
            if trade['id'] == trade_id and trade['status'] == 'open':
                trade['exit_price'] = exit_price
                trade['exit_timestamp'] = exit_timestamp
                trade['exit_fee'] = exit_fee
                trade['status'] = 'closed'
                trade['exit_reason'] = exit_reason
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
