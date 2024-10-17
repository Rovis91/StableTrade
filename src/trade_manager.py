from typing import Dict, List, Optional, Any
from src.logger import setup_logger
import csv

class TradeManager:
    """
    TradeManager handles the opening, closing, and management of trades using a List of Dictionaries structure.

    It supports both spot and futures markets by tracking the trade direction (buy/sell for spot, long/short for futures).
    Each trade includes details such as entry and exit prices, fees, stop-loss, take-profit, and trailing stops.

    Attributes:
        trades (List[Dict[str, Any]]): A list storing all trades (both open and closed).
        trades_by_id (Dict[int, Dict[str, Any]]): A dictionary storing trades indexed by their unique trade IDs for fast retrieval.
        trade_counter (int): Counter for generating unique trade IDs.
        base_currency (str): The base currency for trades (e.g., 'USD').
        logger (logging.Logger): Logger instance for logging trade actions.
    """

    OPEN_STATUS = 'open'
    CLOSED_STATUS = 'closed'
    BUY = 'buy'
    SELL = 'sell'
    LONG = 'long'
    SHORT = 'short'

    def __init__(self, base_currency: str = "USD"):
        self.trades: List[Dict[str, Any]] = []
        self.trades_by_id: Dict[int, Dict[str, Any]] = {}
        self.trade_counter: int = 0
        self.base_currency = base_currency
        self.logger = setup_logger('trade_manager')

    def open_trade(self, asset_name: str, base_currency: str, asset_amount: float, 
                   base_amount: float, entry_price: float, entry_timestamp: int, 
                   entry_fee: float, stop_loss: Optional[float] = None, 
                   take_profit: Optional[float] = None, trailing_stop: Optional[float] = None, 
                   direction: str = BUY, entry_reason: str = "buy_signal") -> Dict[str, Any]:
        """
        Open a new trade, tracking its entry details and market type (via direction).
        
        Args:
            (... same as before ...)
        
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
            'status': self.OPEN_STATUS,
            'direction': direction,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': trailing_stop,
            'entry_reason': entry_reason,
            'exit_reason': None,
            'profit_loss': None
        }

        self.trades.append(trade)
        self.trades_by_id[self.trade_counter] = trade

        self.logger.info(
            f"Trade {self.trade_counter} opened for {asset_amount} {asset_name} "
            f"({base_amount} {base_currency}) as {direction} at price {entry_price}."
        )
        return trade

    def close_trade(self, trade_id: int, exit_price: float, exit_timestamp: int, 
                    exit_fee: float, exit_reason: str = "sell_signal") -> Optional[Dict[str, Any]]:
        """
        Close an existing trade and update the trade database.
        
        Args:
            (... same as before ...)
        
        Returns:
            Optional[Dict[str, Any]]: The closed trade or None if the trade is not found or already closed.
        """
        trade = self.trades_by_id.get(trade_id)

        if trade and trade['status'] == self.OPEN_STATUS:
            trade['exit_price'] = exit_price
            trade['exit_timestamp'] = exit_timestamp
            trade['exit_fee'] = exit_fee
            trade['status'] = self.CLOSED_STATUS
            trade['exit_reason'] = exit_reason

            if trade['direction'] in [self.BUY, self.LONG]:
                pl_base = (exit_price - trade['entry_price']) * trade['asset_amount'] - trade['entry_fee'] - exit_fee
            else:  # SELL or SHORT
                pl_base = (trade['entry_price'] - exit_price) * trade['asset_amount'] - trade['entry_fee'] - exit_fee

            trade['profit_loss'] = pl_base

            self.logger.info(
                f"Trade {trade_id} closed at {exit_price}, reason: {exit_reason}, "
                f"exit_fee: {exit_fee}, profit/loss: {pl_base} {trade['base_currency']}"
            )
            return trade

        self.logger.error(f"Attempt to close trade {trade_id} failed: trade not found or already closed.")
        return None

    def get_trade(self, trade_id: Optional[int] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve trades by their ID or status (open or closed).

        Args:
            (... same as before ...)

        Returns:
            List[Dict[str, Any]]: A list of trades that match the search criteria.
        """
        if trade_id is not None:
            trade = self.trades_by_id.get(trade_id)
            if not trade:
                self.logger.warning(f"Trade with ID {trade_id} not found.")
            return [trade] if trade else []

        if status is not None:
            return [trade for trade in self.trades if trade['status'] == status]

        return self.trades

    def modify_trade_parameters(self, trade_id: int, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Modify the stop-loss or take-profit of an existing trade.

        Args:
            (... same as before ...)

        Returns:
            Optional[Dict[str, Any]]: The modified trade or None if the trade is not found.
        """
        trade = self.trades_by_id.get(trade_id)

        if trade and trade['status'] == self.OPEN_STATUS:
            if stop_loss is not None:
                trade['stop_loss'] = stop_loss
            if take_profit is not None:
                trade['take_profit'] = take_profit

            self.logger.info(
                f"Modified trade {trade_id}: stop_loss={stop_loss}, take_profit={take_profit}"
            )
            return trade

        self.logger.error(f"Failed to modify trade {trade_id}: trade not found or already closed.")
        return None

    def export_to_csv(self, filepath: str) -> None:
        """
        Export all trades to a CSV file.

        Args:
            filepath (str): The path where the CSV file will be saved.
        """
        import csv

        try:
            if not self.trades:
                self.logger.warning("No trades to export.")
                return

            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = list(self.trades[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for trade in self.trades:
                    writer.writerow(trade)

            self.logger.info(f"Exported {len(self.trades)} trades to {filepath}")

        except Exception as e:
            self.logger.error(f"Error exporting trades to CSV: {str(e)}")
            raise
    
    def save_trades_to_csv(self, filepath: str) -> None:
        """
        Save all trades (both open and closed) to a CSV file.

        Args:
            filepath (str): The path where the CSV file will be saved.
        """
        try:
            if not self.trades:
                self.logger.warning("No trades to export.")
                return

            with open(filepath, 'w', newline='') as csvfile:
                # Assuming all trades have the same keys, use the first trade to get the fieldnames
                fieldnames = list(self.trades[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for trade in self.trades:
                    writer.writerow(trade)

            self.logger.info(f"Exported {len(self.trades)} trades to {filepath}")

        except Exception as e:
            self.logger.error(f"Error exporting trades to CSV: {str(e)}")
            raise