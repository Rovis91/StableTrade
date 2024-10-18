from typing import Dict, List, Optional, Any, Union
from src.logger import setup_logger
import csv

class TradeManager:
    """
    TradeManager handles the opening, closing, and management of trades using a list of dictionaries.

    It supports both spot and futures markets by tracking the trade direction (buy/sell for spot, long/short for futures).
    Each trade includes details such as entry and exit prices, fees, stop-loss, take-profit, and trailing stops.

    Attributes:
        trades (List[Dict[str, Any]]): A list storing all trades (both open and closed).
        trade_counter (int): Counter for generating unique trade IDs.
        logger (logging.Logger): Logger instance for logging trade actions.
    """

    OPEN_STATUS = 'open'
    CLOSED_STATUS = 'closed'
    BUY = 'buy'
    SELL = 'sell'
    LONG = 'long'
    SHORT = 'short'

    def __init__(self, base_currency: str = "USD", logger=None):
        """
        Initialize the TradeManager.
        """
        self.trades: List[Dict[str, Any]] = []
        self.trade_counter: int = 0
        self.base_currency = base_currency
        self.logger = logger if logger else setup_logger('backtest_engine')

    def _find_trade_by_id(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """
        Find a trade by its ID within the trades list.

        Args:
            trade_id (int): The unique ID of the trade to find.

        Returns:
            Optional[Dict[str, Any]]: The trade dictionary if found; otherwise, None.
        """
        for trade in self.trades:
            if trade['id'] == trade_id:
                return trade
        return None

    def open_trade(
        self,
        asset_name: str,
        base_currency: str,
        asset_amount: float,
        base_amount: float,
        entry_price: float,
        entry_timestamp: int,
        entry_fee: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        direction: str = BUY,
        entry_reason: str = "buy_signal",
    ) -> Dict[str, Any]:
        """
        Open a new trade, tracking its entry details and market type (via direction).

        Args:
            asset_name (str): The name of the asset being traded.
            base_currency (str): The base currency used in the trade.
            asset_amount (float): The amount of the asset being traded.
            base_amount (float): The amount of the base currency used in the trade.
            entry_price (float): The price at which the trade is entered.
            entry_timestamp (int): The timestamp of when the trade is opened.
            entry_fee (float): The fee incurred when entering the trade.
            stop_loss (Optional[float], optional): The stop-loss price. Defaults to None.
            take_profit (Optional[float], optional): The take-profit price. Defaults to None.
            trailing_stop (Optional[float], optional): The trailing stop amount. Defaults to None.
            direction (str, optional): The direction of the trade ('buy', 'sell', 'long', 'short'). Defaults to BUY.
            entry_reason (str, optional): The reason for entering the trade. Defaults to "buy_signal".

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

        self.logger.info(
            f"Trade {self.trade_counter} opened for {asset_amount} {asset_name} "
            f"({base_amount} {base_currency}) as {direction} at price {entry_price}."
        )
        return trade

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_timestamp: int,
        exit_fee: float,
        exit_reason: str = "sell_signal",
    ) -> Dict[str, Any]:
        """
        Close an existing trade and update the trade database.

        Args:
            trade_id (int): The unique ID of the trade to close.
            exit_price (float): The price at which the trade is exited.
            exit_timestamp (int): The timestamp of when the trade is closed.
            exit_fee (float): The fee incurred when exiting the trade.
            exit_reason (str, optional): The reason for exiting the trade. Defaults to "sell_signal".

        Returns:
            Dict[str, Any]: The closed trade.
        """
        trade = self._find_trade_by_id(trade_id)

        if trade is None:
            self.logger.error(f"Trade with ID {trade_id} not found.")
            return None

        if trade['status'] != self.OPEN_STATUS:
            self.logger.error(f"Trade {trade_id} is already closed.")
            return None

        trade['exit_price'] = exit_price
        trade['exit_timestamp'] = exit_timestamp
        trade['exit_fee'] = exit_fee
        trade['status'] = self.CLOSED_STATUS
        trade['exit_reason'] = exit_reason

        if trade['direction'] in [self.BUY, self.LONG]:
            pl_base = (
                (exit_price - trade['entry_price']) * trade['asset_amount']
                - trade['entry_fee']
                - exit_fee
            )
        elif trade['direction'] in [self.SELL, self.SHORT]:
            pl_base = (
                (trade['entry_price'] - exit_price) * trade['asset_amount']
                - trade['entry_fee']
                - exit_fee
            )
        else:
            self.logger.error(
                f"Trade {trade_id} has an invalid direction: {trade['direction']}"
            )
            return None

        trade['profit_loss'] = pl_base

        self.logger.info(
            f"Trade {trade_id} closed at {exit_price}, reason: {exit_reason}, "
            f"exit_fee: {exit_fee}, profit/loss: {pl_base} {trade['base_currency']}"
        )
        return trade

    def get_trade(
        self, trade_id: Optional[int] = None, status: Optional[str] = None
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve trades by their ID or status (open or closed).

        Args:
            trade_id (Optional[int], optional): The ID of a specific trade to retrieve. Defaults to None.
            status (Optional[str], optional): The status of trades to retrieve ('open' or 'closed'). Defaults to None.

        Returns:
            Union[List[Dict[str, Any]], Dict[str, Any]]: A list of trades matching the criteria or a single trade if trade_id is provided.
        """
        if trade_id is not None:
            trade = self._find_trade_by_id(trade_id)
            if trade is None:
                self.logger.warning(f"Trade with ID {trade_id} not found.")
                return None
            return trade

        if status is not None:
            return [trade for trade in self.trades if trade['status'] == status]

        return self.trades

    def modify_trade_parameters(
        self, trade_id: int, stop_loss: Optional[float] = None, take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Modify the stop-loss or take-profit of an existing trade.

        Args:
            trade_id (int): The unique ID of the trade to modify.
            stop_loss (Optional[float], optional): The new stop-loss price. Defaults to None.
            take_profit (Optional[float], optional): The new take-profit price. Defaults to None.

        Returns:
            Dict[str, Any]: The modified trade.
        """
        trade = self._find_trade_by_id(trade_id)

        if trade is None:
            self.logger.error(f"Trade with ID {trade_id} not found.")
            return None

        if trade['status'] != self.OPEN_STATUS:
            self.logger.error(f"Cannot modify trade {trade_id} as it is already closed.")
            return None

        if stop_loss is not None:
            trade['stop_loss'] = stop_loss
        if take_profit is not None:
            trade['take_profit'] = take_profit

        self.logger.info(
            f"Modified trade {trade_id}: stop_loss={trade.get('stop_loss')}, take_profit={trade.get('take_profit')}"
        )
        return trade

    def export_trades_to_csv(self, filepath: str) -> None:
        """
        Export all trades (both open and closed) to a CSV file.

        Args:
            filepath (str): The path where the CSV file will be saved.

        Raises:
            IOError: If there is an error writing to the file.
        """
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
