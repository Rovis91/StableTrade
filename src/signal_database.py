import csv
import logging
from typing import Dict, List, Any, Optional
from src.logger import setup_logger

class SignalDatabase:
    STATUS_PENDING = 'pending'
    COLUMN_SIGNAL_ID = 'signal_id'
    COLUMN_TRADE_ID = 'trade_id'
    COLUMN_STATUS = 'status'
    COLUMN_REASON = 'reason'

    def __init__(self):
        """
        Initialize the SignalDatabase with default columns, an empty signal list, and a signal counter.
        Sets up logging for the class.
        """
        self.columns = {
            'timestamp': int,
            'asset_name': str,
            'action': str,
            'amount': float,
            'price': float,
            'stop_loss': float,
            'take_profit': float,
            'trailing_stop': float,
            self.COLUMN_SIGNAL_ID: int,
            self.COLUMN_TRADE_ID: Optional[int],
            self.COLUMN_STATUS: str,
            self.COLUMN_REASON: str
        }
        self.signals: List[Dict[str, Any]] = []
        self.signal_counter: int = 0
        self.logger = logging.getLogger(__name__)

    def add_signals(self, signals: List[Dict[str, Any]]) -> None:
        """
        Add a list of signals to the SignalDatabase.

        Args:
            signals (List[Dict[str, Any]]): A list of signal dictionaries to add.

        The method validates each signal, assigns a unique signal ID, and ensures all required fields are present.
        Fields missing in the input are assigned default values based on their type.
        """
        if not signals:
            return

        for signal in signals:
            self._validate_signal(signal)
            self.signal_counter += 1
            signal[self.COLUMN_SIGNAL_ID] = self.signal_counter
            signal[self.COLUMN_STATUS] = self.STATUS_PENDING
            signal[self.COLUMN_TRADE_ID] = None

            for column, dtype in self.columns.items():
                if column not in signal:
                    if dtype == Optional[int]:
                        signal[column] = None
                    elif dtype in (int, float):
                        signal[column] = 0
                    else:
                        signal[column] = ''
                elif dtype in (int, float):
                    signal[column] = dtype(float(signal[column]))
                elif dtype == Optional[int]:
                    signal[column] = int(float(signal[column])) if signal[column] is not None else None
                elif dtype == str:
                    signal[column] = str(signal[column])

            self.signals.append(signal)  

    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate a signal to ensure it contains required fields and valid values.

        Args:
            signal (Dict[str, Any]): The signal dictionary to validate.

        Returns:
            bool: True if the signal is valid, False otherwise.

        Raises:
            ValueError: If any required field is missing or contains invalid data.
        """
        required_fields = ['timestamp', 'asset_name', 'action', 'amount', 'price']
        for field in required_fields:
            if field not in signal:
                raise ValueError(f"Missing required field: {field}")

        action = signal['action']
        if action not in ['buy', 'sell', 'close']:
            raise ValueError(f"Invalid action: {action}")

        try:
            amount = float(signal['amount'])
            if amount <= 0 or (action in ['buy', 'sell'] and amount > 1):
                raise ValueError(f"Invalid amount in signal: {amount}. Must be between 0 and 1 for buy/sell actions.")
        except ValueError:
            raise ValueError(f"Invalid amount in signal: {signal['amount']}.")

        try:
            price = float(signal['price'])
            if price <= 0:
                raise ValueError(f"Invalid price: {price}.")
        except ValueError:
            raise ValueError(f"Invalid price in signal: {signal['price']}.")

        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        
        if stop_loss is not None:
            try:
                stop_loss = float(stop_loss)
                if stop_loss <= 0:
                    raise ValueError(f"Invalid stop_loss: {stop_loss}. Must be a valid number greater than 0.")
            except ValueError:
                raise ValueError(f"Invalid stop_loss: {signal['stop_loss']}. Must be a valid number greater than 0.")

        if take_profit is not None:
            try:
                take_profit = float(take_profit)
                if take_profit <= 0:
                    raise ValueError(f"Invalid take_profit: {take_profit}. Must be a valid number greater than 0.")
            except ValueError:
                raise ValueError(f"Invalid take_profit: {signal['take_profit']}. Must be a valid number greater than 0.")

        if action == 'close':
            if 'trade_id' not in signal:
                raise ValueError("Close signal is missing 'trade_id'")
            trades = self.trade_manager.get_trade(trade_id=signal['trade_id'])
            if not trades:
                raise ValueError(f"No trade found with id {signal['trade_id']}")
            if isinstance(trades, list):
                trade = trades[0] if trades else None
            else:
                trade = trades
            if not trade or trade.get('status') != 'open':
                raise ValueError(f"Trade {signal['trade_id']} is not open")

        return True

    def get_signals(self, **filters: Any) -> List[Dict[str, Any]]:
        """
        Retrieve signals from the database, optionally filtering based on field values.

        Args:
            **filters: Arbitrary keyword arguments to filter the signals (e.g., status='pending').

        Returns:
            List[Dict[str, Any]]: A list of signals that match the provided filters.
        """
        if not filters:
            return self.signals.copy()

        filtered_signals = [
            signal.copy() for signal in self.signals
            if all(signal.get(key) == value for key, value in filters.items())
        ]

        self.logger.info(f"Retrieved {len(filtered_signals)} signals matching filters: {filters}")
        return filtered_signals

    def update_signal_field(self, signal_id: int, field_name: str, new_value: Any) -> None:
        """
        Update a specific field for a signal identified by its signal_id.

        Args:
            signal_id (int): The ID of the signal to update.
            field_name (str): The name of the field to update ('trade_id', 'status', or 'reason').
            new_value (Any): The new value to assign to the field (should match expected type).

        Raises:
            ValueError: If the field is not allowed or the signal ID is not found.
            TypeError: If the new value type does not match the expected field type.
        """
        allowed_fields = {
            'trade_id': int,
            'status': str,
            'reason': str,
        }

        if field_name not in allowed_fields:
            self.logger.error(f"Field '{field_name}' is not allowed to be updated.")
            raise ValueError(f"Field '{field_name}' cannot be updated.")

        expected_type = allowed_fields[field_name]
        if not isinstance(new_value, expected_type):
            self.logger.error(f"Field '{field_name}' expects value of type {expected_type.__name__}.")
            raise TypeError(f"Expected {expected_type.__name__} for field '{field_name}', got {type(new_value).__name__}.")

        for signal in self.signals:
            if signal[self.COLUMN_SIGNAL_ID] == signal_id:
                signal[field_name] = new_value
                self.logger.info(f"Updated signal {signal_id} field '{field_name}' to '{new_value}'")
                return

        self.logger.error(f"Signal {signal_id} not found in the database.")
        raise ValueError(f"Signal {signal_id} not found.")

    def to_csv(self, filepath: str) -> None:
        """
        Export the signals to a CSV file.

        Args:
            filepath (str): The path to the CSV file to write.

        Raises:
            IOError: If there is an issue writing to the file.
            Exception: For any unexpected errors during export.
        """
        if not self.signals:
            self.logger.warning("No signals to export.")
            return

        try:
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = list(self.columns.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for signal in self.signals:
                    writer.writerow(signal)

            self.logger.info(f"Exported {len(self.signals)} signals to {filepath}")
        except IOError as e:
            self.logger.error(f"Error exporting signals to CSV: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while exporting to CSV: {str(e)}")
            raise