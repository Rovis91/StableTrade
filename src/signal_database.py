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
    
    def _validate_signal(self, signal: Dict[str, Any]) -> None:
        required_fields = ['timestamp', 'asset_name', 'action', 'amount', 'price']
        for field in required_fields:
            if field not in signal:
                raise ValueError(f"Missing required field: {field}")

        if signal['action'] not in ['buy', 'sell', 'close']:
            raise ValueError(f"Invalid action: {signal['action']}")

        if float(signal['amount']) <= 0:
            raise ValueError(f"Invalid amount: {signal['amount']}")

        if float(signal['price']) <= 0:
            raise ValueError(f"Invalid price: {signal['price']}")
    
    def get_signals(self, **filters: Any) -> List[Dict[str, Any]]:
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
        # Define allowed fields and their expected data types
        allowed_fields = {
            'trade_id': int,
            'status': str,
            'reason': str,
        }

        # Check if the provided field name is in the list of allowed fields
        if field_name not in allowed_fields:
            self.logger.error(f"Field '{field_name}' is not allowed to be updated.")
            raise ValueError(f"Field '{field_name}' cannot be updated.")

        # Check if the new value matches the expected type for the field
        expected_type = allowed_fields[field_name]
        if not isinstance(new_value, expected_type):
            self.logger.error(f"Field '{field_name}' expects value of type {expected_type.__name__}.")
            raise TypeError(f"Expected {expected_type.__name__} for field '{field_name}', got {type(new_value).__name__}.")

        # Iterate through signals to find the one matching the provided signal_id
        for signal in self.signals:
            if signal[self.COLUMN_SIGNAL_ID] == signal_id:
                # Update the specified field with the new value
                signal[field_name] = new_value
                self.logger.info(f"Updated signal {signal_id} field '{field_name}' to '{new_value}'")
                return

        # If the signal with the provided signal_id is not found, log an error and raise an exception
        self.logger.error(f"Signal {signal_id} not found in the database.")
        raise ValueError(f"Signal {signal_id} not found.")

    def update_trade_id(self, signal_id: int, trade_id: int) -> None:
        for signal in self.signals:
            if signal[self.COLUMN_SIGNAL_ID] == signal_id:
                signal[self.COLUMN_TRADE_ID] = trade_id
                self.logger.info(f"Updated signal {signal_id} with trade ID: {trade_id}")
                return

        self.logger.error(f"Signal {signal_id} not found in the database.")
        raise ValueError(f"Signal {signal_id} not found.")

    def to_csv(self, filepath: str) -> None:
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