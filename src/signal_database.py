import uuid
import csv
from typing import Dict, List, Any, Optional
from src.logger import setup_logger

class SignalDatabase:
    """
    A database for managing trading signals using a List of Dictionaries structure.

    This class provides methods for adding, retrieving, and updating trading signals,
    as well as exporting them to a CSV file. It uses a list of dictionaries for 
    efficient memory usage and faster operations compared to a pandas DataFrame.

    Attributes
    ----------
    STATUS_PENDING : str
        Constant representing the pending status for new signals.
    COLUMN_SIGNAL_ID : str
        Constant for the column name of the signal ID.
    COLUMN_STATUS : str
        Constant for the column name of the signal status.
    COLUMN_REASON : str
        Constant for the column name of the status change reason.
    columns : dict
        Dictionary defining the structure and types of signal data.
    signals : List[Dict[str, Any]]
        List storing all the signal dictionaries.
    logger : logging.Logger
        Logger instance for the SignalDatabase.

    Methods
    -------
    add_signals(signals: List[Dict[str, Any]]) -> None
        Add new signals to the database.
    get_signals(**filters) -> List[Dict[str, Any]]
        Retrieve signals based on provided filters.
    update_signal_status(signal_id: str, new_status: str, reason: Optional[str] = None) -> None
        Update the status of a specific signal.
    export_to_csv(filepath: str) -> None
        Export all signals to a CSV file.
    """

    STATUS_PENDING = 'pending'
    COLUMN_SIGNAL_ID = 'signal_id'
    COLUMN_STATUS = 'status'
    COLUMN_REASON = 'reason'

    def __init__(self):
        """
        Initializes the SignalDatabase.
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
            self.COLUMN_SIGNAL_ID: str,
            self.COLUMN_STATUS: str,
            'trade_id': int,
            self.COLUMN_REASON: str
        }
        self.signals: List[Dict[str, Any]] = []
        self.logger = setup_logger('signal_database')

    def _get_default_value(self, dtype: type) -> Any:
        """
        Get the default value for a given data type.

        Parameters
        ----------
        dtype : type
            The Python type to get a default value for.

        Returns
        -------
        Any
            The default value for the given type.
        """
        if dtype is int:
            return 0
        elif dtype is float:
            return 0.0
        elif dtype is str:
            return ''
        else:
            return None

    def add_signals(self, signals: List[Dict[str, Any]]) -> None:
        """
        Add new signals to the database.

        This method processes a list of signal dictionaries, assigns unique IDs,
        ensures all required fields are present, and adds them to the database.

        Parameters
        ----------
        signals : List[Dict[str, Any]]
            A list of dictionaries, each representing a signal to be added.

        Raises
        ------
        Exception
            If there's an error during the signal addition process.
        """"""
        Add new signals to the database.

        This method processes a list of signal dictionaries, assigns unique IDs,
        ensures all required fields are present, and adds them to the database.

        Parameters
        ----------
        signals : List[Dict[str, Any]]
            A list of dictionaries, each representing a signal to be added.

        Raises
        ------
        Exception
            If there's an error during the signal addition process.
        """
        try:
            if not signals:
                self.logger.warning("No signals to add.")
                return

            for signal in signals:
                # Assign unique ID to each signal
                signal[self.COLUMN_SIGNAL_ID] = str(uuid.uuid4())
                signal[self.COLUMN_STATUS] = self.STATUS_PENDING

                # Ensure all required columns are present in each signal with default values
                for column, dtype in self.columns.items():
                    if column not in signal:
                        signal[column] = self._get_default_value(dtype)

                # Convert values to the correct type
                for column, dtype in self.columns.items():
                    signal[column] = dtype(signal[column])

            # Add signals to the list
            self.signals.extend(signals)

            self.logger.info(f"Added {len(signals)} new signals to the database.")
            self.logger.debug(f"Added signals: {signals}")

        except Exception as e:
            self.logger.error(f"Error adding signals: {str(e)}")
            raise

    def get_signals(self, **filters) -> List[Dict[str, Any]]:
        """
        Retrieve signals based on provided filters.

        This method returns a list of signals that match the given filter criteria.

        Parameters
        ----------
        **filters : dict
            Keyword arguments representing field-value pairs to filter signals.

        Returns
        -------
        List[Dict[str, Any]]
            A list of signal dictionaries that match the filter criteria.

        Raises
        ------
        Exception
            If there's an error during the signal retrieval process.
        """
        try:
            if not filters:
                return self.signals.copy()

            filtered_signals = []
            for signal in self.signals:
                if all(signal.get(key) == value for key, value in filters.items()):
                    filtered_signals.append(signal.copy())

            self.logger.info(f"Retrieved {len(filtered_signals)} signals with applied filters.")
            return filtered_signals

        except Exception as e:
            self.logger.error(f"Error filtering signals: {str(e)}")
            raise

    def update_signal_status(self, signal_id: str, new_status: str, reason: Optional[str] = None) -> None:
        """
        Update the status of a specific signal.

        This method finds a signal by its ID and updates its status and optionally the reason.

        Parameters
        ----------
        signal_id : str
            The unique identifier of the signal to update.
        new_status : str
            The new status to set for the signal.
        reason : Optional[str], optional
            The reason for the status change (default is None).

        Raises
        ------
        ValueError
            If the signal with the given ID is not found.
        Exception
            If there's an error during the update process.
        """
        try:
            for signal in self.signals:
                if signal[self.COLUMN_SIGNAL_ID] == signal_id:
                    signal[self.COLUMN_STATUS] = new_status
                    if reason is not None:
                        signal[self.COLUMN_REASON] = reason
                    self.logger.info(f"Updated signal {signal_id} with new status '{new_status}'.")
                    return

            self.logger.error(f"Signal {signal_id} not found in the database.")
            raise ValueError(f"Signal {signal_id} not found.")

        except Exception as e:
            self.logger.error(f"Error updating signal status for signal {signal_id}: {str(e)}")
            raise
    
    def to_csv(self, filepath: str) -> None:
        """
        Export all signals to a CSV file.

        This method writes all signals in the database to a CSV file at the specified path.

        Parameters
        ----------
        filepath : str
            The file path where the CSV will be saved.

        Raises
        ------
        Exception
            If there's an error during the export process.
        """
        try:
            if not self.signals:
                self.logger.warning("No signals to export.")
                return

            with open(filepath, 'w', newline='') as csvfile:
                # Assuming all signals have the same keys, use the first signal to get the fieldnames
                fieldnames = list(self.signals[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for signal in self.signals:
                    writer.writerow(signal)

            self.logger.info(f"Exported {len(self.signals)} signals to {filepath}")

        except Exception as e:
            self.logger.error(f"Error exporting signals to CSV: {str(e)}")
            raise