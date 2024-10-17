import pandas as pd
from typing import Dict, List, Any, Optional
import uuid
from src.logger import setup_logger


class SignalDatabase:
    """
    A database for managing trading signals.

    Attributes
    ----------
    STATUS_PENDING : str
        Default status for newly added signals.
    COLUMN_SIGNAL_ID : str
        Column name for signal ID.
    COLUMN_STATUS : str
        Column name for status.
    COLUMN_REASON : str
        Column name for reason of status change.

    Methods
    -------
    add_signals(signals: List[Dict[str, Any]]) -> None
        Adds new signals to the database.
    get_signals(**filters) -> pd.DataFrame
        Retrieves signals based on the provided filters.
    update_signal_status(signal_id: str, new_status: str, reason: Optional[str] = None) -> None
        Updates the status of a signal given its ID.
    """

    STATUS_PENDING = 'pending'
    COLUMN_SIGNAL_ID = 'signal_id'
    COLUMN_STATUS = 'status'
    COLUMN_REASON = 'reason'

    def __init__(self, buffer_size: int = 1):  # Buffer size set to 1 for immediate flush
        """
        Initializes the SignalDatabase.

        Parameters
        ----------
        buffer_size : int, optional
            Number of signals to store in memory before flushing to the main DataFrame (default is 1 for immediate flushing).
        """
        self.columns = {
            'timestamp': 'int64',
            'asset_name': 'str',
            'action': 'str',
            'amount': 'float64',
            'price': 'float64',
            'stop_loss': 'float64',
            'take_profit': 'float64',
            'trailing_stop': 'float64',
            self.COLUMN_SIGNAL_ID: 'str',
            self.COLUMN_STATUS: 'str',
            'trade_id': 'int64',
            self.COLUMN_REASON: 'str'
        }
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_size = buffer_size
        self.signals = pd.DataFrame(columns=self.columns.keys()).astype(self.columns)
        self.logger = setup_logger('signal_database')


    def add_signals(self, signals: List[Dict[str, Any]]) -> None:
        """
        Adds new signals to the database.

        Parameters
        ----------
        signals : List[Dict[str, Any]]
            A list of signal dictionaries to be added.
        """
        try:
            if not signals:
                self.logger.warning("No signals to add.")
                return

            for signal in signals:
                # Assign unique ID to each signal
                signal[self.COLUMN_SIGNAL_ID] = str(uuid.uuid4())

                # Ensure all required columns are present in each signal with default values
                for column, dtype in self.columns.items():
                    if column not in signal:
                        signal[column] = self._get_default_value(dtype)

            # Add to buffer and flush immediately
            self.buffer.extend(signals)
            self.flush_buffer()

            # Log added signals
            self.logger.debug(f"Added signals: {signals}")

        except Exception as e:
            self.logger.error(f"Error adding signals: {str(e)}")
            raise


    def flush_buffer(self) -> None:
        """
        Flushes the buffer to the main signals DataFrame.
        """
        try:
            if not self.buffer:
                return

            new_df = pd.DataFrame(self.buffer)
            self.buffer = []

            # Ensure all required columns are present
            missing_columns = set(self.columns.keys()) - set(new_df.columns)
            for column in missing_columns:
                new_df[column] = self._get_default_value(self.columns[column])

            # Assign status to pending for all new signals
            new_df[self.COLUMN_STATUS] = self.STATUS_PENDING

            # Convert columns to specified data types
            new_df = new_df.astype(self.columns)

            # Concatenate with the original signals DataFrame without changing the index
            self.signals = pd.concat([self.signals, new_df], ignore_index=True)
            self.logger.info(f"Flushed {len(new_df)} signals to main DataFrame.")

        except Exception as e:
            self.logger.error(f"Error flushing buffer: {str(e)}")
            raise

    def get_signals(self, **filters) -> pd.DataFrame:
        """
        Retrieves signals based on the provided filters.

        Parameters
        ----------
        filters : dict
            Column-value pairs to filter signals.

        Returns
        -------
        pd.DataFrame
            A DataFrame of signals that match the provided filters.
        """
        try:
            if not filters:
                return self.signals.copy()

            query_str = " & ".join([f"{col} == @value" for col, value in filters.items() if col in self.signals.columns])
            return self.signals.query(query_str).copy()

        except Exception as e:
            self.logger.error(f"Error filtering signals: {str(e)}")
            raise

    def update_signal_status(self, signal_id: str, new_status: str, reason: Optional[str] = None) -> None:
        """
        Updates the status of a signal given its ID.

        Parameters
        ----------
        signal_id : str
            The unique identifier of the signal to be updated.
        new_status : str
            The new status to be assigned to the signal.
        reason : Optional[str]
            The reason for the status change, if any.

        Raises
        ------
        ValueError
            If the signal ID is not found in the database.
        """
        try:
            # Log the state of signals before attempting update
            self.logger.debug(f"Signals in database before update: {self.signals[[self.COLUMN_SIGNAL_ID, self.COLUMN_STATUS]].to_dict(orient='records')}")

            # Use a boolean mask to find the rows with the matching signal_id
            mask = self.signals[self.COLUMN_SIGNAL_ID] == signal_id

            # Check if the signal_id exists in the DataFrame
            if not mask.any():
                self.logger.error(f"Signal {signal_id} not found in the database.")
                raise ValueError(f"Signal {signal_id} not found.")

            # Update the status and reason for the matching signal
            self.signals.loc[mask, self.COLUMN_STATUS] = new_status
            if reason is not None:
                self.signals.loc[mask, self.COLUMN_REASON] = reason

            self.logger.info(f"Updated signal {signal_id} with new status '{new_status}'.")

        except Exception as e:
            self.logger.error(f"Error updating signal status for signal {signal_id}: {str(e)}")
            raise




    def _get_default_value(self, dtype: str) -> Any:
        """
        Returns the default value based on the provided data type.

        Parameters
        ----------
        dtype : str
            The data type for which to return a default value.

        Returns
        -------
        Any
            The default value for the given data type.
        """
        if 'int' in dtype:
            return 0
        elif 'float' in dtype:
            return 0.0
        elif 'str' in dtype:
            return ''
        else:
            return None
