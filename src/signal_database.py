import pandas as pd
from typing import Dict, List, Any, Optional
import uuid
from src.logger import setup_logger


class SignalDatabase:
    """
    SignalDatabase class for storing, retrieving, and updating trading signals.

    The signals are stored in a Pandas DataFrame with predefined columns, including 
    attributes such as timestamp, asset name, action, stop loss, take profit, and status.

    Attributes:
        signals (pd.DataFrame): DataFrame to store signal information.
        logger (logging.Logger): Logger instance for logging database operations.
    """

    STATUS_PENDING = 'pending'
    COLUMN_SIGNAL_ID = 'signal_id'
    COLUMN_STATUS = 'status'
    COLUMN_REASON = 'reason'

    def __init__(self, logger: Optional[Any] = None):
        """
        Initialize the signal database with predefined columns.

        Args:
            logger (Optional[logging.Logger]): A logger instance for logging database operations.
                                               If not provided, a default logger will be set up.
        """
        self.signals = pd.DataFrame(columns=[
            'timestamp', 'asset_name', 'action', 'amount', 'price',
            'stop_loss', 'take_profit', 'trailing_stop', self.COLUMN_STATUS,
            'trade_id', self.COLUMN_SIGNAL_ID, self.COLUMN_REASON
        ])
        self.logger = logger if logger else setup_logger('signal_database')

    def add_signals(self, signals: List[Dict[str, Any]]) -> None:
        """
        Add one or more signals to the database.

        Each signal should be a dictionary with keys matching the expected columns 
        in the database, such as 'timestamp', 'asset_name', 'action', etc. The status 
        of each new signal is automatically set to 'pending'.

        Args:
            signals (List[Dict[str, Any]]): A list of dictionaries, where each dictionary 
                                            represents a signal to be added.

        Example:
            signal = {'timestamp': 1625140800, 'asset_name': 'BTC', 'action': 'buy', 'amount': 1.0, 'price': 40000}
            signal_database.add_signals([signal])
        """
        new_signals = []
        for signal in signals:
            # Assign a unique signal ID using uuid and set the status to 'pending'
            signal[self.COLUMN_STATUS] = self.STATUS_PENDING
            signal[self.COLUMN_SIGNAL_ID] = str(uuid.uuid4())
            new_signals.append(signal)

        if new_signals:
            # Concatenate the new signals into the existing DataFrame
            self.signals = pd.concat([self.signals, pd.DataFrame(new_signals)], ignore_index=True)

        self.logger.info(f"Added {len(signals)} new signals to the database.")

    def get_signals(self, 
                    timestamp: Optional[int] = None, 
                    asset_name: Optional[str] = None, 
                    status: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve signals based on optional filters.

        Args:
            timestamp (Optional[int]): Filter signals by timestamp.
            asset_name (Optional[str]): Filter signals by asset name.
            status (Optional[str]): Filter signals by status.

        Returns:
            pd.DataFrame: DataFrame containing the filtered signals.
        """
        filtered_signals = self.signals

        if timestamp is not None:
            filtered_signals = filtered_signals[filtered_signals['timestamp'] == timestamp]

        if asset_name is not None:
            filtered_signals = filtered_signals[filtered_signals['asset_name'] == asset_name]

        if status is not None:
            filtered_signals = filtered_signals[filtered_signals[self.COLUMN_STATUS] == status]

        self.logger.debug(f"Retrieved {len(filtered_signals)} signals with applied filters.")
        return filtered_signals

    def update_signal_status(self, signal_id: str, new_status: str, reason: Optional[str] = None) -> None:
        """
        Update the status of a specific signal.

        Args:
            signal_id (str): The unique ID of the signal to update.
            new_status (str): The new status to set for the signal.
            reason (Optional[str]): The reason for the status update, if any.
        """
        # Update status and reason in one chained `loc` call to reduce overhead
        update_condition = self.signals[self.COLUMN_SIGNAL_ID] == signal_id
        self.signals.loc[update_condition, [self.COLUMN_STATUS, self.COLUMN_REASON]] = [new_status, reason]

        self.logger.info(f"Updated signal {signal_id} status to '{new_status}'.")
