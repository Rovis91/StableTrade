import pandas as pd
import logging
from typing import List, Dict, Optional, Union, Set

class SignalDatabase:
    def __init__(self, verbose: bool = False) -> None:
        """
        Initializes an empty signal database.
        Signals are stored in a pandas DataFrame for easy access and manipulation.

        Args:
            verbose (bool): If True, enables verbose DEBUG logging.
        """
        self.columns = ['signal_id', 'market', 'action', 'amount', 'timestamp', 
                        'status', 'stop_loss', 'take_profit', 'trailing_stop', 
                        'reason', 'close_reason']
        self.signals: pd.DataFrame = pd.DataFrame(columns=self.columns)
        self.signal_counter: int = 0
        self.logger = logging.getLogger(__name__)
        self.valid_statuses: Set[str] = {'pending', 'validated', 'canceled', 'executed'}
        self.verbose = verbose

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        self.logger.info("SignalDatabase initialized")
        self.logger.debug("Initial columns: %s", self.columns)

    def store_signals(self, new_signals: List[Dict[str, Union[str, float, int, None]]]) -> None:
        """
        Stores new signals in the database.

        Args:
            new_signals (List[Dict]): A list of signal dictionaries to be stored.

        Raises:
            ValueError: If new_signals is empty or contains invalid data.
        """
        try:
            if not new_signals:
                raise ValueError("No new signals provided for storage.")
            
            for signal in new_signals:
                self.signal_counter += 1
                signal['signal_id'] = self.signal_counter
                signal['status'] = 'pending'
                signal['close_reason'] = None

                # Validate signal structure
                missing_keys = set(self.columns) - set(signal.keys())
                if missing_keys:
                    raise ValueError(f"Signal {self.signal_counter} is missing required keys: {missing_keys}")

                if self.verbose:
                    self.logger.debug("Storing signal: %s", signal)

            new_df = pd.DataFrame(new_signals)
            self.signals = pd.concat([self.signals, new_df], ignore_index=True)
            self.logger.info("Stored %d new signals. Total signals: %d", len(new_signals), len(self.signals))
        except Exception as e:
            self.logger.error("Error storing signals: %s", str(e), exc_info=True)
            raise

    def get_signals(self, status: Optional[str] = None, timestamp: Optional[int] = None, market: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieves signals based on their status, timestamp, or market.
        
        Args:
            status (Optional[str]): The status of the signals to filter by.
            timestamp (Optional[int]): The timestamp of the signals to filter by.
            market (Optional[str]): The market (pair) for the signals to filter by.

        Returns:
            pd.DataFrame: A DataFrame of signals matching the query.

        Raises:
            ValueError: If an invalid status is provided.
        """
        try:
            if status and status not in self.valid_statuses:
                raise ValueError(f"Invalid status: {status}")

            signals_query = self.signals

            if status:
                signals_query = signals_query[signals_query['status'] == status]
            if timestamp:
                signals_query = signals_query[signals_query['timestamp'] == timestamp]
            if market:
                signals_query = signals_query[signals_query['market'] == market]

            self.logger.info("Retrieved %d signals matching criteria: status=%s, timestamp=%s, market=%s", 
                             len(signals_query), status, timestamp, market)

            if self.verbose:
                self.logger.debug("Query parameters: status=%s, timestamp=%s, market=%s", status, timestamp, market)
                self.logger.debug("Retrieved signals:\n%s", signals_query)

            return signals_query
        except Exception as e:
            self.logger.error("Error retrieving signals: %s", str(e), exc_info=True)
            return pd.DataFrame()

    def validate_signals(self, timestamp: int) -> None:
        """
        Validates all pending signals at a given timestamp.

        Args:
            timestamp (int): The timestamp of the signals to validate.

        Raises:
            ValueError: If no pending signals exist for the given timestamp.
        """
        try:
            mask = (self.signals['status'] == 'pending') & (self.signals['timestamp'] == timestamp)
            pending_count = mask.sum()

            if pending_count == 0:
                self.logger.warning("No pending signals found at timestamp %d", timestamp)
                return

            self.signals.loc[mask, 'status'] = 'validated'
            validated_count = mask.sum()

            self.logger.info("Validated %d signals at timestamp %d", validated_count, timestamp)
            
            if self.verbose:
                self.logger.debug("Signals before validation: %d, after validation: %d", 
                                  pending_count, validated_count)
        except Exception as e:
            self.logger.error("Error validating signals: %s", str(e), exc_info=True)
            raise

    def cancel_signals(self, timestamp: int, reason: str = 'not_executed') -> None:
        """
        Cancels all pending signals at a given timestamp and assigns a cancellation reason.

        Args:
            timestamp (int): The timestamp of the signals to cancel.
            reason (str): The reason for canceling the signal.
        """
        try:
            mask = (self.signals['status'] == 'pending') & (self.signals['timestamp'] == timestamp)
            pending_count = mask.sum()

            if pending_count == 0:
                self.logger.warning("No pending signals to cancel at timestamp %d", timestamp)
                return

            self.signals.loc[mask, 'status'] = 'canceled'
            self.signals.loc[mask, 'close_reason'] = reason
            canceled_count = mask.sum()

            self.logger.info("Canceled %d signals at timestamp %d with reason: %s", 
                             canceled_count, timestamp, reason)

            if self.verbose:
                self.logger.debug("Signals before cancellation: %d, after cancellation: %d", 
                                  pending_count, canceled_count)
        except Exception as e:
            self.logger.error("Error canceling signals: %s", str(e), exc_info=True)
            raise

    def expire_signals(self, timestamp: int) -> None:
        """
        Cancels all remaining unexecuted signals at the end of each iteration.

        Args:
            timestamp (int): The timestamp for which to expire signals.
        """
        try:
            self.cancel_signals(timestamp, reason='expired')
        except Exception as e:
            self.logger.error("Error expiring signals: %s", str(e), exc_info=True)
            raise

    def update_signal_status(self, signal_id: int, new_status: str, close_reason: Optional[str] = None) -> None:
        """
        Updates the status of a specific signal based on its signal ID.

        Args:
            signal_id (int): The unique ID of the signal.
            new_status (str): The new status of the signal.
            close_reason (Optional[str]): The reason for closing the signal, if applicable.

        Raises:
            ValueError: If the signal ID does not exist or the new status is invalid.
        """
        try:
            if new_status not in self.valid_statuses:
                raise ValueError(f"Invalid status '{new_status}' for signal {signal_id}")
            
            mask = self.signals['signal_id'] == signal_id
            if not mask.any():
                raise ValueError(f"No signal found with ID {signal_id}")
            
            old_status = self.signals.loc[mask, 'status'].iloc[0]
            self.signals.loc[mask, 'status'] = new_status
            if close_reason:
                self.signals.loc[mask, 'close_reason'] = close_reason

            self.logger.info("Updated signal %d from status %s to %s with reason: %s", 
                             signal_id, old_status, new_status, close_reason)

            if self.verbose:
                self.logger.debug("Signal %d details after update: %s", 
                                  signal_id, self.signals.loc[mask].to_dict('records')[0])
        except Exception as e:
            self.logger.error("Error updating signal status: %s", str(e), exc_info=True)
            raise

    def get_active_signals(self, timestamp: int) -> pd.DataFrame:
        """
        Retrieve all active signals (validated and pending) for a given timestamp.

        Args:
            timestamp (int): The timestamp for which to retrieve active signals.

        Returns:
            pd.DataFrame: A DataFrame of active signals.
        """
        try:
            active_signals = self.signals[(self.signals['status'].isin(['pending', 'validated'])) & 
                                          (self.signals['timestamp'] == timestamp)]
            self.logger.info("Retrieved %d active signals for timestamp %d", len(active_signals), timestamp)

            if self.verbose:
                self.logger.debug("Active signals:\n%s", active_signals)

            return active_signals
        except Exception as e:
            self.logger.error("Error retrieving active signals: %s", str(e), exc_info=True)
            return pd.DataFrame()

    def get_trade_signals(self, trade_id: int) -> pd.DataFrame:
        """
        Retrieve all signals related to a specific trade.

        Args:
            trade_id (int): The unique trade ID.

        Returns:
            pd.DataFrame: A DataFrame of signals linked to that trade ID.
        """
        try:
            trade_signals = self.signals[self.signals['trade_id'] == trade_id]
            self.logger.info("Retrieved %d signals for trade ID %d", len(trade_signals), trade_id)

            if self.verbose:
                self.logger.debug("Signals for trade ID %d:\n%s", trade_id, trade_signals)

            return trade_signals
        except Exception as e:
            self.logger.error("Error retrieving trade signals: %s", str(e), exc_info=True)
            return pd.DataFrame()

    def export_signals_to_csv(self, filename: str) -> None:
        """
        Export all signals to a CSV file.

        Args:
            filename (str): The name of the file to export to.
        """
        try:
            self.signals.to_csv(filename, index=False)
            self.logger.info("Exported %d signals to %s", len(self.signals), filename)

            if self.verbose:
                self.logger.debug("Export complete. File: %s, Signals: %d", filename, len(self.signals))
        except Exception as e:
            self.logger.error("Error exporting signals to CSV: %s", str(e), exc_info=True)
            raise

    def get_signal_stats(self) -> Dict[str, int]:
        """
        Get summary statistics of signals.

        Returns:
            Dict[str, int]: A dictionary containing signal statistics.
        """
        try:
            stats = {
                'total_signals': len(self.signals),
                'pending_signals': len(self.signals[self.signals['status'] == 'pending']),
                'validated_signals': len(self.signals[self.signals['status'] == 'validated']),
                'canceled_signals': len(self.signals[self.signals['status'] == 'canceled']),
                'executed_signals': len(self.signals[self.signals['status'] == 'executed'])
            }

            self.logger.info("Signal statistics: %s", stats)

            if self.verbose:
                self.logger.debug("Detailed signal counts by status:\n%s", 
                                  self.signals['status'].value_counts())

            return stats
        except Exception as e:
            self.logger.error("Error getting signal stats: %s", str(e), exc_info=True)
            return {}

    def check_data_integrity(self) -> None:
        """
        Perform data integrity checks on the signal database.
        """
        try:
            # Check for duplicate signal IDs
            duplicate_ids = self.signals[self.signals.duplicated(subset=['signal_id'])]
            if not duplicate_ids.empty:
                self.logger.warning("Found %d duplicate signal IDs", len(duplicate_ids))

            # Check for invalid statuses
            invalid_statuses = self.signals[~self.signals['status'].isin(self.valid_statuses)]
            if not invalid_statuses.empty:
                self.logger.warning("Found %d signals with invalid statuses", len(invalid_statuses))

            # Check for missing values in required fields
            required_fields = ['signal_id', 'market', 'action', 'amount', 'timestamp', 'status']
            missing_values = self.signals[required_fields].isnull().sum()
            if missing_values.sum() > 0:
                self.logger.warning("Found missing values in required fields:\n%s", missing_values[missing_values > 0])

            self.logger.info("Data integrity check completed")
        except Exception as e:
            self.logger.error("Error during data integrity check: %s", str(e), exc_info=True)

    def log_database_state(self) -> None:
        """
        Log the current state of the signal database.
        """
        try:
            total_signals = len(self.signals)
            status_counts = self.signals['status'].value_counts()
            latest_timestamp = self.signals['timestamp'].max() if not self.signals.empty else None

            self.logger.info("Current SignalDatabase State:")
            self.logger.info("Total signals: %d", total_signals)
            self.logger.info("Status counts:\n%s", status_counts)
            self.logger.info("Latest timestamp: %s", latest_timestamp)

            if self.verbose:
                self.logger.debug("Signals head:\n%s", self.signals.head())
                self.logger.debug("Signals tail:\n%s", self.signals.tail())
        except Exception as e:
            self.logger.error("Error logging database state: %s", str(e), exc_info=True)
