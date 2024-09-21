import pandas as pd
import logging

class SignalDatabase:
    def __init__(self):
        """
        Initializes an empty signal database.
        Signals are stored in a pandas DataFrame for easy access and manipulation.
        """
        # Define columns for the signals dataframe
        columns = ['signal_id', 'market', 'action', 'amount', 'timestamp', 
                   'status', 'stop_loss', 'take_profit', 'trailing_stop', 
                   'reason', 'close_reason']
        self.signals = pd.DataFrame(columns=columns)
        self.signal_counter = 0  # Unique ID counter for each signal

    def store_signals(self, new_signals):
        """
        Stores new signals in the database.

        Args:
            new_signals (list of dict): A list of signal dictionaries to be stored.
                                        Each signal dictionary should have the following keys:
                                        'market', 'action', 'amount', 'timestamp', 
                                        'stop_loss', 'take_profit', 'trailing_stop', and 'reason'
        """
        # Assign unique signal IDs
        for signal in new_signals:
            self.signal_counter += 1
            signal['signal_id'] = self.signal_counter
            signal['status'] = 'pending'  # Default status for new signals
            signal['close_reason'] = None  # Only used when closing signals

        # Append new signals to the dataframe
        self.signals = pd.concat([self.signals, pd.DataFrame(new_signals)], ignore_index=True)

    def get_signals(self, status=None, timestamp=None, market=None):
        """
        Retrieves signals based on their status, timestamp, or market.
        
        Args:
            status (str, optional): The status of the signals ('pending', 'validated', 'canceled').
                                    If None, retrieves signals with any status.
            timestamp (int, optional): The timestamp of the signals. If None, retrieves signals for all timestamps.
            market (str, optional): The market (pair) for the signals. If None, retrieves signals for all markets.

        Returns:
            pd.DataFrame: A dataframe of signals matching the query.
        """
        signals_query = self.signals

        if status:
            signals_query = signals_query[signals_query['status'] == status]
        if timestamp:
            signals_query = signals_query[signals_query['timestamp'] == timestamp]
        if market:
            signals_query = signals_query[signals_query['market'] == market]

        return signals_query

    def validate_signals(self, timestamp):
        """
        Validates all pending signals at a given timestamp.

        Args:
            timestamp (int): The timestamp of the signals to validate.
        """
        # Set status to 'validated' for pending signals at the given timestamp
        self.signals.loc[(self.signals['status'] == 'pending') & 
                         (self.signals['timestamp'] == timestamp), 'status'] = 'validated'
        logging.info(f"Validated signals at timestamp {timestamp}")

    def cancel_signals(self, timestamp, reason='not_executed'):
        """
        Cancels all pending signals at a given timestamp and assigns a cancellation reason.

        Args:
            timestamp (int): The timestamp of the signals to cancel.
            reason (str): The reason for canceling the signal.
        """
        # Set status to 'canceled' for pending signals at the given timestamp
        self.signals.loc[(self.signals['status'] == 'pending') & 
                         (self.signals['timestamp'] == timestamp), 'status'] = 'canceled'
        self.signals.loc[(self.signals['status'] == 'canceled') & 
                         (self.signals['timestamp'] == timestamp), 'close_reason'] = reason
        logging.info(f"Canceled signals at timestamp {timestamp} with reason: {reason}")

    def expire_signals(self, timestamp):
        """
        Cancels all remaining unexecuted signals at the end of each iteration.
        This function can be called at the end of the backtest loop to ensure no signals are left pending.

        Args:
            timestamp (int): The timestamp for which to expire signals.
        """
        self.cancel_signals(timestamp, reason='expired')
        logging.info(f"Expired signals at timestamp {timestamp}")

    def update_signal_status(self, signal_id, new_status, close_reason=None):
        """
        Updates the status of a specific signal based on its signal ID.

        Args:
            signal_id (int): The unique ID of the signal.
            new_status (str): The new status of the signal ('validated', 'canceled', etc.).
            close_reason (str, optional): The reason for closing the signal if applicable.
        """
        self.signals.loc[self.signals['signal_id'] == signal_id, 'status'] = new_status
        if close_reason:
            self.signals.loc[self.signals['signal_id'] == signal_id, 'close_reason'] = close_reason
        logging.info(f"Updated signal {signal_id} to status {new_status} with reason: {close_reason}")

    def get_active_signals(self, timestamp):
        """
        Retrieve all active signals (validated and pending) for a given timestamp.

        Args:
            timestamp (int): The timestamp for which to retrieve active signals.

        Returns:
            pd.DataFrame: A dataframe of active signals.
        """
        active_signals = self.signals[(self.signals['status'].isin(['pending', 'validated'])) & 
                                      (self.signals['timestamp'] == timestamp)]
        return active_signals

    def get_trade_signals(self, trade_id):
        """
        Retrieve all signals related to a specific trade.

        Args:
            trade_id (int): The unique trade ID.

        Returns:
            pd.DataFrame: A dataframe of signals linked to that trade ID.
        """
        return self.signals[self.signals['trade_id'] == trade_id]
