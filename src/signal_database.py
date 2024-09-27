import logging
import pandas as pd
from typing import Dict, List, Any, Optional


class SignalDatabase:
    def __init__(self):
        """
        Initialize the signal database with predefined columns.
        """
        self.signals = pd.DataFrame(columns=[
            'timestamp', 'asset_name', 'action', 'amount', 'price',
            'stop_loss', 'take_profit', 'trailing_stop', 'status',
            'trade_id', 'signal_id', 'reason'
        ])
        self.logger = logging.getLogger(__name__)

    def add_signals(self, signals: List[Dict[str, Any]]) -> None:
        """
        Add one or more signals to the database.

        Args:
            signals (List[Dict[str, Any]]): A list of signal dictionaries to be added.
        """
        new_signals = []
        for signal in signals:
            signal['status'] = 'pending'
            signal['signal_id'] = len(self.signals) + 1
            new_signals.append(signal)
        
        if new_signals:
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
            filtered_signals = filtered_signals[filtered_signals['status'] == status]

        self.logger.debug(f"Retrieved {len(filtered_signals)} signals with applied filters.")
        return filtered_signals

    def update_signal_status(self, signal_id: int, new_status: str, reason: Optional[str] = None) -> None:
        """
        Update the status of a specific signal.

        Args:
            signal_id (int): The ID of the signal to update.
            new_status (str): The new status to set for the signal.
            reason (Optional[str]): The reason for the status update, if any.
        """
        self.signals.loc[self.signals['signal_id'] == signal_id, 'status'] = new_status
        if reason is not None:
            self.signals.loc[self.signals['signal_id'] == signal_id, 'reason'] = reason
        
        self.logger.info(f"Updated signal {signal_id} status to '{new_status}'.")