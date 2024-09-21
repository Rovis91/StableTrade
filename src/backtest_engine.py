import logging
import pandas as pd
from src.data_preprocessor import DataPreprocessor

class BacktestEngine:
    def __init__(self, assets, strategies, portfolio, trade_manager, slippage=0.0, latency=0):
        """
        Initialize the backtest engine.

        Args:
            assets (dict): A dictionary of asset names and their corresponding CSV file paths.
            strategies (dict): A dictionary of asset names and their corresponding strategy instances.
            portfolio (Portfolio): Portfolio instance for managing capital and trades.
            trade_manager (TradeManager): TradeManager instance to execute trades.
            slippage (float): Slippage to be applied to trades.
            latency (int): Simulated latency in milliseconds.
        """
        self.assets = assets
        self.strategies = strategies
        self.portfolio = portfolio
        self.trade_manager = trade_manager
        self.slippage = slippage
        self.latency = latency
        self.market_data = {}
        self.unified_timestamps = []
        self.logger = logging.getLogger(__name__)
    
    def preprocess_data(self):
        """Preprocess the data for all assets and prepare the unified timestamp list."""
        self.logger.info("Preprocessing data for all assets...")

        for asset_name, csv_path in self.assets.items():
            strategy = self.strategies[asset_name]
            required_indicators = strategy.get_required_indicators()

            # Initialize the DataPreprocessor with the input CSV path
            preprocessor = DataPreprocessor(csv_path)

            # Preprocess data (this will load existing data if indicators match)
            try:
                data = preprocessor.preprocess_data(required_indicators)
                self.market_data[asset_name] = data
                self.logger.info(f"Data preprocessed for asset: {asset_name}")
            except Exception as e:
                self.logger.error(f"Error preprocessing data for {asset_name}: {e}")

        # Create unified timestamp list based on all assets
        self.unified_timestamps = sorted(set(ts for asset in self.market_data.values() for ts in asset.index))
        self.logger.info(f"Unified timestamps generated with {len(self.unified_timestamps)} entries.")

    def run_backtest(self):
        """Run the backtest on all assets, processing each timestamp sequentially."""
        self.logger.info("Starting backtest...")

        for timestamp in self.unified_timestamps:
            # self.logger.debug(f"Processing timestamp: {timestamp}")

            for asset_name, data in self.market_data.items():
                if timestamp in data.index:
                    row_data = data.loc[timestamp]
                    strategy = self.strategies[asset_name]

                    # Get active trades for this asset
                    active_trades = self.trade_manager.get_trade(status='open')

                    # Execute the strategy to generate signals
                    signals = strategy.generate_signals(row_data, active_trades)

                    # Log generated signals
                    if signals:
                        self.logger.info(f"Generated signals at {timestamp} for {asset_name}: {signals}")
                    
                    # Update trailing stop before checking stop loss or take profit
                    self._update_trailing_stops(asset_name, row_data['close'])

                    # Check stop-loss and take-profit for active trades
                    self._check_stop_loss_take_profit(asset_name, row_data['close'], timestamp)

                    # Pass signals to portfolio for validation and execution
                    self.portfolio.process_signals(signals=signals, market_prices={asset_name: row_data['close']}, timestamp=timestamp)

        # Log final summary after the backtest
        self.log_final_summary()

    def log_final_summary(self):
        """Log a final summary of trades after the backtest completes."""
        all_trades = self.trade_manager.get_trade()
        open_trades = self.trade_manager.get_trade(status='open')
        closed_trades = self.trade_manager.get_trade(status='closed')

        self.logger.info("Backtest completed.")
        self.logger.info(f"Total trades executed: {len(all_trades)}")
        self.logger.info(f"Open trades: {len(open_trades)}")
        self.logger.info(f"Closed trades: {len(closed_trades)}")

        if open_trades:
            self.logger.info(f"Open trades details: {open_trades}")
        
        if closed_trades:
            self.logger.info(f"Closed trades details: {closed_trades}")

    def _update_trailing_stops(self, asset_name, current_price):
        """Update trailing stop for open trades."""
        for trade in self.trade_manager.get_trade(status='open'):
            if trade['asset_name'] == asset_name and trade['trailing_stop'] is not None:
                # For long trades, adjust trailing stop only if the current price is higher
                if trade['amount'] > 0:
                    new_stop_loss = max(trade['stop_loss'], current_price * (1 - trade['trailing_stop'] / 100))
                    self.trade_manager.modify_trade_parameters(trade_id=trade['id'], stop_loss=new_stop_loss)
                
                # For short trades, adjust trailing stop only if the current price is lower
                if trade['amount'] < 0:
                    new_stop_loss = min(trade['stop_loss'], current_price * (1 + trade['trailing_stop'] / 100))
                    self.trade_manager.modify_trade_parameters(trade_id=trade['id'], stop_loss=new_stop_loss)

    def _check_stop_loss_take_profit(self, asset_name, current_price, timestamp):
        """
        Check stop loss and take profit conditions for open trades and close them if conditions are met.
        
        Args:
            asset_name (str): The asset being processed (e.g., 'BTCUSD').
            current_price (float): The current market price of the asset.
            timestamp (int): The current timestamp being processed.
        """
        for trade in self.trade_manager.get_trade(status='open'):
            if trade['asset_name'] == asset_name:
                # Check stop loss
                if trade['stop_loss'] is not None and current_price <= trade['stop_loss']:
                    self._trigger_stop_loss(trade, current_price, timestamp)
                
                # Check take profit
                if trade['take_profit'] is not None and current_price >= trade['take_profit']:
                    self._trigger_take_profit(trade, current_price, timestamp)

    def _trigger_stop_loss(self, trade, current_price, timestamp):
        """Trigger stop loss and close the trade."""
        self.logger.info(f"Stop loss triggered for trade {trade['id']} at {timestamp}.")
        exit_fee = self.portfolio.get_fee(trade['asset_name'], 'exit')
        self.trade_manager.close_trade(
            trade_id=trade['id'],
            exit_price=current_price,
            exit_timestamp=timestamp,
            exit_fee=exit_fee,
            exit_reason="stop_loss"
        )

    def _trigger_take_profit(self, trade, current_price, timestamp):
        """Trigger take profit and close the trade."""
        self.logger.info(f"Take profit triggered for trade {trade['id']} at {timestamp}.")
        exit_fee = self.portfolio.get_fee(trade['asset_name'], 'exit')
        self.trade_manager.close_trade(
            trade_id=trade['id'],
            exit_price=current_price,
            exit_timestamp=timestamp,
            exit_fee=exit_fee,
            exit_reason="take_profit"
        )
