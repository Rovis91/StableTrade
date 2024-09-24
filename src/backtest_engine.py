import logging
import pandas as pd
from src.data_preprocessor import DataPreprocessor
from src.metrics import MetricsModule

class BacktestEngine:
    def __init__(self, assets, strategies, portfolio, trade_manager, base_currency="USD", slippage=0.0, latency=0):
        """
        Initialize the backtest engine.

        Args:
            assets (dict): A dictionary of asset names and their corresponding CSV file paths.
            strategies (dict): A dictionary of asset names and their corresponding strategy instances.
            portfolio (Portfolio): Portfolio instance for managing capital and trades.
            trade_manager (TradeManager): TradeManager instance to execute trades.
            base_currency (str): The base currency to be used in the portfolio.
            slippage (float): Slippage to be applied to trades.
            latency (int): Simulated latency in milliseconds.
        """
        self.assets = assets
        self.strategies = strategies
        self.portfolio = portfolio
        self.trade_manager = trade_manager
        self.base_currency = base_currency
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
                continue

        # Create unified timestamp list based on all assets
        if self.market_data:
            self.unified_timestamps = sorted(set(ts for asset in self.market_data.values() for ts in asset.index))
            self.logger.info(f"Unified timestamps generated with {len(self.unified_timestamps)} entries.")
        else:
            self.logger.error("No market data available after preprocessing.")

    def run_backtest(self):
        """Run the backtest on all assets, processing each timestamp sequentially."""
        self.logger.info("Starting backtest...")

        if not self.unified_timestamps:
            self.logger.error("Backtest cannot proceed. No timestamps available.")
            return

        for timestamp in self.unified_timestamps:
            signals = []
            self.logger.debug(f"Processing timestamp {timestamp}")

            market_prices = {}
            for asset_name, data in self.market_data.items():
                if timestamp in data.index:
                    row_data = data.loc[timestamp]
                    strategy = self.strategies[asset_name]

                    # Add current price to market prices dictionary
                    market_prices[asset_name] = row_data['close']

                    # Log the active trades for this asset before processing
                    active_trades = self.trade_manager.get_trade(status='open')
                    self.logger.debug(f"Active trades for {asset_name} at timestamp {timestamp}: {active_trades}")

                    # Update trailing stops if necessary
                    self._update_trailing_stops(asset_name, row_data['close'])

                    # Check stop-loss and take-profit conditions
                    self._check_stop_loss_take_profit(asset_name, row_data['close'], timestamp)
                    
                    # Get portfolio value and cash before generating signals
                    portfolio_value = self.portfolio.get_total_value(market_prices={asset_name: row_data['close']})
                    portfolio_cash = self.portfolio.holdings.get('cash', 0)

                    # Execute the strategy to generate signals
                    signal = strategy.generate_signal(row_data, active_trades, portfolio_value, portfolio_cash)
                    self.logger.debug(f"Signal generated for {asset_name}: {signal}")

                    if signal:
                        signals.append(signal)

            # Process signals only if available
            if signals:
                self.logger.debug(f"Processing signals at {timestamp}: {signals}")
                self.portfolio.process_signals(signals=signals, market_prices=market_prices, timestamp=timestamp)

        # Log final summary after backtest
        self.log_final_summary()

    def log_final_summary(self):
        """Log a final summary of trades and portfolio performance after the backtest completes."""
        
        all_trades = self.trade_manager.get_trade()

        if not all_trades:
            self.logger.error("No trades executed during the backtest.")
            return

        # Ensure market_data is properly structured and passed for portfolio evaluation
        if not self.market_data:
            self.logger.error("Market data not available for portfolio evaluation.")
            return

        # Ensure that the timestamps in portfolio_history align with market_data
        if not self.portfolio.history:
            self.logger.error("Portfolio history not available for trade summary.")
            return

        try:
            metrics_module = MetricsModule(base_currency=self.base_currency)
            trade_summary = metrics_module.calculate_trade_summary(
                trades=all_trades,
                portfolio_history=self.portfolio.history,
                market_data=self.market_data,
                slippage=self.slippage
            )

            # Log summary statistics after metrics are calculated
            self.logger.info("Backtest completed.")
            self.logger.info(f"Total trades executed: {trade_summary['total_trades']}")
            self.logger.info(f"Total profit/loss: {trade_summary['total_profit']}")
            self.logger.info(f"Total fees paid: {trade_summary['total_fees']}")
            self.logger.info(f"Number of winning trades: {trade_summary['winning_trades']}")
            self.logger.info(f"Number of losing trades: {trade_summary['losing_trades']}")
            self.logger.info(f"Trades closed due to stop loss: {trade_summary['stop_loss_trades']}")
            self.logger.info(f"Trades closed due to take profit: {trade_summary['take_profit_trades']}")
            self.logger.info(f"Average holding time: {trade_summary['average_holding_time']:.2f} minutes")
            self.logger.info(f"Final portfolio value: {trade_summary['final_portfolio_value']}")
            self.logger.info(f"Total percent increase: {trade_summary['total_percent_increase']:.2f}%")
            self.logger.info(f"Average percent return per month: {trade_summary['avg_percent_per_month']:.2f}%")

        except Exception as e:
            self.logger.error(f"Error calculating trade summary: {e}", exc_info=True)

    def _update_trailing_stops(self, asset_name, current_price):
        """Update trailing stop for open trades."""
        self.logger.debug(f"Updating trailing stops for {asset_name} with current price: {current_price}")

        for trade in self.trade_manager.get_trade(status='open'):
            market_type = trade.get('market_type', 'spot')

            if trade['asset_name'] == asset_name and trade['trailing_stop'] is not None:
                previous_stop = trade['stop_loss']
                
                # For long trades or buy trades, adjust trailing stop only if the current price is higher
                if (market_type == 'futures' and trade['direction'] == 'long') or (market_type == 'spot' and trade['direction'] == 'buy'):
                    new_stop_loss = max(trade['stop_loss'], current_price * (1 - trade['trailing_stop'] / 100))
                # For short trades or sell trades, adjust trailing stop only if the current price is lower
                elif (market_type == 'futures' and trade['direction'] == 'short') or (market_type == 'spot' and trade['direction'] == 'sell'):
                    new_stop_loss = min(trade['stop_loss'], current_price * (1 + trade['trailing_stop'] / 100))
                else:
                    self.logger.debug(f"No update required for trailing stop on trade {trade['id']}.")

                if new_stop_loss != previous_stop:
                    self.trade_manager.modify_trade_parameters(trade_id=trade['id'], stop_loss=new_stop_loss)
                    self.logger.debug(f"Trailing stop updated for trade {trade['id']} from {previous_stop} to {new_stop_loss}")
                else:
                    self.logger.debug(f"No change in trailing stop for trade {trade['id']}, remains at {previous_stop}")


    def _check_stop_loss_take_profit(self, asset_name, current_price, timestamp):
        """
        Check stop loss and take profit conditions for open trades and close them if conditions are met.
        """
        for trade in self.trade_manager.get_trade(status='open'):
            if trade['asset_name'] == asset_name:
                # Check stop loss
                if trade['stop_loss'] is not None and current_price <= trade['stop_loss']:
                    self._trigger_stop_loss(trade, current_price, timestamp)

                # Check take profit
                if trade['take_profit'] is not None and current_price >= trade['take_profit']:
                    self._trigger_take_profit(trade, current_price, timestamp)

                self.logger.debug(f"Stop-loss/take-profit checked for trade {trade['id']} at {timestamp}")

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
