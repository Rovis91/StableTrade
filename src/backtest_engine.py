import logging
import pandas as pd
from typing import Dict, List, Any
from src.data_preprocessor import DataPreprocessor
from src.metrics import MetricsModule

class BacktestEngine:
    """
    A class to run backtests on trading strategies for multiple assets.

    This engine preprocesses data, runs strategies, manages trades, and calculates performance metrics.
    """

    def __init__(self, assets: Dict[str, str], strategies: Dict[str, Any], portfolio, trade_manager, 
                 base_currency: str = "USD", slippage: float = 0.0, latency: int = 0, 
                 metrics: MetricsModule = None, signal_database=None, debug_mode: bool = False):
        """
        Initialize the BacktestEngine with the given parameters.
        """
        self.logger = logging.getLogger(__name__)
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.logger.setLevel(logging.DEBUG)
        
        self._validate_inputs(assets, strategies, portfolio, trade_manager, base_currency, slippage, latency)
        
        self.assets = assets
        self.strategies = strategies
        self.portfolio = portfolio
        self.trade_manager = trade_manager
        self.base_currency = base_currency
        self.slippage = slippage
        self.latency = latency
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.unified_timestamps: List[int] = []
        self.metrics = metrics if metrics else MetricsModule(base_currency=base_currency)
        self.signal_database = signal_database

        self.logger.info("BacktestEngine initialized with %d assets and %d strategies", 
                         len(assets), len(strategies))
        if self.debug_mode:
            self.logger.debug("Debug mode is ON. Verbose logging enabled.")
            self.logger.debug("Initialized attributes: %s", vars(self))

    def _validate_inputs(self, assets, strategies, portfolio, trade_manager, base_currency, slippage, latency):
        """Validate input parameters for the BacktestEngine."""
        try:
            assert isinstance(assets, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in assets.items()), \
                "Assets must be a dictionary with string keys and values."
            assert isinstance(strategies, dict) and all(isinstance(k, str) for k in strategies.keys()), \
                "Strategies must be a dictionary with string keys."
            assert hasattr(portfolio, 'process_signals') and callable(getattr(portfolio, 'process_signals')), \
                "Portfolio must have a 'process_signals' method."
            assert hasattr(trade_manager, 'get_trade') and callable(getattr(trade_manager, 'get_trade')), \
                "TradeManager must have a 'get_trade' method."
            assert isinstance(base_currency, str), "Base currency must be a string."
            assert isinstance(slippage, (int, float)) and slippage >= 0, "Slippage must be a non-negative number."
            assert isinstance(latency, int) and latency >= 0, "Latency must be a non-negative integer."
        except AssertionError as e:
            self.logger.error("Input validation failed: %s", str(e))
            raise ValueError(str(e))

        self.logger.debug("Input validation completed successfully")

    def preprocess_data(self) -> None:
        """Preprocess the data for all assets and prepare the unified timestamp list."""
        self.logger.info("Starting data preprocessing for all assets...")

        for asset_name, csv_path in self.assets.items():
            self.logger.debug("Preprocessing data for asset: %s", asset_name)
            strategy = self.strategies[asset_name]
            required_indicators = strategy.get_required_indicators()

            preprocessor = DataPreprocessor(csv_path)

            try:
                data = preprocessor.preprocess_data(required_indicators)
                self.market_data[asset_name] = data
                self.logger.info("Data preprocessed successfully for asset: %s", asset_name)
                if self.debug_mode:
                    self.logger.debug("Preprocessed data shape for %s: %s", asset_name, data.shape)
            except Exception as e:
                self.logger.error("Error preprocessing data for %s: %s", asset_name, str(e), exc_info=True)
                raise

        if len(self.market_data) != len(self.assets):
            self.logger.error("Not all assets were successfully preprocessed.")
            raise ValueError("Preprocessing failed for some assets.")

        self.unified_timestamps = sorted(set(ts for asset in self.market_data.values() for ts in asset.index))
        self.logger.info("Unified timestamps generated with %d entries.", len(self.unified_timestamps))
        if self.debug_mode:
            self.logger.debug("First 5 timestamps: %s", self.unified_timestamps[:5])

    def run_backtest(self) -> None:
        """Run the backtest across all unified timestamps."""
        self.logger.info("Starting backtest...")
        if not self.unified_timestamps:
            self.logger.error("No timestamps available for backtesting.")
            raise ValueError("No timestamps available. Ensure data preprocessing was successful.")

        total_timestamps = len(self.unified_timestamps)
        self.logger.info("Total timestamps for backtest: %d", total_timestamps)

        self._log_initial_state()

        for i, timestamp in enumerate(self.unified_timestamps):
            if i % (total_timestamps // 10) == 0:
                self.logger.info("Backtest progress: %.1f%%", (i / total_timestamps * 100))
            
            if self.debug_mode:
                self.logger.debug("Processing timestamp: %d", timestamp)

            try:
                self._process_timestamp(timestamp)

            except Exception as e:
                self.logger.error("Error during backtest at timestamp %d: %s", timestamp, str(e), exc_info=True)

        self.log_final_summary()

    def _process_timestamp(self, timestamp: int) -> None:
        """Process a single timestamp in the backtest."""
        all_signals = []
        market_prices = {}

        for asset_name, data in self.market_data.items():
            if timestamp in data.index:
                row_data = data.loc[timestamp]
                current_price = row_data['close']
                market_prices[asset_name] = current_price

                # 1. Check stop losses and take profits
                stop_loss_take_profit_signals = self._check_stop_loss_take_profit(asset_name, current_price, timestamp)
                all_signals.extend(stop_loss_take_profit_signals)

                # 2. Update trailing stops
                self._update_trailing_stops(asset_name, current_price)

                # 3. Generate strategy signals
                strategy_signals = self._generate_strategy_signals(asset_name, row_data, current_price, timestamp)
                all_signals.extend(strategy_signals)

        # 4. Process all signals after checking all assets
        if all_signals:
            self.logger.debug("Processing %d signals at timestamp %d", len(all_signals), timestamp)
            self._process_signals(all_signals, market_prices, timestamp)

    def _check_stop_loss_take_profit(self, asset_name: str, current_price: float, timestamp: int) -> List[Dict[str, Any]]:
        """Check for stop loss and take profit conditions for open trades of a specific asset."""
        close_signals = []
        active_trades = [trade for trade in self.trade_manager.get_trade(status='open') 
                         if trade['asset_name'] == asset_name]
        
        for trade in active_trades:
            if trade['stop_loss'] is not None and current_price <= trade['stop_loss']:
                self.logger.info("Stop loss triggered for trade %d at price %.8f", trade['id'], current_price)
                close_signals.append(self._generate_close_signal(trade, current_price, timestamp, "stop_loss"))
            elif trade['take_profit'] is not None and current_price >= trade['take_profit']:
                self.logger.info("Take profit triggered for trade %d at price %.8f", trade['id'], current_price)
                close_signals.append(self._generate_close_signal(trade, current_price, timestamp, "take_profit"))

        if self.debug_mode and close_signals:
            self.logger.debug("Generated %d close signals for asset %s", len(close_signals), asset_name)

        return close_signals

    def _update_trailing_stops(self, asset_name: str, current_price: float) -> None:
        """Update trailing stops for open trades of a specific asset."""
        active_trades = [trade for trade in self.trade_manager.get_trade(status='open') 
                         if trade['asset_name'] == asset_name]
        
        for trade in active_trades:
            if trade['trailing_stop'] is not None:
                trailing_stop_pct = trade['trailing_stop'] / 100
                current_stop_loss = trade['stop_loss']

                if trade['direction'] in ['buy', 'long']:
                    new_stop_loss = current_price * (1 - trailing_stop_pct)
                    if new_stop_loss > current_stop_loss:
                        self.trade_manager.modify_trade_parameters(trade_id=trade['id'], stop_loss=new_stop_loss)
                        self.logger.info("Updated trailing stop for trade %d (%s). New stop loss: %.8f", 
                                         trade['id'], asset_name, new_stop_loss)
                elif trade['direction'] in ['sell', 'short']:
                    new_stop_loss = current_price * (1 + trailing_stop_pct)
                    if new_stop_loss < current_stop_loss:
                        self.trade_manager.modify_trade_parameters(trade_id=trade['id'], stop_loss=new_stop_loss)
                        self.logger.info("Updated trailing stop for trade %d (%s). New stop loss: %.8f", 
                                         trade['id'], asset_name, new_stop_loss)

    def _generate_strategy_signals(self, asset_name: str, row_data: pd.Series, current_price: float, timestamp: int) -> List[Dict[str, Any]]:
        """Generate strategy signals for a specific asset."""
        strategy = self.strategies[asset_name]
        active_trades = [trade for trade in self.trade_manager.get_trade(status='open') 
                         if trade['asset_name'] == asset_name]
        
        portfolio_value = self.portfolio.get_total_value({asset_name: current_price})
        portfolio_cash = self.portfolio.holdings.get(self.base_currency, 0)

        signals = strategy.generate_signal(row_data, active_trades, portfolio_value, portfolio_cash)
        
        if signals:
            signals = [signals] if isinstance(signals, dict) else signals
            for signal in signals:
                signal['timestamp'] = timestamp
            
            if self.debug_mode:
                self.logger.debug("Generated %d signals for asset %s: %s", len(signals), asset_name, signals)

        return signals

    def _process_signals(self, signals: List[Dict[str, Any]], market_prices: Dict[str, float], timestamp: int) -> None:
        """Process a batch of trading signals."""
        self.logger.debug("Processing %d signals at timestamp %d", len(signals), timestamp)
        
        if self.signal_database:
            self.signal_database.store_signals(signals)
            self.logger.debug("Stored %d signals in the signal database", len(signals))

        try:
            self.portfolio.process_signals(signals=signals, market_prices=market_prices, timestamp=timestamp)
            self.logger.debug("Signals processed successfully")
        except Exception as e:
            self.logger.error("Error processing signals: %s", str(e), exc_info=True)

    def _generate_close_signal(self, trade: Dict[str, Any], current_price: float, timestamp: int, reason: str) -> Dict[str, Any]:
        """Generate a close signal for a trade."""
        return {
            'action': 'close',
            'trade_id': trade['id'],
            'asset_name': trade['asset_name'],
            'amount': trade['asset_amount'],
            'price': current_price,
            'timestamp': timestamp,
            'reason': reason
        }

    def _log_initial_state(self) -> None:
        """Log the initial state of the portfolio at the start of the backtest."""
        self.logger.info("Initial portfolio state:")
        self.logger.info("Cash balance: %.2f %s", self.portfolio.holdings.get(self.base_currency, 0), self.base_currency)
        for asset, amount in self.portfolio.holdings.items():
            if asset != self.base_currency:
                self.logger.info("Asset %s: %.8f", asset, amount)


    def log_final_summary(self) -> None:
        """Log the final summary of the backtest."""
        pass
    '''
        self.logger.info("Backtest completed. Generating final summary...")
        all_trades = self.portfolio.trade_manager.get_trade()

        if not all_trades:
            self.logger.warning("No trades were executed during the backtest.")
            return

        try:
            trade_summary = self.metrics.calculate_trade_summary(
                trades=all_trades,
                portfolio_history=self.portfolio.history,
                market_data=self.market_data,
                slippage=self.slippage
            )

            self.logger.info("Backtest Summary:")
            for key, value in trade_summary.items():
                self.logger.info("%s: %s", key.replace('_', ' ').title(), value)

            # Log final portfolio state
            final_timestamp = max(self.unified_timestamps)
            final_market_prices = {
                asset: data.loc[final_timestamp, 'close'] 
                for asset, data in self.market_data.items() 
                if final_timestamp in data.index
            }
            final_portfolio_value = self.portfolio.get_total_value(final_market_prices)
            self.logger.info("Final Portfolio State:")
            self.logger.info("Total portfolio value: %.2f %s", final_portfolio_value, self.base_currency)
            for asset, amount in self.portfolio.holdings.items():
                if asset == self.base_currency:
                    self.logger.info("Cash balance: %.2f %s", amount, self.base_currency)
                else:
                    asset_value = amount * final_market_prices.get(asset, 0)
                    self.logger.info
        except Exception as e:
            self.logger.error(f"An error occurred during the backtest: {e}", exc_info=True)
            '''