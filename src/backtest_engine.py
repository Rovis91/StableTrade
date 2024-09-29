from src.logger import setup_logger
import pandas as pd
from typing import Dict, List, Any, Optional
from src.data_preprocessor import DataPreprocessor
from src.metrics import MetricsModule
from src.signal_database import SignalDatabase
import uuid
from tqdm import tqdm


class BacktestEngine:
    """
    A class to run backtests on trading strategies for multiple assets.

    This engine preprocesses data, runs strategies, manages trades, and calculates performance metrics.

    Attributes:
        assets (Dict[str, str]): A dictionary mapping asset names to their corresponding CSV paths.
        strategies (Dict[str, Any]): A dictionary mapping asset names to their strategy objects.
        portfolio: Portfolio instance for managing asset holdings.
        trade_manager: TradeManager instance for managing trades.
        base_currency (str): The base currency for trading (default: 'USD').
        slippage (float): The slippage rate to apply to trades.
        latency (int): The simulated latency (in milliseconds) to apply to trade execution.
        metrics (MetricsModule): An instance of MetricsModule for calculating performance metrics.
        signal_database (SignalDatabase): An instance of SignalDatabase for storing and retrieving signals.
    """

    STATUS_OPEN = 'open'
    STATUS_CLOSED = 'closed'
    BUY = 'buy'
    SELL = 'sell'
    LONG = 'long'
    SHORT = 'short'

    def __init__(self, assets: Dict[str, str], strategies: Dict[str, Any], portfolio, trade_manager, 
                 base_currency: str = "USD", slippage: float = 0.0, latency: int = 0, 
                 metrics: Optional[MetricsModule] = None, signal_database: Optional[SignalDatabase] = None,
                 logger: Optional[Any] = None):
        """
        Initialize the BacktestEngine with the given parameters.

        Args:
            assets (Dict[str, str]): Dictionary mapping asset names to CSV file paths.
            strategies (Dict[str, Any]): Dictionary mapping asset names to strategy instances.
            portfolio: Instance managing portfolio operations.
            trade_manager: Instance managing trade operations.
            base_currency (str): The base currency for trading.
            slippage (float): The slippage rate for trades.
            latency (int): The simulated latency for trade execution.
            metrics (Optional[MetricsModule]): MetricsModule instance for calculating performance metrics.
            signal_database (Optional[SignalDatabase]): SignalDatabase instance for storing signals.
            logger (Optional[logging.Logger]): Custom logger for BacktestEngine logging.
        """
        self.logger = logger if logger else setup_logger('backtest_engine')
        
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
        self.metrics = metrics
        self.signal_database = signal_database if signal_database else SignalDatabase(logger=self.logger)

        self.logger.info("BacktestEngine initialized with %d assets and %d strategies", 
                         len(assets), len(strategies))

    def _validate_inputs(self, assets, strategies, portfolio, trade_manager, base_currency, slippage, latency):
        """Validate input parameters for the BacktestEngine."""
        if not (isinstance(assets, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in assets.items())):
            raise ValueError("Assets must be a dictionary with string keys and values.")
        if not (isinstance(strategies, dict) and all(isinstance(k, str) for k in strategies.keys())):
            raise ValueError("Strategies must be a dictionary with string keys.")
        if not (hasattr(portfolio, 'process_signals') and callable(getattr(portfolio, 'process_signals'))):
            raise ValueError("Portfolio must have a 'process_signals' method.")
        if not (hasattr(trade_manager, 'get_trade') and callable(getattr(trade_manager, 'get_trade'))):
            raise ValueError("TradeManager must have a 'get_trade' method.")
        if not isinstance(base_currency, str):
            raise ValueError("Base currency must be a string.")
        if not (isinstance(slippage, (int, float)) and slippage >= 0):
            raise ValueError("Slippage must be a non-negative number.")
        if not (isinstance(latency, int) and latency >= 0):
            raise ValueError("Latency must be a non-negative integer.")

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
            except Exception as e:
                self.logger.error("Error preprocessing data for %s: %s", asset_name, str(e), exc_info=True)
                raise

        if len(self.market_data) != len(self.assets):
            self.logger.error("Not all assets were successfully preprocessed.")
            raise ValueError("Preprocessing failed for some assets.")

        self.unified_timestamps = sorted(set(ts for asset in self.market_data.values() for ts in asset.index))
        self.logger.info("Unified timestamps generated with %d entries.", len(self.unified_timestamps))

    def run_backtest(self) -> None:
        """Run the backtest across all unified timestamps."""
        self.logger.info("Starting backtest...")
        if not self.unified_timestamps:
            self.logger.error("No timestamps available for backtesting.")
            raise ValueError("No timestamps available. Ensure data preprocessing was successful.")

        total_timestamps = len(self.unified_timestamps)
        self.logger.info("Total timestamps for backtest: %d", total_timestamps)

        self._log_initial_state()

        for timestamp in tqdm(self.unified_timestamps, desc="Backtesting Progress"):
            try:
                self._process_timestamp(timestamp)
            except Exception as e:
                self.logger.error("Error during backtest at timestamp %d: %s", timestamp, str(e), exc_info=True)

        # Close all open trades at the end of the backtest
        self._close_all_open_trades(self.unified_timestamps[-1])

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
        
        # 5. Store the portfolio history
        self.portfolio.store_history(timestamp)

    def _check_stop_loss_take_profit(self, asset_name: str, current_price: float, timestamp: int) -> List[Dict[str, Any]]:
        """Check for stop loss and take profit conditions for open trades of a specific asset."""
        close_signals = []
        active_trades = [trade for trade in self.trade_manager.get_trade(status=self.STATUS_OPEN) 
                         if trade['asset_name'] == asset_name]
        
        for trade in active_trades:
            if trade['stop_loss'] is not None and current_price <= trade['stop_loss']:
                self.logger.info("Stop loss triggered for trade %d at price %.8f", trade['id'], current_price)
                close_signals.append(self._generate_close_signal(trade, current_price, timestamp, "stop_loss"))
            elif trade['take_profit'] is not None and current_price >= trade['take_profit']:
                self.logger.info("Take profit triggered for trade %d at price %.8f", trade['id'], current_price)
                close_signals.append(self._generate_close_signal(trade, current_price, timestamp, "take_profit"))

        return close_signals

    def _update_trailing_stops(self, asset_name: str, current_price: float) -> None:
        """Update trailing stops for open trades of a specific asset."""
        active_trades = [trade for trade in self.trade_manager.get_trade(status=self.STATUS_OPEN) 
                         if trade['asset_name'] == asset_name]
        
        for trade in active_trades:
            if trade['trailing_stop'] is not None:
                trailing_stop_pct = trade['trailing_stop'] / 100
                current_stop_loss = trade['stop_loss']

                if trade['direction'] in [self.BUY, self.LONG]:
                    new_stop_loss = current_price * (1 - trailing_stop_pct)
                    if new_stop_loss > current_stop_loss:
                        self.trade_manager.modify_trade_parameters(trade_id=trade['id'], stop_loss=new_stop_loss)
                        self.logger.info("Updated trailing stop for trade %d (%s). New stop loss: %.8f", 
                                         trade['id'], asset_name, new_stop_loss)
                elif trade['direction'] in [self.SELL, self.SHORT]:
                    new_stop_loss = current_price * (1 + trailing_stop_pct)
                    if new_stop_loss < current_stop_loss:
                        self.trade_manager.modify_trade_parameters(trade_id=trade['id'], stop_loss=new_stop_loss)
                        self.logger.info("Updated trailing stop for trade %d (%s). New stop loss: %.8f", 
                                         trade['id'], asset_name, new_stop_loss)

    def _generate_strategy_signals(self, asset_name: str, row_data: pd.Series, current_price: float, timestamp: int) -> List[Dict[str, Any]]:
        """Generate strategy signals for a specific asset."""
        strategy = self.strategies[asset_name]
        active_trades = [trade for trade in self.trade_manager.get_trade(status=self.STATUS_OPEN) 
                         if trade['asset_name'] == asset_name]
        
        portfolio_value = self.portfolio.get_total_value({asset_name: current_price})
        portfolio_cash = self.portfolio.holdings.get(self.base_currency, 0)

        signals = strategy.generate_signal(row_data, active_trades, portfolio_value, portfolio_cash)
        
        if signals:
            signals = [signals] if isinstance(signals, dict) else signals
            for signal in signals:
                signal['timestamp'] = timestamp

        return signals

    def _process_signals(self, signals: List[Dict[str, Any]], market_prices: Dict[str, float], timestamp: int) -> None:
        """Process a batch of trading signals."""
        self.logger.debug("Processing %d signals at timestamp %d", len(signals), timestamp)
        self.signal_database.add_signals(signals)
        
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

    def log_final_summary(self):
        self.logger.info("Generating final backtest summary...")

        try:
            # Calculate metrics
            sharpe_ratio = self.metrics.calculate_sharpe_ratio(self.market_data)
            max_drawdown = self.metrics.calculate_max_drawdown(self.market_data)
            cumulative_return = self.metrics.calculate_cumulative_return(self.market_data)
            profit_factor = self.metrics.calculate_profit_factor(self.trade_manager.get_trade())
            win_rate = self.metrics.calculate_win_rate(self.trade_manager.get_trade())
            
            # Get trade summary
            trade_summary = self.metrics.calculate_trade_summary(self.trade_manager.get_trade(), self.market_data)

            if "error" in trade_summary:
                self.logger.warning(trade_summary["error"])
                return

            # Log results
            self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            self.logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")
            self.logger.info(f"Cumulative Return: {cumulative_return:.2%}")
            self.logger.info(f"Profit Factor: {profit_factor:.4f}")
            self.logger.info(f"Win Rate: {win_rate:.2%}")
            self.logger.info("Trade Summary:")
            for key, value in trade_summary.items():
                self.logger.info(f"  {key}: {value}")

        except ValueError as e:
            self.logger.warning(f"Unable to generate summary: {str(e)}")

        self.logger.info("Backtest summary generation completed.")

    def _close_all_open_trades(self, final_timestamp: int) -> None:
        """
        Close all open trades at the end of the backtest.

        Args:
            final_timestamp (int): The timestamp of the last data point in the backtest.
        """
        self.logger.info("Closing all open trades at the end of the backtest.")
        open_trades = self.trade_manager.get_trade(status=self.STATUS_OPEN)
        self.logger.debug(f"Found {len(open_trades)} open trades to close.")
        close_signals = []

        for trade in open_trades:
            asset_name = trade['asset_name']
            current_price = self.market_data[asset_name].loc[final_timestamp, 'close']
            
            close_signal = self._generate_close_signal(trade, current_price, final_timestamp, "end_of_backtest")
            close_signals.append(close_signal)

        if close_signals:
            self.logger.info(f"Generated {len(close_signals)} close signals at the end of backtest.")
            market_prices = {asset: self.market_data[asset].loc[final_timestamp, 'close'] for asset in self.market_data}
            self._process_signals(close_signals, market_prices, final_timestamp)
        else:
            self.logger.info("No open trades to close at the end of the backtest.")
