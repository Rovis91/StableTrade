from src.logger import setup_logger
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.execution_model import ExecutionModel
from src.portfolio import Portfolio

# Create a logger for this module
logger = setup_logger(__name__)

class BacktestEngine:
    def __init__(self, data_path: str, output_path: str, strategy, execution_model: ExecutionModel, portfolio: Portfolio, preprocess_data: bool = True):
        """
        Initialize the backtest engine.

        Args:
            data_path (str): Path to the input data file.
            output_path (str): Path to save the processed data file.
            strategy: Strategy object implementing the trading strategy.
            execution_model (ExecutionModel): Execution model for placing and managing orders.
            portfolio (Portfolio): Portfolio object for managing positions.
            preprocess_data (bool): Whether to preprocess data before backtesting.
        """
        self.data_path = data_path
        self.output_path = output_path
        self.strategy = strategy
        self.execution_model = execution_model
        self.portfolio = portfolio
        self.data = None
        self.preprocess_data = preprocess_data
        self.initial_cash = portfolio.cash_balance

        # Validate input parameters
        self._validate_parameters()

    def _validate_parameters(self):
        """
        Validate input parameters to ensure they are correctly set.
        """
        if not isinstance(self.data_path, str) or not isinstance(self.output_path, str):
            raise ValueError("data_path and output_path must be strings.")
        if not hasattr(self.strategy, 'generate_signals'):
            raise ValueError("Strategy must have a 'generate_signals' method.")
        if not isinstance(self.execution_model, ExecutionModel):
            raise ValueError("execution_model must be an instance of ExecutionModel.")
        if not isinstance(self.portfolio, Portfolio):
            raise ValueError("portfolio must be an instance of Portfolio.")

    def run(self):
        """
        Run the backtest.
        """
        logger.info("Starting backtest...")
        if self.preprocess_data:
            self._preprocess_data()
        self._load_data()
        
        if self.data is None or self.data.empty or not self._verify_data_format(self.data):
            logger.error("Data is empty, not loaded, or incorrectly formatted. Aborting backtest.")
            return

        self._execute_backtest()
        self._finalize_backtest()

    def _preprocess_data(self):
        """
        Preprocess data by calculating required indicators.
        """
        logger.info("Preprocessing data for strategy...")
        required_indicators = self.strategy.get_required_indicators()
        preprocessor = DataPreprocessor(self.data_path, self.output_path)
        preprocessor.preprocess_data(required_indicators)

    def _load_data(self):
        """
        Load data from the output path.
        """
        try:
            logger.info("Loading enriched data...")
            data_loader = DataLoader(self.output_path)
            self.data = data_loader.load_data()
            logger.info(f"Data loaded. Total records: {len(self.data)}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self.data = None

    def _verify_data_format(self, data):
        """
        Verify the data format to ensure it contains the necessary columns.
        
        Args:
            data (pd.DataFrame): The DataFrame containing market data.

        Returns:
            bool: True if the data format is correct, False otherwise.
        """
        required_columns = ['close']
        if not all(column in data.columns for column in required_columns):
            logger.error(f"Data is missing required columns: {', '.join(required_columns)}")
            return False
        return True

    def _execute_backtest(self):
        """
        Execute the backtest by processing market events.
        """
        logger.info("Executing backtest...")
        
        # Precompute strategy signals for the entire dataset (vectorized operations)
        self.data['signals'] = self.data.apply(self._generate_signals_vectorized, axis=1)
        
        # Iterate through precomputed signals
        for timestamp, row in self.data.iterrows():
            self._process_market_event(timestamp, row)

    def _generate_signals_vectorized(self, market_data):
        """
        Precompute strategy signals for each row of the dataset.
        """
        return self.strategy.generate_signals(market_data)

    def _process_market_event(self, timestamp, market_data):
        """
        Process a single market event, executing trades and setting stop orders.
        
        Args:
            timestamp: The timestamp of the market event.
            market_data: The market data for the event.
        """
        logger.debug(f"Processing market event at {timestamp}")

        current_price = market_data['close']
        signals = market_data['signals']

        action = signals.get('action')
        amount = signals.get('amount', 0)
        stop_loss = signals.get('stop_loss')
        take_profit = signals.get('take_profit')
        trailing_stop = signals.get('trailing_stop')

        try:
            if action in ['buy', 'sell']:
                self._place_market_order(action, amount, current_price, timestamp)
                self._set_stop_orders(action, amount, current_price, stop_loss, take_profit, trailing_stop, timestamp)
            
            self._match_orders(current_price)
            self._log_portfolio_status(current_price)
        
        except Exception as e:
            logger.error(f"Error processing market event at {timestamp}: {e}")

    def _place_market_order(self, action, amount, current_price, timestamp):
        """
        Place a market order and update the portfolio.
        
        Args:
            action (str): The type of action ('buy' or 'sell').
            amount (float): The amount to trade.
            current_price (float): The current market price.
            timestamp: The timestamp of the market event.
        """
        order_type = 'market'
        executed_order = self.execution_model.place_order(order_type, amount, current_price=current_price, timestamp=timestamp)
        logger.info(f"Order executed: {executed_order}")
        
        if executed_order is not None:
            self.portfolio.update(executed_order)
        else:
            logger.warning(f"Failed to execute {action} order.")

    def _set_stop_orders(self, action, amount, current_price, stop_loss, take_profit, trailing_stop, timestamp):
        """
        Place stop-loss, take-profit, and trailing stop orders.
        
        Args:
            action (str): The type of action ('buy' or 'sell').
            amount (float): The amount to trade.
            current_price (float): The current market price.
            stop_loss (float or None): The stop-loss percentage.
            take_profit (float or None): The take-profit percentage.
            trailing_stop (float or None): The trailing stop percentage.
            timestamp: The timestamp of the market event.
        """
        try:
            if stop_loss is not None:
                stop_price = current_price * (1 - stop_loss) if action == 'buy' else current_price * (1 + stop_loss)
                self.execution_model.place_order('stop', -amount, stop_price=stop_price, timestamp=timestamp)

            if take_profit is not None:
                take_profit_price = current_price * (1 + take_profit) if action == 'buy' else current_price * (1 - take_profit)
                self.execution_model.place_order('stop', -amount, stop_price=take_profit_price, timestamp=timestamp)

            if trailing_stop is not None:
                self.execution_model.place_order('trailing_stop', -amount, trailing_amount=trailing_stop, current_price=current_price, timestamp=timestamp)

        except Exception as e:
            logger.error(f"Error setting stop orders: {e}")

    def _match_orders(self, current_price):
        """
        Match all active orders against the current market price.
        """
        self.execution_model.match_limit_orders(current_price)
        self.execution_model.match_stop_orders(current_price)
        self.execution_model.match_trailing_stop_orders(current_price)
        self.execution_model.remove_executed_orders()

    def _log_portfolio_status(self, current_price):
        """
        Log the current portfolio value and P&L.
        """
        portfolio_value = self.portfolio.get_portfolio_value({'asset': current_price})
        pnl = self.portfolio.get_pnl(self.initial_cash, {'asset': current_price})
        logger.info(f"Portfolio Value: {portfolio_value:.2f}, P&L: {pnl:.2f}")

    def _finalize_backtest(self):
        """
        Finalize the backtest and log the results.
        """
        logger.info("Finalizing backtest...")
        final_value = self.portfolio.get_portfolio_value({'asset': self.data.iloc[-1]['close']})
        final_pnl = self.portfolio.get_pnl(self.initial_cash, {'asset': self.data.iloc[-1]['close']})
        logger.info(f"Final Portfolio Value: {final_value:.2f}")
        logger.info(f"Total P&L: {final_pnl:.2f}")
