from typing import Dict, List
from src.logger import setup_logger

class Portfolio:
    """
    Portfolio class that manages asset holdings, trade execution, and portfolio performance.
    This class is responsible for handling incoming trading signals, maintaining a record of trades,
    and providing metrics related to the portfolio's performance.

    Attributes:
        base_currency (str): The base currency for the portfolio (e.g., 'EUR', 'USD').
        holdings (Dict[str, float]): A dictionary storing current holdings of assets and cash.
        history (List[Dict]): A list that keeps track of portfolio state over time.
        portfolio_config (Dict[str, Dict]): The configuration of each asset, including market type, fees, etc.
        trade_manager (TradeManager): The trade manager for handling trades.
        logger (logging.Logger): Logger instance for logging portfolio activities.
        signal_database (SignalDatabase): The signal database instance for storing signal data.

    Modules:
        process_signals: Processes incoming trading signals and executes trades.
        _validate_portfolio_config: Validates the portfolio configuration.
        _open_trade: Executes a trade based on a validated signal.
        store_history: Stores the current portfolio state in the history.
    """
    REQUIRED_PORTFOLIO_KEYS = {'market_type', 'fees', 'max_trades', 'max_exposure'}

    def __init__(self, initial_cash: float, portfolio_config: Dict[str, Dict], 
                 signal_database, trade_manager, base_currency: str):
        """
        Initialize the portfolio with the given initial cash balance, base currency, and portfolio configurations.

        Args:
            initial_cash (float): Starting cash balance in the base currency.
            portfolio_config (Dict[str, Dict]): Configuration for each asset (market type, fees, max exposure, etc.).
            signal_database (SignalDatabase): The signal database to store and retrieve signals.
            trade_manager (TradeManager): The trade manager instance for managing trades.
            base_currency (str): The base currency for the portfolio (e.g., 'EUR', 'USD').
        """
        self.base_currency = base_currency
        self.holdings = {base_currency: initial_cash}
        self.history: List[Dict] = []
        self.portfolio_config = portfolio_config
        self.trade_manager = trade_manager
        self.logger = setup_logger('portfolio')
        self.signal_database = signal_database

        self._validate_portfolio_config(portfolio_config)

        self.logger.info(f"Portfolio initialized with {initial_cash} {base_currency}")
    
    def _validate_portfolio_config(self, config: Dict[str, Dict]) -> None:
        """
        Validate the portfolio configuration.

        Args:
            config (Dict[str, Dict]): The portfolio configuration to validate.
        """
        if not isinstance(config, dict):
            raise TypeError("Portfolio configuration must be a dictionary")

        for asset, asset_config in config.items():
            if not isinstance(asset, str):
                raise TypeError(f"Asset key must be a string, got {type(asset)}")
            
            if not isinstance(asset_config, dict):
                raise TypeError(f"Configuration for asset {asset} must be a dictionary")

            if not self.REQUIRED_PORTFOLIO_KEYS.issubset(asset_config.keys()):
                missing_keys = self.REQUIRED_PORTFOLIO_KEYS - asset_config.keys()
                error_msg = f"Invalid configuration for asset {asset}. Missing required keys: {missing_keys}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Validate specific fields
            if asset_config['market_type'] not in ['spot', 'futures']:
                raise ValueError(f"Invalid market type for asset {asset}: {asset_config['market_type']}")
            
            if not isinstance(asset_config['fees'], dict) or 'entry' not in asset_config['fees'] or 'exit' not in asset_config['fees']:
                raise ValueError(f"Invalid fee structure for asset {asset}")
            
            if not isinstance(asset_config['max_trades'], (int, float)) or asset_config['max_trades'] <= 0:
                raise ValueError(f"Invalid max_trades for asset {asset}: {asset_config['max_trades']}")
            
            if not isinstance(asset_config['max_exposure'], (int, float)) or asset_config['max_exposure'] <= 0 or asset_config['max_exposure'] > 1:
                raise ValueError(f"Invalid max_exposure for asset {asset}: {asset_config['max_exposure']}")

        self.logger.debug("Portfolio configuration validated successfully")
  
    def process_signals(self, signals: List[Dict], market_prices: Dict[str, float], timestamp: int) -> None:
        """
        Process a list of trading signals, executing trades and updating the portfolio accordingly.

        Args:
            signals (List[Dict]): List of trading signals to process.
            market_prices (Dict[str, float]): Current market prices for assets.
            timestamp (int): The current timestamp for the signals being processed.
        """
        self.logger.info(f"Processing {len(signals)} signals at timestamp {timestamp}")

        for signal in signals:
            try:
                if not self.signal_database.validate_signal(signal):
                    self.logger.warning(f"Invalid signal format: {signal}")
                    self.signal_database.update_signal_field(signal_id=signal['signal_id'], field_name='status', new_value='rejected')
                    continue

                asset_name = signal['asset_name']
                action = signal['action']
                asset_price = market_prices[asset_name]
                
                if action == 'close':
                    self._close_trade(signal, timestamp)
                elif action in ['buy', 'sell']:
                    if not self.validate_portfolio_constraints(signal, market_prices):
                        self.signal_database.update_signal_field(signal_id=signal['signal_id'], field_name='status', new_value='rejected')
                        continue

                    trade_amount_percentage = float(signal['amount'])
                    if action == 'buy':
                        available_cash = self.holdings[self.base_currency]
                        cash_to_use = available_cash * trade_amount_percentage
                        asset_quantity = cash_to_use / asset_price
                    else:  # 'sell'
                        asset_holdings = self.holdings.get(asset_name, 0.0)
                        asset_quantity = asset_holdings * trade_amount_percentage
                        cash_to_use = asset_quantity * asset_price

                    self._open_trade(signal=signal, asset_quantity=asset_quantity, 
                                    base_amount=cash_to_use, asset_price=asset_price, 
                                    timestamp=timestamp)

            except Exception as e:
                self.logger.error(f"Error processing signal: {e}", exc_info=True)

        self.store_history(timestamp)
        open_trades = self.trade_manager.get_trade(status='open')
        self.logger.debug(f"After processing signals at {timestamp}, open trades: {len(open_trades)}")

    def validate_portfolio_constraints(self, signal: Dict, market_prices: Dict[str, float]) -> bool:
        """
        Validates whether a trade can be executed based on portfolio constraints.

        Args:
            signal (Dict): The trading signal to validate.
            market_prices (Dict[str, float]): Current market prices for assets.

        Returns:
            bool: True if the signal is valid and can be executed, False otherwise.
        """
        asset = signal['asset_name']
        action = signal['action']
        amount = float(signal['amount'])
        price = market_prices[asset]

        try:
            entry_fee = self.get_fee(asset_name=asset, fee_type='entry')
            market_type = self.portfolio_config[asset]['market_type']
            max_trades = self.portfolio_config[asset].get('max_trades', float('inf'))
            max_exposure = float(self.portfolio_config[asset].get('max_exposure'))

            open_trades = self.trade_manager.get_trade(status='open')
            asset_open_trades = [trade for trade in open_trades if trade['asset_name'] == asset]

            if len(asset_open_trades) >= max_trades:
                self.logger.warning(f"Max trades limit reached for {asset}. Cannot open new trade.")
                return False

            portfolio_value = self.get_total_value(market_prices=market_prices)
            current_exposure = self.get_asset_exposure(asset=asset, market_prices=market_prices)
            new_exposure = (amount * price) / portfolio_value if portfolio_value else 0

            if current_exposure + new_exposure > max_exposure:
                self.logger.info(f"Max exposure limit reached for {asset}.")
                return False

            if market_type == 'spot':
                available_cash = self.holdings[self.base_currency]
                total_cost = amount * price + entry_fee
                if action == 'buy' and available_cash < total_cost:
                    self.logger.info(f"Not enough {self.base_currency} for the trade.")
                    return False
                elif action == 'sell' and self.holdings.get(asset, 0.0) < amount:
                    self.logger.info(f"Not enough {asset} holdings for the trade.")
                    return False
            elif market_type == 'futures':
                # TODO: Implement futures validation
                pass

            return True

        except Exception as e:
            self.logger.error(f"Error during signal validation: {e}", exc_info=True)
            return False
        
    def _open_trade(self, signal: Dict, asset_quantity: float, base_amount: float, asset_price: float, timestamp: int) -> None:
        """
        Open a new trade based on a validated signal and update portfolio holdings.

        Args:
            signal (Dict): The validated trade signal containing trade details.
            asset_quantity (float): The quantity of the asset to trade.
            base_amount (float): The amount in base currency involved in the trade.
            asset_price (float): The current price of the asset.
            timestamp (int): The timestamp of the trade execution.
        """
        try:
            asset_name = signal['asset_name']
            entry_fee = self.get_fee(asset_name, 'entry') * base_amount
            stop_loss = float(signal.get('stop_loss', 0))
            take_profit = float(signal.get('take_profit', 0))
            trailing_stop = float(signal.get('trailing_stop', 0))
            entry_reason = signal.get('reason', 'buy_signal')

            self.logger.info(f"Opening trade for {asset_name} with action {signal['action']}")

            trade = self.trade_manager.open_trade(
                asset_name=asset_name,
                base_currency=self.base_currency,
                asset_amount=asset_quantity,
                base_amount=base_amount,
                entry_price=asset_price,
                entry_timestamp=timestamp,
                entry_fee=entry_fee,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=trailing_stop,
                direction=signal['action'],
                entry_reason=entry_reason
            )
            self.logger.debug(f"Trade opened: {trade}")

            # Update signal status to 'executed' in the signal database
            signal_id = signal['signal_id']

            self.signal_database.update_signal_field(signal_id=signal_id, field_name='status', new_value='executed')
            self.signal_database.update_signal_field(signal_id=signal_id, field_name='trade_id', new_value=trade['id'])


            # Update portfolio holdings based on the executed trade
            market_type = self.portfolio_config[asset_name]['market_type']
            if market_type == 'spot':
                if signal['action'] == 'buy':
                    entry_fee = self.get_fee(asset_name, 'entry') * base_amount  
                    self.holdings[self.base_currency] -= base_amount + entry_fee
                    self.holdings[asset_name] = self.holdings.get(asset_name, 0) + asset_quantity
                elif signal['action'] == 'sell':
                    exit_fee = self.get_fee(asset_name, 'exit') * base_amount
                    self.holdings[self.base_currency] += base_amount - exit_fee
                    self.holdings[asset_name] -= asset_quantity
                    if self.holdings[asset_name] <= 0:
                        self.holdings[asset_name] = 0
            elif market_type == 'futures':
                self.logger.info(f"Futures trade opened for {asset_name}, amount: {asset_quantity}. No direct asset holding updated.")
                # TODO: Implement futures trade handling

            self.logger.info(f"Updated holdings after opening trade. New {self.base_currency} balance: {self.holdings[self.base_currency]}")

        except Exception as e:
            self.logger.error(f"Error opening trade: {e}", exc_info=True)
            raise

    def _close_trade(self, signal: Dict, timestamp: int) -> None:
        """
        Close an existing trade based on a signal.

        Args:
            signal (Dict): The close trade signal containing trade details.
            timestamp (int): The timestamp of the trade closure.
        """
        try:
            asset_name = signal['asset_name']
            trade_id = signal['trade_id']
            exit_price = float(signal['price'])
            exit_reason = signal['reason']

            self.logger.info(f"Attempting to close trade {trade_id} for {asset_name}")

            trades = self.trade_manager.get_trade(trade_id=trade_id)
            if isinstance(trades, list):
                trade = trades[0] if trades else None
            else:
                trade = trades

            if trade and trade.get('status') == 'open':
                exit_fee = self.get_fee(asset_name, 'exit') * trade['base_amount']
                closed_trade = self.trade_manager.close_trade(
                    trade_id=trade_id,
                    exit_price=exit_price,
                    exit_timestamp=timestamp,
                    exit_fee=exit_fee,
                    exit_reason=exit_reason
                )
                if closed_trade:
                    self.update_holdings(closed_trade)
                    self.logger.info(f"Trade {trade_id} closed successfully")
                    signal_id = signal['signal_id']
                    self.signal_database.update_signal_field(signal_id=signal_id, field_name='status', new_value='executed')
                    self.signal_database.update_signal_field(signal_id=signal_id, field_name='trade_id', new_value=trade_id)

                else:
                    self.logger.error(f"Failed to close trade {trade_id}")
            else:
                self.logger.warning(f"Trade {trade_id} not found or already closed. Trade status: {trade['status'] if trade else 'Not found'}")

        except Exception as e:
            self.logger.error(f"Error closing trade: {e}", exc_info=True)
            raise

    def update_holdings(self, trade: Dict) -> None:
        """
        Updates portfolio holdings and cash after a trade is closed.

        Args:
            trade (Dict): The closed trade details.
        """
        try:
            asset = trade['asset_name']
            amount = float(trade['asset_amount'])
            base_amount = float(trade['base_amount'])
            exit_fee = float(trade['exit_fee'])

            market_type = self.portfolio_config[asset]['market_type']

            if market_type == 'spot':
                if trade['direction'] == 'sell':
                    if self.holdings.get(asset, 0) < amount:
                        raise ValueError(f"Not enough {asset} holdings to sell. Available: {self.holdings.get(asset, 0)}, Required: {amount}")

                    self.holdings[self.base_currency] += base_amount - exit_fee
                    self.holdings[asset] -= amount
                    if self.holdings[asset] <= 0:
                        del self.holdings[asset]
                else:  # buy
                    self.holdings[self.base_currency] -= base_amount + exit_fee
                    self.holdings[asset] = self.holdings.get(asset, 0) + amount

                self.logger.info(f"Updated holdings after closing {trade['direction']} trade of {amount} {asset}. "
                                f"New {self.base_currency} balance: {self.holdings[self.base_currency]}")
            elif market_type == 'futures':
                self.logger.info(f"Futures trade closed for {asset}, amount: {amount}. No direct asset holding updated.")
                # TODO: Implement futures trade handling
            else:
                raise ValueError(f"Unsupported market type {market_type} for asset {asset}")

        except Exception as e:
            self.logger.error(f"Error updating holdings: {e}", exc_info=True)
            raise

    def get_total_value(self, market_prices: Dict[str, float]) -> float:
        """
        Calculates the total value of the portfolio (cash + current asset values).

        Args:
            market_prices (Dict[str, float]): Current market prices for assets.

        Returns:
            float: The total portfolio value.
        """
        total_value = self.holdings.get(self.base_currency, 0)

        for asset, quantity in self.holdings.items():
            if asset != self.base_currency and asset in market_prices:
                asset_value = quantity * market_prices[asset]
                total_value += asset_value

        return total_value

    def get_fee(self, asset_name: str, fee_type: str = 'entry') -> float:
        """
        Get the fee for the specified asset.

        Args:
            asset_name (str): The name of the asset.
            fee_type (str): The type of fee ('entry' or 'exit').

        Returns:
            float: The fee percentage.
        """
        return float(self.portfolio_config.get(asset_name, {}).get('fees', {}).get(fee_type, 0))

    def get_asset_exposure(self, asset: str, market_prices: Dict[str, float]) -> float:
        """
        Calculate the current exposure for a specific asset.

        Args:
            asset (str): The asset to calculate exposure for.
            market_prices (Dict[str, float]): Current market prices for assets.

        Returns:
            float: The current exposure of the asset as a percentage of the total portfolio value.
        """
        if asset not in self.holdings or asset not in market_prices:
            return 0
        
        portfolio_value = self.get_total_value(market_prices)
        if portfolio_value == 0:
            return 0
        
        exposure = (self.holdings[asset] * market_prices[asset]) / portfolio_value
        self.logger.info(f"Current exposure for {asset}: {exposure:.2%}")
        return exposure

    def store_history(self, timestamp: int) -> None:
        """
        Store the current portfolio state in the history.

        Args:
            timestamp (int): The current timestamp to associate with the portfolio state.
        """
        self.history.append({
            'timestamp': timestamp,
            'holdings': self.holdings.copy()
        })
        self.logger.debug(f"Portfolio state stored at timestamp {timestamp}")