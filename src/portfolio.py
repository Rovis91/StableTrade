import os
import logging
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

        # Validate the portfolio configuration to ensure it has required parameters
        self._validate_portfolio_config(portfolio_config)

        self.logger.info(f"Portfolio initialized with {initial_cash} {base_currency}")
    
    def _validate_portfolio_config(self, config: Dict[str, Dict]) -> None:
        """
        Validate the portfolio configuration.

        Args:
            config (Dict[str, Dict]): The portfolio configuration to validate.

        Raises:
            ValueError: If the configuration is invalid.
        """
        for asset, asset_config in config.items():
            if not self.REQUIRED_PORTFOLIO_KEYS.issubset(asset_config.keys()):
                missing_keys = self.REQUIRED_PORTFOLIO_KEYS - asset_config.keys()
                error_msg = f"Invalid configuration for asset {asset}. Missing required keys: {missing_keys}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        self.logger.debug("Portfolio configuration validated successfully")

    def process_signals(self, signals: List[Dict], market_prices: Dict[str, float], timestamp: int) -> None:
        self.logger.info(f"Processing {len(signals)} signals at timestamp {timestamp}")

        for signal in signals:
            try:
                asset_name = signal['asset_name']
                action = signal['action']
                
                if action == 'close':
                    self.logger.info(f"Received close signal for trade {signal.get('trade_id')} of {asset_name}")
                    self._close_trade(signal, timestamp)
                elif action in ['buy', 'sell']:
                    trade_amount_percentage = float(signal['amount'])
                    asset_price = market_prices[asset_name]

                    # Validate the signal
                    if not self.validate_signal(signal, market_prices):
                        signal_id = signal['signal_id']
                        self.signal_database.update_signal_status(signal_id, 'rejected')
                        continue

                    # Perform action based on the signal's action type
                    available_cash = self.holdings.get(self.base_currency, 0.0)
                    asset_holdings = self.holdings.get(asset_name, 0.0)

                    if action == 'buy':
                        cash_to_use = available_cash * trade_amount_percentage
                        asset_quantity = cash_to_use / asset_price
                        base_amount = cash_to_use
                        self._execute_trade(signal, asset_quantity, base_amount, asset_price, timestamp)
                    elif action == 'sell':
                        asset_quantity = asset_holdings * trade_amount_percentage
                        base_amount = asset_quantity * asset_price
                        self._execute_trade(signal, asset_quantity, base_amount, asset_price, timestamp)
                else:
                    self.logger.warning(f"Unknown action {action} in signal: {signal}")

            except KeyError as e:
                self.logger.error(f"Missing key in signal: {e}", exc_info=True)
            except Exception as e:
                self.logger.error(f"Error processing signal: {e}", exc_info=True)

        # Store the current portfolio state in the history for tracking purposes
        self.store_history(timestamp)
        
        # Log the number of open trades after processing signals
        open_trades = self.trade_manager.get_trade(status='open')
        self.logger.debug(f"After processing signals at {timestamp}, open trades: {len(open_trades)}")

    def validate_signal(self, signal: Dict, market_prices: Dict[str, float]) -> bool:
        """
        Validates whether a trade can be executed based on portfolio constraints.

        Args:
            signal (Dict): The trading signal to validate.
            market_prices (Dict[str, float]): Current market prices for assets.

        Returns:
            bool: True if the signal is valid and can be executed, False otherwise.
        """
        asset = signal.get('asset_name')
        action = signal.get('action')
        amount = float(signal.get('amount', 0.0))
        price = market_prices.get(asset, 0.0)

        if not asset or not action:
            self.logger.error("Signal is missing required fields: 'asset_name' or 'action'")
            return False

        try:
            entry_fee = self.get_fee(asset, 'entry')
            market_type = self.portfolio_config[asset]['market_type']
            max_trades = self.portfolio_config[asset].get('max_trades', float('inf'))
            max_exposure = float(self.portfolio_config[asset].get('max_exposure', 1.0))

            open_trades = self.trade_manager.get_trade(status='open')
            asset_open_trades = [trade for trade in open_trades if trade['asset_name'] == asset]

            # Check max trades and exposure limits
            if len(asset_open_trades) >= max_trades:
                self.logger.warning(f"Max trades limit reached for {asset}. Cannot open new trade.")
                return False

            portfolio_value = self.get_total_value(market_prices)
            current_exposure = self.get_asset_exposure(asset, market_prices)
            new_exposure = (amount * price) / portfolio_value

            if current_exposure + new_exposure > max_exposure:
                self.logger.info(f"Max exposure limit reached for {asset}.")
                return False

            # Spot market validation
            if market_type == 'spot':
                available_cash = self.holdings.get(self.base_currency, 0.0)
                total_cost = amount * price + entry_fee
                if action == 'buy' and available_cash >= total_cost:
                    return True
                elif action == 'sell' and self.holdings.get(asset, 0.0) >= amount:
                    return True
                else:
                    self.logger.info(f"Not enough {self.base_currency} or holdings for the trade.")

            # Futures market validation (can be expanded)
            elif market_type == 'futures':
                return True  # Futures logic can be expanded here

        except KeyError as e:
            self.logger.error(f"Key error during signal validation: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Unexpected error during signal validation: {e}", exc_info=True)

        return False

    def _execute_trade(self, signal: Dict, asset_quantity: float, base_amount: float, asset_price: float, timestamp: int) -> None:
            """
            Execute a validated trade and update portfolio holdings.

            Args:
                signal (Dict): The validated trade signal containing trade details.
                asset_quantity (float): The quantity of the asset to trade.
                base_amount (float): The amount in base currency involved in the trade.
                asset_price (float): The current price of the asset.
                timestamp (int): The timestamp of the trade execution.
            """
            try:
                asset_name = signal['asset_name']
                entry_fee = self.get_fee(asset_name, 'entry')
                stop_loss = float(signal.get('stop_loss', 0.0))
                take_profit = float(signal.get('take_profit', 0.0))
                trailing_stop = float(signal.get('trailing_stop', 0.0))
                entry_reason = signal.get('reason', 'buy_signal')

                self.logger.info(f"Executing trade for {asset_name} with action {signal['action']}")

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
                self.logger.debug(f"Trade executed: {trade}")

                # Update signal status to 'executed' in the signal database
                signal_id = signal['signal_id']
                self.signal_database.update_signal_status(signal_id, 'executed')


                # Update portfolio holdings based on the executed trade
                market_type = self.portfolio_config[asset_name]['market_type']
                if market_type == 'spot':
                    if signal['action'] == 'buy':
                        self.holdings[self.base_currency] -= base_amount + entry_fee
                        self.holdings[asset_name] = self.holdings.get(asset_name, 0.0) + asset_quantity
                    elif signal['action'] == 'sell':
                        self.holdings[self.base_currency] += base_amount - self.get_fee(asset_name, 'exit')
                        self.holdings[asset_name] -= asset_quantity
                        if self.holdings[asset_name] == 0:
                            del self.holdings[asset_name]
                elif market_type == 'futures':
                    self.logger.info(f"Futures trade executed for {asset_name}, amount: {asset_quantity}. No direct asset holding updated.")

                self.logger.info(f"Updated holdings after executing trade. New {self.base_currency} balance: {self.holdings[self.base_currency]}")

            except Exception as e:
                self.logger.error(f"Error executing trade: {e}", exc_info=True)

    def _close_trade(self, signal: Dict, timestamp: int) -> None:
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
                exit_fee = self.get_fee(asset_name, 'exit')
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
                    self.signal_database.update_signal_status(signal['signal_id'], 'executed')
                else:
                    self.logger.error(f"Failed to close trade {trade_id}")
            else:
                self.logger.warning(f"Trade {trade_id} not found or already closed. Trade status: {trade.get('status') if trade else 'Not found'}")

        except Exception as e:
            self.logger.error(f"Error closing trade: {e}", exc_info=True)

    def update_holdings(self, trade: Dict) -> None:
        """
        Updates portfolio holdings and cash after a trade is executed.

        Args:
            trade (Dict): The executed trade details.
        """
        try:
            asset = trade['asset_name']
            amount = trade['asset_amount']
            base_amount = trade['base_amount']
            entry_fee = trade['entry_fee']

            market_type = self.portfolio_config[asset]['market_type']

            if market_type == 'spot':
                if trade['direction'] == 'sell':
                    exit_fee = self.get_fee(asset, 'exit')
                    self.holdings[self.base_currency] += base_amount - exit_fee
                    self.holdings[asset] -= amount
                    if self.holdings[asset] == 0:
                        del self.holdings[asset]
                    self.logger.info(f"Updated holdings after selling {amount} of {asset}. "
                                     f"New {self.base_currency} balance: {self.holdings[self.base_currency]}")
                else:  # buy
                    self.holdings[self.base_currency] -= base_amount + entry_fee
                    self.holdings[asset] = self.holdings.get(asset, 0.0) + amount
                    self.logger.info(f"Updated holdings after buying {amount} of {asset}. "
                                     f"New {self.base_currency} balance: {self.holdings[self.base_currency]}")
            elif market_type == 'futures':
                self.logger.info(f"Futures trade executed for {asset}, amount: {amount}. No direct asset holding updated.")
                # TODO: Implement futures trade handling
            else:
                self.logger.error(f"Unsupported market type {market_type} for asset {asset}")

        except Exception as e:
            self.logger.error(f"Error updating holdings: {e}", exc_info=True)

    def get_total_value(self, market_prices: Dict[str, float]) -> float:
        """
        Calculates the total value of the portfolio (cash + current asset values).

        Args:
            market_prices (Dict[str, float]): Current market prices for assets.

        Returns:
            float: The total portfolio value.
        """
        total_value = self.holdings.get(self.base_currency, 0.0)

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
            float: The fee amount.
        """
        fee = float(self.portfolio_config.get(asset_name, {}).get('fees', {}).get(fee_type, 0.0))
        return fee

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
            return 0.0
        
        portfolio_value = self.get_total_value(market_prices)
        if portfolio_value == 0:
            return 0.0
        
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