import logging
from typing import Dict, List, Any
from src.trade_manager import TradeManager

class Portfolio:
    def __init__(self, initial_cash: float, portfolio_config: Dict[str, Dict], base_currency: str, verbose: bool = False):
        """
        Initialize the portfolio with the given initial cash balance, base currency, and portfolio configurations.

        Args:
            initial_cash (float): Starting cash balance in the base currency.
            portfolio_config (Dict[str, Dict]): Configuration for each asset (market type, fees, max exposure, etc.)
            base_currency (str): The base currency for the portfolio (e.g., 'EUR', 'USD').
            verbose (bool): If True, enables verbose DEBUG logging.
        """
        self.base_currency = base_currency
        self.holdings = {base_currency: initial_cash}
        self.history: List[Dict] = []
        self.portfolio_config = portfolio_config
        self.trade_manager = TradeManager()
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        self._validate_portfolio_config(portfolio_config)

        self.logger.info(f"Portfolio initialized with {initial_cash} {base_currency}")
        if self.verbose:
            self.logger.debug(f"Initial portfolio config: {portfolio_config}")

    def _validate_portfolio_config(self, config: Dict[str, Dict]) -> None:
        """
        Validate the portfolio configuration.

        Args:
            config (Dict[str, Dict]): The portfolio configuration to validate.

        Raises:
            ValueError: If the configuration is invalid.
        """
        required_keys = {'market_type', 'fees', 'max_trades', 'max_exposure'}
        for asset, asset_config in config.items():
            if not required_keys.issubset(asset_config.keys()):
                missing_keys = required_keys - asset_config.keys()
                error_msg = f"Invalid configuration for asset {asset}. Missing required keys: {missing_keys}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        self.logger.debug("Portfolio configuration validated successfully")

    def process_signals(self, signals: List[Dict], market_prices: Dict[str, float], timestamp: int) -> None:
        """
        Process a batch of signals, validate them, and execute trades if possible.

        Args:
            signals (List[Dict]): A list of signals to process.
            market_prices (Dict[str, float]): Current prices for each asset.
            timestamp (int): The current timestamp for when signals are processed.
        """
        self.logger.info(f"Processing {len(signals)} signals at timestamp {timestamp}")
        for signal in signals:
            try:
                if self.verbose:
                    self.logger.debug(f"Processing signal: {signal}")

                asset_name = signal['asset_name']
                action = signal['action']
                trade_amount_percentage = float(signal['amount'])
                asset_price = market_prices[asset_name]

                if self.validate_signal(signal, market_prices):
                    available_cash = self.holdings.get(self.base_currency, 0.0)
                    asset_holdings = self.holdings.get(asset_name, 0.0)

                    if action in ['buy', 'sell']:
                        if action == 'buy':
                            cash_to_use = available_cash * trade_amount_percentage
                            asset_quantity = cash_to_use / asset_price
                            base_amount = cash_to_use
                        else:  # sell
                            asset_quantity = asset_holdings * trade_amount_percentage
                            base_amount = asset_quantity * asset_price

                        self._execute_trade(signal, asset_quantity, base_amount, asset_price, timestamp)
                    elif action == 'close':
                        self._close_trade(signal, timestamp)
                    else:
                        self.logger.warning(f"Unknown action {action} in signal: {signal}")
            except KeyError as e:
                self.logger.error(f"Missing key in signal: {e}", exc_info=True)
            except Exception as e:
                self.logger.error(f"Error processing signal: {e}", exc_info=True)

        self._log_portfolio_state(timestamp)
        self._perform_portfolio_health_check(market_prices)

    def validate_signal(self, signal: Dict, market_prices: Dict[str, float]) -> bool:
        """
        Validates whether a trade can be executed based on portfolio constraints.

        Args:
            signal (Dict): The trading signal to validate.
            market_prices (Dict[str, float]): Current market prices for assets.

        Returns:
            bool: True if the signal is valid and can be executed, False otherwise.
        """
        try:
            asset = signal['asset_name']
            action = signal['action']
            amount = float(signal['amount'])
            price = market_prices[asset]
            entry_fee = self.get_fee(asset, 'entry')

            market_type = self.portfolio_config[asset]['market_type']
            max_trades = self.portfolio_config[asset].get('max_trades', float('inf'))
            max_exposure = float(self.portfolio_config[asset].get('max_exposure', 1.0))

            open_trades = self.trade_manager.get_trade(status='open')
            asset_open_trades = [trade for trade in open_trades if trade['asset_name'] == asset]

            if self.verbose:
                self.logger.debug(f"Validating signal: asset={asset}, action={action}, amount={amount}, price={price}")

            if action in ['buy', 'sell']:
                if len(asset_open_trades) >= max_trades:
                    self.logger.warning(f"Max trades limit reached for {asset}. Cannot open new trade.")
                    return False

                portfolio_value = self.get_total_value(market_prices)
                current_exposure = self.get_asset_exposure(asset, market_prices)
                new_exposure = (amount * price) / portfolio_value

                if current_exposure + new_exposure > max_exposure:
                    self.logger.warning(f"Max exposure limit reached for {asset}. Current: {current_exposure:.2%}, "
                                        f"New Trade: {new_exposure:.2%}, Max Allowed: {max_exposure:.2%}.")
                    return False

                if market_type == 'spot':
                    if action == 'buy':
                        available_cash = self.holdings.get(self.base_currency, 0.0)
                        total_cost = amount * price + entry_fee
                        if available_cash >= total_cost:
                            return True
                        else:
                            self.logger.warning(f"Not enough {self.base_currency} to buy {amount} of {asset} at {price}. "
                                                f"Required: {total_cost}, Available: {available_cash}")
                    elif action == 'sell':
                        if asset in self.holdings and self.holdings[asset] >= amount:
                            return True
                        else:
                            self.logger.warning(f"Not enough {asset} to sell. Required: {amount}, Available: {self.holdings.get(asset, 0.0)}")
                elif market_type == 'futures':
                    self.logger.info(f"Validating futures trade for {asset}. Leverage/margin rules can be implemented here.")
                    return True  # TODO: Implement futures validation logic
                else:
                    self.logger.error(f"Unsupported market type {market_type} for asset {asset}")
                    return False
            elif action == 'close':
                if not asset_open_trades:
                    self.logger.warning(f"No open trades for {asset} to close.")
                    return False
                return True

        except KeyError as e:
            self.logger.error(f"Missing key in signal or market prices: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}", exc_info=True)

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
            stop_loss = float(signal['stop_loss']) if signal.get('stop_loss') else None
            take_profit = float(signal['take_profit']) if signal.get('take_profit') else None
            trailing_stop = float(signal['trailing_stop']) if signal.get('trailing_stop') else None
            entry_reason = signal.get('reason', 'buy_signal')

            self.logger.info(f"Executing trade: {signal}")
            if self.verbose:
                self.logger.debug(f"Trade details: asset_quantity={asset_quantity}, base_amount={base_amount}, "
                                  f"asset_price={asset_price}, timestamp={timestamp}")

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

            self.update_holdings(trade)

            portfolio_value = self.get_total_value({asset_name: asset_price})
            new_exposure = self.get_asset_exposure(asset_name, {asset_name: asset_price})
            self.logger.info(f"New exposure for {asset_name}: {new_exposure:.2%} of total portfolio value.")

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}", exc_info=True)

    def _close_trade(self, signal: Dict, timestamp: int) -> None:
        """
        Close an existing trade based on the provided signal.

        Args:
            signal (Dict): The close signal containing trade details.
            timestamp (int): The timestamp of the trade closure.
        """
        try:
            asset_name = signal['asset_name']
            trade_id = signal.get('trade_id')
            exit_price = float(signal['price'])
            exit_reason = signal.get('reason', 'close_signal')

            if self.verbose:
                self.logger.debug(f"Closing trade: asset={asset_name}, trade_id={trade_id}, "
                                  f"exit_price={exit_price}, exit_reason={exit_reason}")

            if trade_id:
                trade = self.trade_manager.get_trade(trade_id=trade_id)
                if trade and trade['status'] == 'open':
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
                else:
                    self.logger.warning(f"Trade {trade_id} not found or already closed.")
            else:
                self.logger.warning(f"No trade_id provided in close signal for {asset_name}")

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

            if self.verbose:
                self.logger.debug(f"Updating holdings: asset={asset}, amount={amount}, "
                                  f"base_amount={base_amount}, entry_fee={entry_fee}")

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
                if self.verbose:
                    self.logger.debug(f"Asset: {asset}, Quantity: {quantity}, Value: {asset_value}")

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
        if self.verbose:
            self.logger.debug(f"Fee for {asset_name} ({fee_type}): {fee}")
        return fee

    def _log_portfolio_state(self, timestamp: int) -> None:
        """
        Logs the current portfolio state at a specific timestamp.

        Args:
            timestamp (int): The timestamp of the portfolio update.
        """
        self.history.append({
            'timestamp': timestamp,
            'holdings': self.holdings.copy()
        })
        self.logger.info(f"Portfolio state at {timestamp}:")
        self.logger.info(f"Cash balance: {self.holdings.get(self.base_currency, 0)} {self.base_currency}")
        for asset, amount in self.holdings.items():
            if asset != self.base_currency:
                self.logger.info(f"Asset: {asset}, Amount: {amount}")
        
        if self.verbose:
            self.logger.debug(f"Detailed portfolio state: {self.holdings}")

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

    def _perform_portfolio_health_check(self, market_prices: Dict[str, float]) -> None:
        """
        Perform a health check on the portfolio and log any potential issues.

        Args:
            market_prices (Dict[str, float]): Current market prices for assets.
        """
        self.logger.info("Performing portfolio health check...")
        total_value = self.get_total_value(market_prices)
        cash_ratio = self.holdings.get(self.base_currency, 0) / total_value

        if cash_ratio < 0.05:
            self.logger.warning(f"Low cash reserves: {cash_ratio:.2%} of portfolio value")
        elif cash_ratio > 0.5:
            self.logger.warning(f"High cash reserves: {cash_ratio:.2%} of portfolio value")

        for asset, amount in self.holdings.items():
            if asset != self.base_currency:
                exposure = self.get_asset_exposure(asset, market_prices)
                max_exposure = float(self.portfolio_config[asset].get('max_exposure', 1.0))
                if exposure > max_exposure * 0.9:
                    self.logger.warning(f"Asset {asset} approaching max exposure: {exposure:.2%} (max: {max_exposure:.2%})")

        open_trades = self.trade_manager.get_trade(status='open')
        if len(open_trades) > 10:
            self.logger.warning(f"High number of open trades: {len(open_trades)}")

        self.logger.info("Portfolio health check completed")

    def calculate_portfolio_performance(self, start_timestamp: int, end_timestamp: int, market_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate the portfolio performance over a specified time period.

        Args:
            start_timestamp (int): The start timestamp for the calculation period.
            end_timestamp (int): The end timestamp for the calculation period.
            market_prices (Dict[str, float]): Current market prices for assets.

        Returns:
            Dict[str, float]: A dictionary containing performance metrics.
        """
        start_value = next((state['holdings'] for state in self.history if state['timestamp'] >= start_timestamp), None)
        end_value = next((state['holdings'] for state in reversed(self.history) if state['timestamp'] <= end_timestamp), None)

        if not start_value or not end_value:
            self.logger.error("Unable to calculate performance: missing portfolio state")
            return {}

        start_total = sum(amount * market_prices.get(asset, 0) for asset, amount in start_value.items())
        end_total = sum(amount * market_prices.get(asset, 0) for asset, amount in end_value.items())

        absolute_return = end_total - start_total
        percent_return = (end_total / start_total - 1) * 100 if start_total > 0 else 0

        performance = {
            'absolute_return': absolute_return,
            'percent_return': percent_return,
            'start_value': start_total,
            'end_value': end_total
        }

        self.logger.info(f"Portfolio performance from {start_timestamp} to {end_timestamp}:")
        self.logger.info(f"Absolute return: {absolute_return:.2f} {self.base_currency}")
        self.logger.info(f"Percent return: {percent_return:.2f}%")

        return performance

    def rebalance_portfolio(self, target_allocations: Dict[str, float], market_prices: Dict[str, float]) -> None:
        """
        Rebalance the portfolio to match target allocations.

        Args:
            target_allocations (Dict[str, float]): Target allocation percentages for each asset.
            market_prices (Dict[str, float]): Current market prices for assets.
        """
        self.logger.info("Starting portfolio rebalance...")
        total_value = self.get_total_value(market_prices)

        for asset, target_percent in target_allocations.items():
            current_value = self.holdings.get(asset, 0) * market_prices.get(asset, 0)
            target_value = total_value * target_percent
            difference = target_value - current_value

            if abs(difference) > total_value * 0.01:  # Only rebalance if difference is more than 1% of portfolio
                if difference > 0:
                    # Buy
                    amount_to_buy = difference / market_prices[asset]
                    self.logger.info(f"Rebalancing: Buy {amount_to_buy:.4f} of {asset}")
                    # Implement buy logic here
                else:
                    # Sell
                    amount_to_sell = -difference / market_prices[asset]
                    self.logger.info(f"Rebalancing: Sell {amount_to_sell:.4f} of {asset}")
                    # Implement sell logic here

        self.logger.info("Portfolio rebalance completed")

    def __str__(self) -> str:
        """
        Return a string representation of the portfolio.

        Returns:
            str: A string representation of the portfolio.
        """
        portfolio_str = f"Portfolio in {self.base_currency}:\n"
        portfolio_str += f"Cash: {self.holdings.get(self.base_currency, 0):.2f}\n"
        for asset, amount in self.holdings.items():
            if asset != self.base_currency:
                portfolio_str += f"{asset}: {amount:.8f}\n"
        return portfolio_str
