import logging

class Portfolio:
    """
    Portfolio class manages asset holdings, cash, and trades for spot and futures markets.
    It handles trade execution, updates portfolio states, and processes trade signals.
    """

    def __init__(self, initial_cash: float, trade_manager, portfolio_config: dict, base_currency: str):
        """
        Initialize the portfolio with the given initial cash balance, base currency, and portfolio configurations.

        Args:
            initial_cash (float): Starting cash balance in the base currency.
            trade_manager (TradeManager): An instance of the TradeManager class.
            portfolio_config (dict): Configuration for each asset (market type, fees, max exposure, etc.)
                                     e.g., {'EUT': {'market_type': 'spot', 'fees': {'entry': 0.001, 'exit': 0.001}, 'max_exposure': 0.3, 'max_trades': 5}}.
            base_currency (str): The base currency for the portfolio (e.g., 'EUR', 'USD').
        """
        # Validate that all assets use the same base currency
        self._validate_base_currency(portfolio_config, base_currency)

        self.holdings = {base_currency: initial_cash}  # Cash is tracked under the base currency (e.g., 'EUR' or 'USD')
        self.history = []  # To store historical snapshots of portfolio state
        self.trade_manager = trade_manager  # Handles trade execution
        self.portfolio_config = portfolio_config  # Asset-specific configurations
        self.base_currency = base_currency  # The currency in which the cash is held
        self.logger = logging.getLogger(__name__)

    def _validate_base_currency(self, portfolio_config: dict, base_currency: str):
        """
        Validates that all assets in the portfolio use the same base currency.

        Args:
            portfolio_config (dict): The configuration for each asset.
            base_currency (str): The base currency to validate against.

        Raises:
            ValueError: If any asset uses a different base currency.
        """
        for asset, config in portfolio_config.items():
            asset_currency = config.get('currency', base_currency)
            if asset_currency != base_currency:
                raise ValueError(f"Asset {asset} uses {asset_currency}, but portfolio is set to {base_currency}.")

    def process_signals(self, signals: list, market_prices: dict, timestamp: int):
        """
        Process a batch of signals, validate them, and execute trades if possible.

        Args:
            signals (list): A list of signals to process.
            market_prices (dict): Current prices for each asset.
            timestamp (int): The current timestamp for when signals are processed.
        """
        # Prioritize sell signals over buy signals
        signals = sorted(signals, key=lambda x: x['action'] == 'sell', reverse=True)
        validated_signals = []

        # Validate each signal
        for signal in signals:
            if self.validate_signal(signal, market_prices):
                validated_signals.append(signal)
            else:
                self.logger.warning(f"Signal validation failed for signal: {signal}")

        # Execute each validated trade
        for signal in validated_signals:
            self.logger.info(f"Executing trade for signal: {signal}")
            self._execute_trade(signal, market_prices, timestamp)

        # Log the updated portfolio state after processing the signals
        self._log_portfolio_state(timestamp)

    def validate_signal(self, signal: dict, market_prices: dict) -> bool:
        """
        Validates whether a trade can be executed based on available cash, max trades, and max exposure.

        Args:
            signal (dict): The signal to validate. Should include 'action', 'amount', 'price', 'asset_name', and optional 'fees'.
            market_prices (dict): Current market prices of the assets.

        Returns:
            bool: True if the signal can be executed, False otherwise.
        """
        # Ensure all necessary fields are present in the signal
        required_fields = ['action', 'amount', 'asset_name', 'price']
        if not all(field in signal for field in required_fields):
            self.logger.error(f"Signal is missing required fields: {signal}")
            return False

        asset = signal['asset_name']
        action = signal['action']
        amount = signal['amount']
        price = signal['price']
        entry_fee = self.get_fee(asset, 'entry')
        total_cost = amount * price

        market_type = self.portfolio_config[asset]['market_type']
        max_trades = self.portfolio_config[asset].get('max_trades', float('inf'))
        max_exposure = self.portfolio_config[asset].get('max_exposure', 1.0)  # Default to 100% exposure if not set

        # Validate the number of open trades for the asset
        open_trades = self.trade_manager.get_trade(status='open')
        asset_open_trades = [trade for trade in open_trades if trade['asset_name'] == asset]

        if len(asset_open_trades) >= max_trades:
            self.logger.warning(f"Max trades limit reached for {asset}. Cannot open new trade.")
            return False

        # Validate the exposure of the portfolio to the asset
        portfolio_value = self.get_total_value(market_prices)
        current_exposure = (self.holdings.get(asset, 0) * market_prices[asset]) / portfolio_value if asset in self.holdings else 0
        new_exposure = total_cost / portfolio_value

        if current_exposure + new_exposure > max_exposure:
            self.logger.warning(f"Max exposure limit reached for {asset}. Current: {current_exposure:.2%}, "
                                f"New Trade: {new_exposure:.2%}, Max Allowed: {max_exposure:.2%}.")
            return False

        # Validate cash availability for spot market buy trades
        if market_type == 'spot':
            if action == 'buy':
                if self.holdings.get(self.base_currency, 0) >= total_cost + entry_fee:
                    return True
                else:
                    self.logger.warning(f"Not enough {self.base_currency} to buy {amount} of {asset} at {price}. "
                                        f"Required: {total_cost + entry_fee}, Available: {self.holdings.get(self.base_currency, 0)}")
            elif action == 'sell':
                if asset in self.holdings and self.holdings[asset] >= amount:
                    return True
                else:
                    self.logger.warning(f"Not enough {asset} to sell. Required: {amount}, Available: {self.holdings.get(asset, 0)}")
        elif market_type == 'futures':
            self.logger.info(f"Validating futures trade for {asset}. Leverage/margin rules can be implemented here.")
            return True  # Placeholder for futures validation logic
        else:
            self.logger.error(f"Unsupported market type {market_type} for {asset}")
            return False

        return False

    def _execute_trade(self, signal: dict, market_prices: dict, timestamp: int):
        """
        Execute a validated trade and update holdings.

        Args:
            signal (dict): The validated trade signal.
            market_prices (dict): Current market prices of assets.
            timestamp (int): The timestamp of trade execution.
        """
        asset_name = signal['asset_name']
        amount = signal['amount']
        entry_price = market_prices[asset_name]
        entry_fee = self.get_fee(asset_name, 'entry')
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        trailing_stop = signal.get('trailing_stop')
        entry_reason = signal.get('reason', 'buy_signal')

        # Log trade execution details
        self.logger.debug(f"Preparing to execute trade: {signal}")

        # Open the trade using TradeManager
        trade = self.trade_manager.open_trade(
            asset_name=asset_name,
            amount=amount,
            entry_price=entry_price,
            entry_timestamp=timestamp,
            entry_fee=entry_fee,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            entry_reason=entry_reason
        )

        # Update portfolio holdings after the trade is executed
        self.update_holdings(trade)

        # Log the new exposure for the asset after the trade
        portfolio_value = self.get_total_value(market_prices)
        new_exposure = (self.holdings.get(asset_name, 0) * market_prices[asset_name]) / portfolio_value if asset_name in self.holdings else 0
        self.logger.info(f"New exposure for {asset_name}: {new_exposure:.2%} of total portfolio value.")

    def update_holdings(self, trade: dict):
        """
        Updates portfolio holdings and cash after a trade is executed.

        Args:
            trade (dict): The trade details (from TradeManager). Includes 'asset_name', 'amount', 'price', 'fees', etc.
        """
        asset = trade['asset_name']
        amount = trade['amount']
        price = trade['entry_price']
        entry_fee = trade.get('entry_fee', 0)

        total_cost = amount * price
        market_type = self.portfolio_config[asset]['market_type']

        if market_type == 'spot':
            if trade['status'] == 'closed':
                # Update holdings for a sell trade
                exit_fee = self.get_fee(asset, 'exit')
                self.holdings[self.base_currency] += (total_cost - exit_fee)
                self.holdings[asset] -= amount
                if self.holdings[asset] == 0:
                    del self.holdings[asset]
                self.logger.info(f"Updated holdings after selling {amount} of {asset}. New {self.base_currency} balance: {self.holdings[self.base_currency]}")
            else:
                # Update holdings for a buy trade
                self.holdings[self.base_currency] -= (total_cost + entry_fee)
                self.holdings[asset] = self.holdings.get(asset, 0) + amount
                self.logger.info(f"Updated holdings after buying {amount} of {asset}. New {self.base_currency} balance: {self.holdings[self.base_currency]}")
        elif market_type == 'futures':
            self.logger.info(f"Futures trade executed for {asset}, amount: {amount}. No direct asset holding updated.")
        else:
            self.logger.error(f"Unsupported market type {market_type} for asset {asset}")

    def _log_portfolio_state(self, timestamp: int):
        """
        Logs the current portfolio state (cash + asset quantities) at a specific timestamp.

        Args:
            timestamp (int): The timestamp of the portfolio update.
        """
        self.history.append({
            'timestamp': timestamp,
            'holdings': self.holdings.copy()  # Store a snapshot of the current holdings
        })
        self.logger.info(f"Portfolio state at {timestamp}: {self.holdings}")

    def get_total_value(self, market_prices: dict) -> float:
        """
        Calculates the total value of the portfolio (cash + current asset values).

        Args:
            market_prices (dict): Dictionary of current prices for each asset.

        Returns:
            float: The total portfolio value.
        """
        total_value = self.holdings.get(self.base_currency, 0)  # Use base currency cash balance

        for asset, quantity in self.holdings.items():
            if asset != self.base_currency and asset in market_prices:
                total_value += quantity * market_prices[asset]

        return total_value

    def get_holdings(self) -> dict:
        """
        Get a snapshot of the current holdings including cash.

        Returns:
            dict: A dictionary of holdings {'cash': 1000, 'BTC': 2.0, ...}
        """
        return self.holdings.copy()

    def log_portfolio(self) -> list:
        """
        Returns the historical portfolio log for analysis.

        Returns:
            list: The list of portfolio states over time [{'timestamp': 12345, 'holdings': {...}}, ...]
        """
        return self.history

    def get_fee(self, asset_name: str, fee_type: str = 'entry') -> float:
        """
        Get the fee for the specified asset.

        Args:
            asset_name (str): The asset being traded (e.g., 'EUTEUR').
            fee_type (str): Either 'entry' or 'exit'.

        Returns:
            float: The fee percentage for the given asset and type.
        """
        return self.portfolio_config.get(asset_name, {}).get('fees', {}).get(fee_type, 0.0)
