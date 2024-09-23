import logging

class Portfolio:
    """
    Portfolio class manages asset holdings, cash, and trades for spot and futures markets.
    It handles trade execution, updates portfolio states, and processes trade signals.
    """

    def __init__(self, initial_cash: float, trade_manager, portfolio_config: dict):
        """
        Initialize the portfolio with the given initial cash balance and portfolio configurations.

        Args:
            initial_cash (float): Starting cash balance.
            trade_manager (TradeManager): An instance of the TradeManager class.
            portfolio_config (dict): Configuration for each asset (market type, fees, etc.)
                                     e.g., {'EUTEUR': {'market_type': 'spot', 'fees': {'entry': 0.001, 'exit': 0.001}}}
        """
        self.holdings = {'cash': initial_cash}
        self.history = []
        self.trade_manager = trade_manager
        self.portfolio_config = portfolio_config
        self.logger = logging.getLogger(__name__)

    def process_signals(self, signals: list, market_prices: dict, timestamp: int):
        """
        Process a batch of signals, validate them, and execute trades if possible.

        Args:
            signals (list): A list of signals to process.
            market_prices (dict): Current prices for each asset.
            timestamp (int): The current timestamp for when signals are processed.
        """
        # Prioritize sell > buy
        signals = sorted(signals, key=lambda x: x['action'] == 'sell', reverse=True)
        validated_signals = []

        # Validate all signals
        for signal in signals:
            if self.validate_signal(signal, market_prices):
                validated_signals.append(signal)
            else:
                self.logger.warning(f"Signal validation failed for signal: {signal}")

        # Execute validated trades via TradeManager
        for signal in validated_signals:
            self.logger.info(f"Executing trade for signal: {signal}")
            self._execute_trade(signal, market_prices, timestamp)

        # Log the updated portfolio state after processing the signals
        self._log_portfolio_state(timestamp)

    def validate_signal(self, signal: dict, market_prices: dict) -> bool:
        """
        Validates whether a trade can be executed based on available cash or assets.

        Args:
            signal (dict): The signal to validate. Should include 'action', 'amount', 'price', 'asset_name', and optional 'fees'.
            market_prices (dict): Current market prices of the assets.

        Returns:
            bool: True if the signal can be executed, False otherwise.
        """
        # Ensure all necessary fields are present
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

        if market_type == 'spot':
            if action == 'buy':
                # Validate cash availability for buy orders
                if self.holdings.get('cash', 0) >= (total_cost + entry_fee):
                    return True
                else:
                    self.logger.warning(f"Not enough cash to buy {amount} of {asset} at {price}. "
                                        f"Required: {total_cost + entry_fee}, Available: {self.holdings.get('cash', 0)}")
            elif action == 'sell':
                # Validate asset availability for sell orders
                if asset in self.holdings and self.holdings[asset] >= amount:
                    return True
                else:
                    self.logger.warning(f"Not enough {asset} to sell. Required: {amount}, Available: {self.holdings.get(asset, 0)}")
        elif market_type == 'futures':
            # For futures, validate availability based on leverage and margin rules (placeholder)
            self.logger.info(f"Validating futures trade for {asset}. Leverage/margin rules can be implemented here.")
            return True  # Futures validation can be expanded based on margin calculations
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

        # Log the trade details at a lower log level for debugging purposes
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

        # Update the portfolio holdings and cash
        self.update_holdings(trade)

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
                # This is a sell trade
                exit_fee = self.get_fee(asset, 'exit')
                self.holdings['cash'] += (total_cost - exit_fee)  # Add to cash after fees
                self.holdings[asset] -= amount  # Reduce asset quantity
                if self.holdings[asset] == 0:
                    del self.holdings[asset]  # Remove asset from holdings if quantity is 0
                self.logger.info(f"Updated holdings after selling {amount} of {asset}. New cash balance: {self.holdings['cash']}")
            else:
                # This is a buy trade
                self.holdings['cash'] -= (total_cost + entry_fee)  # Deduct cash
                self.holdings[asset] = self.holdings.get(asset, 0) + amount  # Add to holdings
                self.logger.info(f"Updated holdings after buying {amount} of {asset}. New cash balance: {self.holdings['cash']}")
        elif market_type == 'futures':
            # For futures, update cash and holdings differently as there's no direct asset holding
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
        total_value = self.holdings.get('cash', 0)

        for asset, quantity in self.holdings.items():
            if asset != 'cash' and asset in market_prices:
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
