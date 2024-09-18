import logging

class Portfolio:
    def __init__(self, initial_cash: float):
        self.cash_balance = initial_cash
        self.holdings = {}  # Dictionary to hold asset quantities

    def update(self, executed_order):
        """
        Updates the portfolio based on an executed order.

        Args:
            executed_order (dict): The order that was executed.
        """
        order_type = executed_order['type']
        asset_name = executed_order.get('asset_name', 'asset')  # Default to 'asset' if not specified
        amount = executed_order['amount']
        executed_price = executed_order['executed_price']

        if order_type == 'market':
            # Validate the trade before execution
            if not self._validate_trade(asset_name, amount, executed_price):
                logging.error(f"Trade validation failed for order: {executed_order}")
                return
            
            # Execute the trade
            self._execute_trade(asset_name, amount, executed_price)

    def _validate_trade(self, asset_name, amount, executed_price):
        """
        Validates if the trade can be executed.

        Args:
            asset_name (str): The name of the asset.
            amount (float): The amount to trade.
            executed_price (float): The price at which the trade is executed.

        Returns:
            bool: True if the trade is valid, False otherwise.
        """
        if amount > 0:  # Buy order
            total_cost = amount * executed_price
            if total_cost > self.cash_balance:
                logging.warning("Insufficient cash to execute buy order.")
                return False
        elif amount < 0:  # Sell order
            if asset_name not in self.holdings or abs(amount) > self.holdings[asset_name]:
                logging.warning("Insufficient holdings to execute sell order.")
                return False
        return True

    def _execute_trade(self, asset_name, amount, executed_price):
        """
        Executes a trade by updating the cash balance and holdings.

        Args:
            asset_name (str): The name of the asset.
            amount (float): The amount to trade.
            executed_price (float): The price at which the trade is executed.
        """
        if amount > 0:  # Buy order
            total_cost = amount * executed_price
            self.cash_balance -= total_cost
            self._add_to_holdings(asset_name, amount)
        elif amount < 0:  # Sell order
            total_sale = abs(amount) * executed_price
            self.cash_balance += total_sale
            self._remove_from_holdings(asset_name, abs(amount))

    def _add_to_holdings(self, asset_name, amount):
        """
        Adds the asset to holdings.

        Args:
            asset_name (str): The name of the asset.
            amount (float): The amount to add to the holdings.
        """
        if asset_name in self.holdings:
            self.holdings[asset_name] += amount
        else:
            self.holdings[asset_name] = amount

    def _remove_from_holdings(self, asset_name, amount):
        """
        Removes the asset from holdings.

        Args:
            asset_name (str): The name of the asset.
            amount (float): The amount to remove from the holdings.
        """
        if asset_name in self.holdings:
            self.holdings[asset_name] -= amount
            if self.holdings[asset_name] <= 0:
                del self.holdings[asset_name]  # Remove the asset if amount becomes zero or negative

    def get_portfolio_value(self, market_prices: dict):
        """
        Calculates the current portfolio value.

        Args:
            market_prices (dict): Dictionary of current prices for each asset.

        Returns:
            float: The total portfolio value.
        """
        total_value = self.cash_balance
        for asset, quantity in self.holdings.items():
            if asset in market_prices:
                total_value += market_prices[asset] * quantity

        return total_value

    def get_pnl(self, initial_cash: float, market_prices: dict):
        """
        Calculates the profit and loss (PnL) of the portfolio.

        Args:
            initial_cash (float): The initial cash to calculate PnL.
            market_prices (dict): Dictionary of current prices for each asset.

        Returns:
            float: The PnL value.
        """
        current_value = self.get_portfolio_value(market_prices)
        return current_value - initial_cash
