import random
import logging

class ExecutionModel:
    def __init__(self, slippage: float = 0.0, latency: int = 0):
        if not (0.0 <= slippage <= 1.0):
            raise ValueError("Slippage must be between 0.0 and 1.0")
        if not isinstance(latency, int) or latency < 0:
            raise ValueError("Latency must be a non-negative integer")
        
        self.slippage = slippage
        self.latency = latency
        self.active_limit_orders = []
        self.active_stop_orders = []
        self.active_trailing_stop_orders = []
        self.executed_orders = []  # Store executed orders for analysis
        self.trade_id_counter = 0  # Counter to assign unique trade IDs

    def place_order(self, order_type: str, amount: float, **kwargs):
        timestamp = kwargs.get('timestamp')
        try:
            if order_type == 'market':
                return self._place_market_order(amount, kwargs.get('current_price'), timestamp)
            elif order_type == 'limit':
                return self._place_limit_order(amount, kwargs.get('price'), timestamp)
            elif order_type == 'stop':
                return self._place_stop_order(amount, kwargs.get('stop_price'), timestamp)
            elif order_type == 'trailing_stop':
                return self._place_trailing_stop_order(amount, kwargs.get('trailing_amount'), kwargs.get('current_price'), timestamp)
            else:
                logging.error(f"Unsupported order type: {order_type}")
                return None
        except Exception as e:
            logging.error(f"Error placing {order_type} order: {e}")
            return None

    def _get_market_price_with_slippage(self, current_price: float, future_prices: list, step: int = 1):
        """
        Adjusts the current price with slippage by considering a future price.
        This simulates slippage in a backtesting environment by using a future price.

        Args:
            current_price (float): The current market price.
            future_prices (list): List of future prices for simulating slippage.
            step (int): Number of steps to look into the future for the price.

        Returns:
            float: The price adjusted for slippage.
        """
        if current_price <= 0:
            raise ValueError("Current price must be a positive value.")
        
        # Use future prices to simulate slippage
        future_price = future_prices[min(step, len(future_prices)-1)]
        slippage_adjustment = future_price * self.slippage * random.uniform(-1, 1)
        return future_price + slippage_adjustment

    def _create_order(self, order_type, amount, executed_price=None, **kwargs):
        timestamp = kwargs.get('timestamp')
        if timestamp is None:
            raise ValueError("Timestamp is required for order creation.")
        
        # Assign a unique trade ID for each order
        self.trade_id_counter += 1

        order = {
            'id': self.trade_id_counter,  # Unique trade ID
            'type': order_type,
            'amount': amount,
            'executed_price': executed_price,
            'timestamp': timestamp,
            'status': kwargs.get('status', 'executed'),
            'stop_price': kwargs.get('stop_price'),
            'trail_amount': kwargs.get('trail_amount'),
        }
        return order

    def _place_market_order(self, amount: float, current_price: float, timestamp):
        if current_price is None:
            logging.error("Missing current price for market order.")
            return None

        # Simulate slippage using a future price (this could be passed as an argument)
        future_prices = [current_price * (1 + random.uniform(-0.01, 0.01)) for _ in range(10)]  # Example of future prices
        executed_price = self._get_market_price_with_slippage(current_price, future_prices)
        logging.info(f"Market order executed: Amount = {amount}, Executed Price = {executed_price:.2f}")
        order = self._create_order('market', amount, executed_price, status='executed', timestamp=timestamp)
        self.executed_orders.append(order)  # Store the executed order
        return order

    def _place_limit_order(self, amount: float, price: float, timestamp):
        if price is None:
            logging.error("Missing price for limit order.")
            return None

        limit_order = self._create_order('limit', amount, price, status='active', timestamp=timestamp)
        self.active_limit_orders.append(limit_order)
        logging.info(f"Limit order placed: Amount = {amount}, Price = {price:.2f}")
        return limit_order

    def _place_stop_order(self, amount: float, stop_price: float, timestamp):
        if stop_price is None:
            logging.error("Missing stop price for stop order.")
            return None

        stop_order = self._create_order('stop', amount, stop_price, status='active', stop_price=stop_price, timestamp=timestamp)
        self.active_stop_orders.append(stop_order)
        logging.info(f"Stop order placed: Amount = {amount}, Stop Price = {stop_price:.2f}")
        return stop_order

    def _place_trailing_stop_order(self, amount: float, trailing_amount: float, current_price: float, timestamp):
        if trailing_amount is None or current_price is None:
            logging.error("Missing parameters for trailing stop order.")
            return None

        stop_price = current_price - trailing_amount if amount > 0 else current_price + trailing_amount
        trailing_stop_order = self._create_order('trailing_stop', amount, stop_price, status='active', trail_amount=trailing_amount, timestamp=timestamp)
        self.active_trailing_stop_orders.append(trailing_stop_order)
        logging.info(f"Trailing stop order placed: Amount = {amount}, Trail Amount = {trailing_amount:.2f}, Initial Stop Price = {stop_price:.2f}")
        return trailing_stop_order

    def match_limit_orders(self, current_price: float):
        self._match_orders(self.active_limit_orders, current_price, order_type='limit')

    def match_stop_orders(self, current_price: float):
        self._match_orders(self.active_stop_orders, current_price, order_type='stop')

    def match_trailing_stop_orders(self, current_price: float):
        for order in self.active_trailing_stop_orders:
            if order['status'] == 'active':
                new_stop_price = (current_price - order['trail_amount']) if order['amount'] > 0 else (current_price + order['trail_amount'])
                if (order['amount'] > 0 and new_stop_price > order['stop_price']) or \
                   (order['amount'] < 0 and new_stop_price < order['stop_price']):
                    order['stop_price'] = new_stop_price

                if (order['amount'] > 0 and current_price <= order['stop_price']) or \
                   (order['amount'] < 0 and current_price >= order['stop_price']):
                    order['status'] = 'executed'
                    order['executed_price'] = current_price
                    order['execution_timestamp'] = order['timestamp']
                    logging.info(f"Trailing stop order executed: ID = {order['id']}, Amount = {order['amount']}, Executed Price = {current_price:.2f}")
                    self.executed_orders.append(order)  # Store the executed order

    def _match_orders(self, orders_list, current_price, order_type):
        for order in orders_list:
            if order['status'] == 'active':
                if (order['amount'] > 0 and current_price <= order['price']) or \
                   (order['amount'] < 0 and current_price >= order['price']):
                    order['status'] = 'executed'
                    order['executed_price'] = current_price
                    order['execution_timestamp'] = order['timestamp']
                    logging.info(f"{order_type.capitalize()} order executed: ID = {order['id']}, Amount = {order['amount']}, Executed Price = {current_price:.2f}")
                    self.executed_orders.append(order)  # Store the executed order

    def remove_executed_orders(self):
        self.active_limit_orders = [order for order in self.active_limit_orders if order['status'] == 'active']
        self.active_stop_orders = [order for order in self.active_stop_orders if order['status'] == 'active']
        self.active_trailing_stop_orders = [order for order in self.active_trailing_stop_orders if order['status'] == 'active']
