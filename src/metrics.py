import numpy as np
import pandas as pd
import logging

class MetricsModule:
    """
    A module to compute financial performance metrics for a trading strategy.
    This includes Sharpe Ratio, Max Drawdown, Cumulative Return, Profit Factor, and Win Rate.
    """

    def __init__(self, base_currency='USD', risk_free_rate=0.01):
        """
        Initialize the MetricsModule.

        Args:
            risk_free_rate (float): The risk-free rate used for the Sharpe Ratio calculation.
        """
        self.risk_free_rate = risk_free_rate
        self.base_currency = base_currency
        self.logger = logging.getLogger(__name__)

    def calculate_sharpe_ratio(self, returns):
        """
        Calculate the Sharpe Ratio for a series of portfolio returns.

        Args:
            returns (pd.Series): A pandas Series representing portfolio returns.

        Returns:
            float: The Sharpe Ratio.
        """
        excess_returns = returns - self.risk_free_rate / 252  # Adjust risk-free rate for daily returns
        avg_excess_return = excess_returns.mean()
        std_dev = excess_returns.std()

        if std_dev == 0:
            return np.nan  # To avoid division by zero

        sharpe_ratio = avg_excess_return / std_dev
        return sharpe_ratio

    def calculate_max_drawdown(self, portfolio_values):
        """
        Calculate the Maximum Drawdown for a series of portfolio values.

        Args:
            portfolio_values (pd.Series): A pandas Series representing portfolio values over time.

        Returns:
            float: The Maximum Drawdown.
        """
        cumulative_max = portfolio_values.cummax()
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max
        max_drawdown = drawdowns.min()
        return max_drawdown

    def calculate_cumulative_return(self, portfolio_values):
        """
        Calculate the Cumulative Return for a series of portfolio values.

        Args:
            portfolio_values (pd.Series): A pandas Series representing portfolio values over time.

        Returns:
            float: The Cumulative Return.
        """
        start_value = portfolio_values.iloc[0]
        end_value = portfolio_values.iloc[-1]
        cumulative_return = (end_value - start_value) / start_value
        return cumulative_return

    def calculate_profit_factor(self, trades):
        """
        Calculate the Profit Factor from a series of executed trades.

        Args:
            trades (list): A list of dictionaries containing trade details.

        Returns:
            float: The Profit Factor, calculated as total profit divided by total loss.
        """
        total_profit = 0
        total_loss = 0

        for trade in trades:
            if trade['status'] == 'closed':
                profit = (trade['exit_price'] - trade['entry_price']) * trade['amount']
                profit -= (trade['entry_fee'] + trade['exit_fee'])  # Subtract fees

                if profit > 0:
                    total_profit += profit
                else:
                    total_loss += abs(profit)

        if total_loss == 0:
            return np.inf  # To avoid division by zero and indicate no losses

        profit_factor = total_profit / total_loss
        return profit_factor

    def calculate_win_rate(self, trades):
        """
        Calculate the Win Rate from a series of executed trades.

        Args:
            trades (list): A list of dictionaries containing trade details.

        Returns:
            float: The Win Rate as a percentage.
        """
        winning_trades = 0
        total_trades = 0

        for trade in trades:
            if trade['status'] == 'closed':
                profit = (trade['exit_price'] - trade['entry_price']) * trade['amount']
                total_trades += 1

                if profit > 0:
                    winning_trades += 1

        if total_trades == 0:
            return np.nan  # No trades executed

        win_rate = (winning_trades / total_trades) * 100
        return win_rate

    def calculate_portfolio_values(self, history, market_prices):
        """
        Calculate the portfolio values at each timestamp using the market prices of assets.

        Args:
            history (list): List of portfolio snapshots, each containing holdings at a given timestamp.
            market_prices (dict): Dictionary of asset prices at each timestamp.

        Returns:
            pd.Series: Portfolio values over time indexed by timestamp.
        """
        portfolio_values = []

        for snapshot in history:
            timestamp = snapshot['timestamp']
            holdings = snapshot['holdings']
            
            # Calculate the total value of holdings (base currency + assets)
            total_value = holdings.get('cash', 0)  # Base currency cash balance
            
            for asset, quantity in holdings.items():
                if asset != 'cash':  # Skip the base currency
                    try:
                        # Retrieve the closest available timestamp for the asset price
                        closest_timestamp = self.get_closest_timestamp(market_prices[asset], timestamp)
                        
                        if closest_timestamp is not None:
                            asset_price = market_prices[asset].loc[closest_timestamp]
                        else:
                            self.logger.warning(f"No price data for {asset} at timestamp {timestamp}. Using previous available price.")
                            continue
                        
                        total_value += quantity * asset_price
                    except KeyError:
                        self.logger.error(f"Price data missing for asset {asset} at timestamp {timestamp}.")
            
            portfolio_values.append({'timestamp': timestamp, 'total_value': total_value})

        # Convert to pandas Series for easier manipulation
        portfolio_values_df = pd.DataFrame(portfolio_values).set_index('timestamp')['total_value']
        return portfolio_values_df

    def get_closest_timestamp(self, data, target_timestamp):
        """
        Retrieve the closest timestamp before the target timestamp from the DataFrame.

        Args:
            data (pd.DataFrame): The historical market data (indexed by timestamp).
            target_timestamp (int): The target timestamp to search for.

        Returns:
            int: The closest available timestamp less than or equal to the target, or None if not found.
        """
        available_timestamps = data.index
        closest_timestamp = available_timestamps[available_timestamps <= target_timestamp].max()
        return closest_timestamp

    def calculate_trade_summary(self, trades, portfolio_history, market_data, slippage=0.0):
        """
        Calculate a summary of trades executed during the backtest and portfolio performance.

        Args:
            trades (list): List of executed trades.
            portfolio_history (list): List of portfolio snapshots over time [{'timestamp': ..., 'holdings': {...}}, ...].
            market_data (dict): Aligned market data for assets.
            slippage (float): Slippage applied to trades.

        Returns:
            dict: A dictionary containing summary statistics including total profit, portfolio value, win rate, 
                average holding time, and performance metrics.
        """
        if not trades:
            return {
                "total_trades": 0,
                "total_profit": 0,
                "total_fees": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "stop_loss_trades": 0,
                "take_profit_trades": 0,
                "average_holding_time": 0,
                "final_portfolio_value": 0,
                "total_percent_increase": 0,
                "avg_percent_per_month": 0
            }

        total_profit = 0
        total_fees = 0
        winning_trades = 0
        losing_trades = 0
        stop_loss_trades = 0
        take_profit_trades = 0
        total_holding_time = 0

        for trade in trades:
            market_type = trade.get('market_type', 'spot')  # Default to 'spot' if not set
            leverage = trade.get('leverage', 1) if market_type == 'futures' else 1  # Apply leverage for futures

            # Adjust exit price with slippage
            slippage_adjusted_exit_price = trade['exit_price'] * (1 - slippage)
            profit = ((slippage_adjusted_exit_price - trade['entry_price']) * trade['amount'] * leverage) - trade['entry_fee'] - trade['exit_fee']
            total_profit += profit
            total_fees += trade['entry_fee'] + trade['exit_fee']

            if profit > 0:
                winning_trades += 1
            else:
                losing_trades += 1

            if trade['exit_reason'] == "stop_loss":
                stop_loss_trades += 1
            elif trade['exit_reason'] == "take_profit":
                take_profit_trades += 1

            holding_time = trade['exit_timestamp'] - trade['entry_timestamp']
            total_holding_time += holding_time

        avg_holding_time = total_holding_time / len(trades) if trades else 0

        # Final Portfolio Value
        final_portfolio_value = self.calculate_portfolio_value_at_end(portfolio_history, market_data)

        # Initial Portfolio Value
        initial_portfolio_value = portfolio_history[0]['holdings'].get(self.base_currency, None)
        
        # Ensure the initial portfolio value is valid
        if initial_portfolio_value is None or initial_portfolio_value == 0:
            self.logger.warning("Initial portfolio value is zero or not found. Skipping percent increase calculation.")
            total_percent_increase = 0
            avg_percent_per_month = 0
        else:
            # Total Percentage Increase
            total_percent_increase = ((final_portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100

            # Average Percent Per Month
            total_duration_in_days = (portfolio_history[-1]['timestamp'] - portfolio_history[0]['timestamp']) / (1000 * 60 * 60 * 24)  # Convert to days
            avg_percent_per_month = (total_percent_increase / total_duration_in_days) * 30  # Adjust for monthly return

        return {
            "total_trades": len(trades),
            "total_profit": total_profit,
            "total_fees": total_fees,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "stop_loss_trades": stop_loss_trades,
            "take_profit_trades": take_profit_trades,
            "average_holding_time": avg_holding_time / 60000,  # Convert milliseconds to minutes
            "final_portfolio_value": final_portfolio_value,
            "total_percent_increase": total_percent_increase,
            "avg_percent_per_month": avg_percent_per_month
        }



    def calculate_portfolio_value_at_end(self, portfolio_history, market_data):
        """
        Calculate the final portfolio value by evaluating asset values at the end of the backtest.

        Args:
            portfolio_history (list): List of portfolio snapshots over time.
            market_data (dict): Historical market data for assets.

        Returns:
            float: The final portfolio value.
        """
        last_snapshot = portfolio_history[-1]
        last_market_prices = {asset: data.iloc[-1]['close'] for asset, data in market_data.items()}  # Get last close prices
        final_portfolio_value = last_snapshot['holdings'].get('cash', 0)  # Base currency balance

        for asset, quantity in last_snapshot['holdings'].items():
            if asset != 'cash' and asset in last_market_prices:
                final_portfolio_value += quantity * last_market_prices[asset]

        return final_portfolio_value