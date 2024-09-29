import pandas as pd
import numpy as np
from src.logger import setup_logger
from typing import Dict, List

class MetricsModule:
    def __init__(self, portfolio, base_currency='USD', risk_free_rate=0.01):
        self.portfolio = portfolio
        self.base_currency = base_currency
        self.risk_free_rate = risk_free_rate
        self.logger = setup_logger('metrics')

    def create_portfolio_value_series(self, market_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        Create a time-indexed series of portfolio values.
        
        Args:
            market_data (Dict[str, pd.DataFrame]): Market data for each asset.
        
        Returns:
            pd.Series: Time-indexed series of portfolio values.
        """
        portfolio_values = []
        for snapshot in self.portfolio.history:
            timestamp = snapshot['timestamp']
            holdings = snapshot['holdings']
            total_value = holdings.get(self.base_currency, 0.0)
            for asset, quantity in holdings.items():
                if asset != self.base_currency and asset in market_data:
                    asset_price = market_data[asset].loc[market_data[asset].index <= timestamp, 'close'].iloc[-1]
                    total_value += quantity * asset_price
            portfolio_values.append((pd.to_datetime(timestamp, unit='ms'), total_value))
        return pd.Series(dict(portfolio_values))

    def calculate_daily_returns(self, market_data: Dict[str, pd.DataFrame]) -> pd.Series:
        portfolio_values = self.create_portfolio_value_series(market_data)
        daily_portfolio_values = portfolio_values.resample('D').last()
        daily_returns = daily_portfolio_values.pct_change(fill_method=None).dropna()
        return daily_returns

    def calculate_sharpe_ratio(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate the Sharpe Ratio for the portfolio.
        
        Args:
            market_data (Dict[str, pd.DataFrame]): Market data for each asset.
        
        Returns:
            float: The Sharpe Ratio.
        """
        daily_returns = self.calculate_daily_returns(market_data)
        excess_returns = daily_returns - self.risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe_ratio

    def calculate_max_drawdown(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate the Maximum Drawdown for the portfolio.
        
        Args:
            market_data (Dict[str, pd.DataFrame]): Market data for each asset.
        
        Returns:
            float: The Maximum Drawdown.
        """
        portfolio_values = self.create_portfolio_value_series(market_data)
        cumulative_max = portfolio_values.cummax()
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max
        return drawdowns.min()

    def calculate_cumulative_return(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate the Cumulative Return for the portfolio.
        
        Args:
            market_data (Dict[str, pd.DataFrame]): Market data for each asset.
        
        Returns:
            float: The Cumulative Return.
        """
        portfolio_values = self.create_portfolio_value_series(market_data)
        return (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]


    def calculate_profit_factor(self, trades: List[Dict]) -> float:
        if not trades:
            return np.nan  # or raise an exception

        total_profit = sum(trade['profit'] for trade in trades if trade['profit'] > 0)
        total_loss = sum(abs(trade['profit']) for trade in trades if trade['profit'] < 0)
        
        if total_loss == 0:
            return np.inf if total_profit > 0 else np.nan

        return total_profit / total_loss
    
    def calculate_win_rate(self, trades: List[Dict]) -> float:
        if not trades:
            return np.nan  # or raise an exception

        winning_trades = sum(1 for trade in trades if trade['profit'] > 0)
        return (winning_trades / len(trades)) * 100
    
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

    def calculate_trade_summary(self, trades: List[Dict], market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calculate a summary of trades executed during the backtest and portfolio performance.
        
        Args:
            trades (List[Dict]): List of executed trades.
            market_data (Dict[str, pd.DataFrame]): Market data for assets.
        
        Returns:
            Dict: A dictionary containing summary statistics or an error message if no trades were executed.
        """
        self.logger.debug(f"Calculating trade summary. Total trades: {len(trades)}")
        if not trades:
            return {"error": "No trades were executed during the backtest."}

        portfolio_values = self.create_portfolio_value_series(market_data)
        
        total_profit = 0
        total_fees = 0
        winning_trades = 0
        losing_trades = 0
        stop_loss_trades = 0
        take_profit_trades = 0
        total_holding_time = 0

        for trade in trades:
            profit = (trade['exit_price'] - trade['entry_price']) * trade['asset_amount']
            profit -= (trade['entry_fee'] + trade['exit_fee'])
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
        final_portfolio_value = portfolio_values.iloc[-1]
        initial_portfolio_value = portfolio_values.iloc[0]
        total_percent_increase = ((final_portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100

        # Calculate average percent per month
        total_days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        avg_percent_per_month = (total_percent_increase / total_days) * 30

        return {
            "total_trades": len(trades),
            "total_profit": total_profit,
            "total_fees": total_fees,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "stop_loss_trades": stop_loss_trades,
            "take_profit_trades": take_profit_trades,
            "average_holding_time": avg_holding_time / (1000 * 60),  # Convert to minutes
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