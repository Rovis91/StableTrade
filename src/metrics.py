import pandas as pd
import numpy as np
import json
from src.logger import setup_logger
from typing import Dict, List, Any, Optional


class MetricsModule:
    """
    A module to compute key metrics and performance statistics for a trading portfolio.

    The MetricsModule provides various methods for calculating Sharpe Ratio, maximum drawdown,
    cumulative returns, and trade summaries. It utilizes portfolio history and market data to 
    derive these metrics.

    Attributes:
        portfolio: The portfolio instance used to calculate historical values.
        base_currency (str): The base currency of the portfolio (e.g., 'USD').
        risk_free_rate (float): The risk-free rate used for calculating excess returns.
        logger (logging.Logger): Logger instance for logging metrics operations.
    """

    STOP_LOSS_REASON = 'stop_loss'
    TAKE_PROFIT_REASON = 'take_profit'
    CASH = 'cash'
    CLOSE = 'close'

    def __init__(self, portfolio, base_currency: str = 'USD', risk_free_rate: float = 0.01, logger: Optional[Any] = None):
        """
        Initialize the MetricsModule with the given parameters.

        Args:
            portfolio: The portfolio instance.
            base_currency (str): The base currency of the portfolio (default: 'USD').
            risk_free_rate (float): The risk-free rate (default: 0.01).
            logger (Optional[logging.Logger]): Logger instance for logging (optional).
        """
        self.portfolio = portfolio
        self.base_currency = base_currency
        self.risk_free_rate = risk_free_rate
        self.logger = logger if logger else setup_logger('metrics')

    def create_portfolio_value_series(self, market_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        Create a time-indexed series of portfolio values.
        
        This method iterates over the portfolio history and computes the total portfolio value at 
        each timestamp, including asset values based on market data.

        Args:
            market_data (Dict[str, pd.DataFrame]): Dictionary mapping asset names to their market data DataFrames.
        
        Returns:
            pd.Series: Time-indexed series representing portfolio values at each timestamp.
        
        Example:
            portfolio_value_series = metrics_module.create_portfolio_value_series(market_data)
        """
        portfolio_values = []
        for snapshot in self.portfolio.history:
            timestamp = snapshot['timestamp']
            holdings = snapshot['holdings']
            total_value = holdings.get(self.base_currency, 0.0)
            for asset, quantity in holdings.items():
                if asset != self.base_currency and asset in market_data:
                    closest_timestamp = self.get_closest_timestamp(market_data[asset], timestamp)
                    asset_price = market_data[asset].loc[closest_timestamp, self.CLOSE] if closest_timestamp else 0.0
                    total_value += quantity * asset_price
            portfolio_values.append((pd.to_datetime(timestamp, unit='ms'), total_value))
        return pd.Series(dict(portfolio_values))

    def calculate_daily_returns(self, market_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        Calculate daily returns of the portfolio.

        Args:
            market_data (Dict[str, pd.DataFrame]): Market data for each asset.
        
        Returns:
            pd.Series: Daily percentage change in portfolio value.
        """
        portfolio_values = self.create_portfolio_value_series(market_data)
        daily_portfolio_values = portfolio_values.resample('D').last()
        daily_returns = daily_portfolio_values.pct_change().dropna()
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
        """
        Calculate the profit factor for the executed trades.

        Args:
            trades (List[Dict]): List of executed trades.

        Returns:
            float: The profit factor.
        """
        if not trades:
            return np.nan

        total_profit = sum(trade['profit'] for trade in trades if trade['profit'] > 0)
        total_loss = sum(abs(trade['profit']) for trade in trades if trade['profit'] < 0)
        
        if total_loss == 0:
            return np.inf if total_profit > 0 else np.nan

        return total_profit / total_loss

    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """
        Calculate the win rate for the executed trades.

        Args:
            trades (List[Dict]): List of executed trades.

        Returns:
            float: The win rate as a percentage.
        """
        if not trades:
            return np.nan

        winning_trades = sum(1 for trade in trades if trade['profit'] > 0)
        return (winning_trades / len(trades)) * 100

    def get_closest_timestamp(self, data: pd.DataFrame, target_timestamp: int) -> Optional[int]:
        """
        Retrieve the closest timestamp before the target timestamp from the DataFrame.

        Args:
            data (pd.DataFrame): The historical market data (indexed by timestamp).
            target_timestamp (int): The target timestamp to search for.

        Returns:
            Optional[int]: The closest available timestamp less than or equal to the target, or None if not found.
        """
        available_timestamps = data.index
        closest_timestamp = available_timestamps[available_timestamps <= target_timestamp].max()
        return closest_timestamp if not pd.isna(closest_timestamp) else None

    def calculate_portfolio_value_at_end(self, portfolio_history: List[Dict], market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate the final portfolio value by evaluating asset values at the end of the backtest.

        Args:
            portfolio_history (List[Dict]): List of portfolio snapshots over time.
            market_data (Dict[str, pd.DataFrame]): Historical market data for assets.

        Returns:
            float: The final portfolio value.
        """
        last_snapshot = portfolio_history[-1]
        last_market_prices = {asset: data.iloc[-1][self.CLOSE] for asset, data in market_data.items()}  # Get last close prices
        final_portfolio_value = last_snapshot['holdings'].get(self.CASH, 0.0)  # Base currency balance

        for asset, quantity in last_snapshot['holdings'].items():
            if asset != self.CASH and asset in last_market_prices:
                final_portfolio_value += quantity * last_market_prices[asset]

        return final_portfolio_value

    def calculate_trade_summary(self, trades: List[Dict], market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calculate a summary of trades executed during the backtest and portfolio performance.
        
        Args:
            trades (List[Dict]): List of executed trades.
            market_data (Dict[str, pd.DataFrame]): Market data for assets.
        
        Returns:
            Dict: A dictionary containing summary statistics or an error message if no trades were executed.
        """
        if not trades:
            return {"error": "No trades were executed during the backtest."}

        portfolio_values = self.create_portfolio_value_series(market_data)
        
        total_profit, total_fees, winning_trades, losing_trades = 0, 0, 0, 0
        stop_loss_trades, take_profit_trades, total_holding_time = 0, 0, 0

        last_timestamp = max(df.index[-1] for df in market_data.values())

        for trade in trades:
            entry_price = trade['entry_price']
            exit_price = trade['exit_price'] or market_data[trade['asset_name']].iloc[-1]['close']
            
            profit = (exit_price - entry_price) * trade['asset_amount']
            profit -= (trade['entry_fee'] + (trade['exit_fee'] or 0))
            total_profit += profit
            total_fees += trade['entry_fee'] + (trade['exit_fee'] or 0)

            if profit > 0:
                winning_trades += 1
            else:
                losing_trades += 1

            if trade['exit_reason'] == self.STOP_LOSS_REASON:
                stop_loss_trades += 1
            elif trade['exit_reason'] == self.TAKE_PROFIT_REASON:
                take_profit_trades += 1

            exit_timestamp = trade['exit_timestamp'] or last_timestamp
            holding_time = exit_timestamp - trade['entry_timestamp']
            total_holding_time += holding_time

        avg_holding_time = total_holding_time / len(trades) if trades else 0
        final_portfolio_value = portfolio_values.iloc[-1]
        initial_portfolio_value = portfolio_values.iloc[0]
        total_percent_increase = ((final_portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100

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
            "average_holding_time_minutes": avg_holding_time / (60 * 1000),
            "final_portfolio_value": final_portfolio_value,
            "total_percent_increase": total_percent_increase,
            "avg_percent_per_month": avg_percent_per_month
        }
    
    def print_summary(self, market_data: Dict[str, pd.DataFrame], trades: List[Dict], signals: List[Dict]) -> None:
        """
        Calculate and print a summary of the backtest results.

        Args:
            market_data (Dict[str, pd.DataFrame]): Market data for each asset.
            trades (List[Dict]): List of executed trades.
            signals (List[Dict]): List of generated signals.
        """
        try:
            sharpe_ratio = self.calculate_sharpe_ratio(market_data)
            max_drawdown = self.calculate_max_drawdown(market_data)
            cumulative_return = self.calculate_cumulative_return(market_data)
            
            trade_summary = self.calculate_trade_summary(trades, market_data)

            total_trades = len(trades)
            total_signals = len(signals)
            assets = list(market_data.keys())
            initial_balance = self.portfolio.history[0]['holdings'][self.base_currency]
            final_balance = trade_summary['final_portfolio_value']

            print("\n=== Backtest Summary ===")
            print(f"Assets: {', '.join(assets)}")
            print(f"Initial Balance: {initial_balance:.2f} {self.base_currency}")
            print(f"Final Balance: {final_balance:.2f} {self.base_currency}")
            print(f"Total Return: {(final_balance - initial_balance):.2f} {self.base_currency} ({cumulative_return:.2%})")
            print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            print(f"Total Trades: {total_trades}")
            print(f"Total Signals: {total_signals}")
            
            for key, value in trade_summary.items():
                if isinstance(value, float):
                    print(f"{key.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")

            print("========================\n")

            # Sanity checks
            assert abs(final_balance - trade_summary['final_portfolio_value']) < 0.01, "Final balance mismatch"
            assert abs(cumulative_return - trade_summary['total_percent_increase'] / 100) < 0.0001, "Cumulative return mismatch"

        except Exception as e:
            self.logger.error(f"Error calculating metrics summary: {str(e)}", exc_info=True)
            print(f"Error occurred while calculating metrics summary: {str(e)}")
            
            if trades:
                print("Sample trade:")
                print(trades[0])
            else:
                print("No trades available")

    def save_metrics_data(self, filepath: str, market_data: Dict[str, pd.DataFrame], trades: List[Dict], signals: List[Dict]) -> None:
        """
        Save all necessary data for metrics calculation to a file.

        Args:
            filepath (str): Path to save the metrics data.
            market_data (Dict[str, pd.DataFrame]): Market data for each asset.
            trades (List[Dict]): List of executed trades.
            signals (List[Dict]): List of generated signals.
        """
        metrics_data = {
            'portfolio_history': self.portfolio.history,
            'base_currency': self.base_currency,
            'risk_free_rate': self.risk_free_rate,
            'trades': trades,
            'signals': signals,
            'market_data': {asset: data.to_json() for asset, data in market_data.items()}
        }

        with open(filepath, 'w') as f:
            json.dump(metrics_data, f)

        self.logger.info(f"Metrics data saved to {filepath}")
    @classmethod
    def load_metrics_data(cls, filepath: str) -> 'MetricsModule':
        """
        Load metrics data from a file and create a MetricsModule instance.

        Args:
            filepath (str): Path to the saved metrics data file.

        Returns:
            MetricsModule: An instance of MetricsModule with loaded data.
        """
        with open(filepath, 'r') as f:
            metrics_data = json.load(f)

        # Recreate the portfolio
        portfolio = type('Portfolio', (), {'history': metrics_data['portfolio_history']})()

        # Create the MetricsModule instance
        metrics_module = cls(portfolio, 
                             base_currency=metrics_data['base_currency'],
                             risk_free_rate=metrics_data['risk_free_rate'])

        # Load market data
        market_data = {asset: pd.read_json(data) for asset, data in metrics_data['market_data'].items()}

        # Store the loaded data
        metrics_module.trades = metrics_data['trades']
        metrics_module.signals = metrics_data['signals']
        metrics_module.market_data = market_data

        return metrics_module

    def generate_metrics(self) -> Dict[str, Any]:
        """
        Generate all metrics using the loaded data.

        Returns:
            Dict[str, Any]: A dictionary containing all calculated metrics.
        """
        sharpe_ratio = self.calculate_sharpe_ratio(self.market_data)
        max_drawdown = self.calculate_max_drawdown(self.market_data)
        cumulative_return = self.calculate_cumulative_return(self.market_data)
        trade_summary = self.calculate_trade_summary(self.trades, self.market_data)

        metrics = {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_return': cumulative_return,
            **trade_summary,
            'total_signals': len(self.signals),
            'assets': list(self.market_data.keys()),
            'initial_balance': self.portfolio.history[0]['holdings'][self.base_currency],
            'final_balance': trade_summary['final_portfolio_value']
        }

        return metrics