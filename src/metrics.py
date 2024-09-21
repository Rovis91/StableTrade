import numpy as np
import pandas as pd

class Metrics:
    def __init__(self, portfolio, trade_manager):
        """
        Initialize the Metrics class.

        Args:
            portfolio (Portfolio): The portfolio object to track.
            trade_manager (TradeManager): The trade manager object to track trades.
        """
        self.portfolio = portfolio
        self.trade_manager = trade_manager

    def calculate_pnl(self, timestamp):
        """
        Calculate total profit and loss (PnL) for the portfolio at a specific timestamp.

        Args:
            timestamp (float): The timestamp for calculating the PnL.

        Returns:
            float: The total PnL.
        """
        portfolio_value = self.portfolio.get_portfolio_value(timestamp)
        return portfolio_value - self.portfolio.initial_cash

    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """
        Calculate the Sharpe ratio for the portfolio.

        Args:
            risk_free_rate (float): The risk-free rate of return. Default is 2%.

        Returns:
            float: The Sharpe ratio.
        """
        returns = self._get_portfolio_returns()
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0

    def calculate_max_drawdown(self):
        """
        Calculate the maximum drawdown of the portfolio.

        Returns:
            float: The maximum drawdown.
        """
        portfolio_values = self.portfolio.get_portfolio_value_over_time()
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return np.min(drawdown)

    def calculate_win_loss_ratio(self):
        """
        Calculate the win/loss ratio for trades.

        Returns:
            float: The win/loss ratio.
        """
        closed_trades = self.trade_manager.get_trades(closed=True)
        wins = closed_trades[closed_trades['exit_price'] > closed_trades['entry_price']].shape[0]
        losses = closed_trades[closed_trades['exit_price'] <= closed_trades['entry_price']].shape[0]
        return wins / losses if losses > 0 else float('inf')

    def calculate_trade_duration(self):
        """
        Calculate the average duration of trades.

        Returns:
            float: The average trade duration in seconds.
        """
        closed_trades = self.trade_manager.get_trades(closed=True)
        durations = closed_trades['exit_timestamp'] - closed_trades['entry_timestamp']
        return durations.mean() if not durations.empty else 0

    def calculate_average_return_per_trade(self):
        """
        Calculate the average return per trade.

        Returns:
            float: The average return per trade.
        """
        closed_trades = self.trade_manager.get_trades(closed=True)
        returns = (closed_trades['exit_price'] - closed_trades['entry_price']) * closed_trades['amount']
        return returns.mean() if not returns.empty else 0

    def calculate_total_return(self, timestamp):
        """
        Calculate the total return of the portfolio.

        Args:
            timestamp (float): The timestamp for calculating total return.

        Returns:
            float: The total return as a percentage.
        """
        initial_cash = self.portfolio.initial_cash
        final_value = self.portfolio.get_portfolio_value(timestamp)
        return (final_value - initial_cash) / initial_cash * 100

    def _get_portfolio_returns(self):
        """
        Get the portfolio returns over time.

        Returns:
            np.array: Array of portfolio returns.
        """
        portfolio_values = self.portfolio.get_portfolio_value_over_time()
        return np.diff(portfolio_values) / portfolio_values[:-1] if len(portfolio_values) > 1 else np.array([])

