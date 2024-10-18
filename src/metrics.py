import pandas as pd
import numpy as np
import json
from io import StringIO
import logging
from src.logger import setup_logger
from typing import Dict, List, Any, Optional

class MetricsModule:
    """
    A module to compute key metrics and performance statistics for a trading portfolio.

    This class provides methods for calculating various trading performance metrics
    such as Sharpe Ratio, Maximum Drawdown, Cumulative Return, and Trade Summaries.
    It also handles saving and loading of metrics data for offline analysis.

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
        Create a time-indexed series of portfolio values, handling duplicate timestamps.
        
        Args:
            market_data (Dict[str, pd.DataFrame]): Dictionary mapping asset names to their market data DataFrames.
        
        Returns:
            pd.Series: Time-indexed series representing portfolio values at each timestamp.
        """
        # Create a unified timestamp index, removing duplicates
        all_timestamps = sorted(set(ts for asset_data in market_data.values() for ts in asset_data.index))
        unified_index = pd.DatetimeIndex(all_timestamps)

        # Pre-process market data, handling duplicates
        processed_market_data = {}
        for asset, data in market_data.items():
            if not data.empty:
                # Group by index and take the last value for each duplicate timestamp
                deduped_data = data.groupby(level=0).last()
                processed_market_data[asset] = deduped_data['close'].reindex(unified_index, method='ffill')

        # Create a DataFrame of holdings, handling duplicates
        holdings_data = []
        for snapshot in self.portfolio.history:
            holdings_data.append({
                'timestamp': pd.to_datetime(snapshot['timestamp'], unit='ms'),
                **snapshot['holdings']
            })
        holdings_df = pd.DataFrame(holdings_data)
        # Group by timestamp and take the last value for each duplicate
        holdings_df = holdings_df.groupby('timestamp', as_index=False).last().set_index('timestamp')

        # Reindex holdings to match the unified index
        holdings_df = holdings_df.reindex(unified_index, method='ffill')

        # Calculate portfolio value
        portfolio_value = holdings_df[self.base_currency].fillna(0)
        for asset, prices in processed_market_data.items():
            if asset in holdings_df.columns:
                portfolio_value += holdings_df[asset].fillna(0) * prices

        return portfolio_value.dropna()
    
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
        daily_returns = daily_portfolio_values.pct_change(fill_method=None).dropna()
        return daily_returns

    def calculate_sharpe_ratio(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate the Sharpe Ratio for the portfolio.
        
        The Sharpe ratio measures the performance of an investment compared to a risk-free asset,
        after adjusting for its risk.

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
        
        Maximum Drawdown measures the largest peak-to-trough decline in the portfolio value.

        Args:
            market_data (Dict[str, pd.DataFrame]): Market data for each asset.
        
        Returns:
            float: The Maximum Drawdown as a percentage.
        """
        portfolio_values = self.create_portfolio_value_series(market_data)
        cumulative_max = portfolio_values.cummax()
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max
        return drawdowns.min()

    def calculate_cumulative_return(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate the Cumulative Return for the portfolio.
        
        Cumulative Return represents the total percentage gain or loss of the portfolio
        over the entire backtest period.

        Args:
            market_data (Dict[str, pd.DataFrame]): Market data for each asset.
        
        Returns:
            float: The Cumulative Return as a percentage.
        """
        portfolio_values = self.create_portfolio_value_series(market_data)
        return (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]

    def calculate_profit_factor(self, trades: List[Dict]) -> float:
        """
        Calculate the profit factor for the executed trades.

        Profit factor is the ratio of gross profit to gross loss. A value greater than 1 indicates
        a profitable system.

        Args:
            trades (List[Dict]): List of executed trades.

        Returns:
            float: The profit factor.
        """
        if not trades:
            return np.nan

        total_profit = sum(trade.get('profit_loss', 0) for trade in trades if trade.get('profit_loss', 0) > 0)
        total_loss = sum(abs(trade.get('profit_loss', 0)) for trade in trades if trade.get('profit_loss', 0) < 0)
        return np.inf if total_loss == 0 else total_profit / total_loss

    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """
        Calculate the win rate for the executed trades.

        Win rate is the percentage of trades that resulted in a profit.

        Args:
            trades (List[Dict]): List of executed trades.

        Returns:
            float: The win rate as a percentage.
        """
        if not trades:
            return np.nan

        winning_trades = sum(1 for trade in trades if trade.get('profit_loss', 0) > 0)
        return (winning_trades / len(trades)) * 100
    
    def calculate_trade_summary(self, trades: List[Dict]) -> Dict:
        """
        Calculate a summary of trades executed during the backtest.
        
        Args:
            trades (List[Dict]): List of executed trades.
        
        Returns:
            Dict: A dictionary containing summary statistics.
        """
        if not trades:
            return {"error": "No trades were executed during the backtest."}

        total_profit = sum(trade['profit_loss'] for trade in trades)
        total_fees = sum(trade['entry_fee'] + trade['exit_fee'] for trade in trades)
        winning_trades = sum(1 for trade in trades if trade['profit_loss'] > 0)
        losing_trades = sum(1 for trade in trades if trade['profit_loss'] <= 0)
        stop_loss_trades = sum(1 for trade in trades if trade['exit_reason'] == 'stop_loss')
        take_profit_trades = sum(1 for trade in trades if trade['exit_reason'] == 'take_profit')
        total_holding_time = sum(trade['exit_timestamp'] - trade['entry_timestamp'] for trade in trades)

        avg_holding_time = total_holding_time / len(trades) if trades else 0

        return {
            "total_trades": len(trades),
            "total_profit": total_profit,
            "total_fees": total_fees,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "stop_loss_trades": stop_loss_trades,
            "take_profit_trades": take_profit_trades,
            "average_holding_time_minutes": avg_holding_time / (1000 * 60),
            "win_rate": (winning_trades / len(trades)) * 100 if trades else 0,
            "average_trade_profit": total_profit / len(trades) if trades else 0
        }    
    
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
            'market_data': {asset: data.reset_index().to_json(orient='split', date_format='iso') for asset, data in market_data.items()}
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
        market_data = {}
        for asset, data_json in metrics_data['market_data'].items():
            try:
                df = pd.read_json(StringIO(data_json))
                if 'timestamp' not in df.columns:
                    df['timestamp'] = pd.to_datetime(df.index, unit='ms')
                df.set_index('timestamp', inplace=True)
                market_data[asset] = df
            except Exception as e:
                logging.error(f"Error loading market data for {asset}: {str(e)}")
                logging.debug(f"Data for {asset}: {data_json[:100]}...")  # Log the first 100 characters of the data

        # Store the loaded data
        metrics_module.trades = metrics_data['trades']
        metrics_module.signals = metrics_data['signals']
        metrics_module.market_data = market_data

        logging.info(f"Loaded market data for assets: {list(market_data.keys())}")
        for asset, df in market_data.items():
            logging.debug(f"Columns for {asset}: {df.columns}")
            logging.debug(f"Index name for {asset}: {df.index.name}")
            logging.debug(f"First few rows for {asset}:\n{df.head()}")

        return metrics_module
    
    def generate_metrics(self) -> Dict[str, Any]:
        """
        Generate a comprehensive set of metrics using the loaded data.

        Returns:
            Dict[str, Any]: A dictionary containing all calculated metrics or error information.
        """
        if not self.market_data:
            logging.error("No market data available to generate metrics.")
            return {"error": "No market data available"}

        try:
            portfolio_value_series = self.create_portfolio_value_series(self.market_data)
            
            if portfolio_value_series.empty:
                logging.error("Generated portfolio value series is empty.")
                return {"error": "Generated portfolio value series is empty"}

            start_date = portfolio_value_series.index[0]
            end_date = portfolio_value_series.index[-1]
            total_days = (end_date - start_date).days

            initial_balance = portfolio_value_series.iloc[0]
            final_balance = portfolio_value_series.iloc[-1]
            total_profit = final_balance - initial_balance
            total_percent_increase = (final_balance / initial_balance - 1) * 100

            sharpe_ratio = self.calculate_sharpe_ratio(self.market_data)
            max_drawdown = self.calculate_max_drawdown(self.market_data)
            cumulative_return = self.calculate_cumulative_return(self.market_data)
            trade_summary = self.calculate_trade_summary(self.trades)
            
            win_rate = self.calculate_win_rate(self.trades)
            profit_factor = self.calculate_profit_factor(self.trades)

            metrics = {
                'start_date': start_date.strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': end_date.strftime('%Y-%m-%d %H:%M:%S'),
                'total_days': total_days,
                'initial_balance': initial_balance,
                'final_balance': final_balance,
                'total_profit': total_profit,
                'total_percent_increase': total_percent_increase,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'cumulative_return': cumulative_return,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_signals': len(self.signals),
                'assets': list(self.market_data.keys()),
                **trade_summary
            }
            
            metrics['annualized_return'] = ((1 + cumulative_return) ** (365 / total_days) - 1) * 100 if total_days > 0 else 0
            metrics['trades_per_day'] = metrics['total_trades'] / total_days if total_days > 0 else 0
            
            # Corrected average monthly return calculation
            total_months = total_days / 30.44  # Average number of days in a month
            metrics['average_monthly_return'] = ((1 + cumulative_return) ** (1 / total_months) - 1) * 100 if total_months > 0 else 0
            
            return metrics
        except Exception as e:
            logging.error(f"Error generating metrics: {str(e)}", exc_info=True)
            return {"error": str(e)}
            
    def print_summary(self) -> None:
        """
        Print a summary of all generated metrics.

        This method calls generate_metrics() and prints the metrics in a formatted,
        easy-to-read manner. It provides an overview of the trading strategy's
        performance, with an option for more detailed output.

        Args:
            detailed (bool): If True, prints additional detailed metrics. Defaults to False.
        """
        metrics = self.generate_metrics()
        
        if "error" in metrics:
            print(f"Error generating metrics: {metrics['error']}")
            return

        print("\n========== Backtest Summary ==========")
        print(f"Start Date: {metrics['start_date']}")
        print(f"End Date: {metrics['end_date']}")
        print(f"Total Days: {metrics['total_days']}")
        print(f"Initial Balance: {metrics['initial_balance']:.2f} {self.base_currency}")
        print(f"Final Balance: {metrics['final_balance']:.2f} {self.base_currency}")
        print(f"Total Profit: {metrics['total_profit']:.2f} {self.base_currency}")
        print(f"Total Percent Increase: {metrics['total_percent_increase']:.2f}%")
        print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
        print(f"Average Monthly Return: {metrics['average_monthly_return']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Cumulative Return: {metrics['cumulative_return']:.2f}%")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Profit Factor: {metrics['profit_factor']:.4f}")
    
        print("\n----- Detailed Metrics -----")
        print(f"Winning Trades: {metrics['winning_trades']}")
        print(f"Losing Trades: {metrics['losing_trades']}")
        print(f"Average Trade Profit: {metrics['average_trade_profit']:.2f} {self.base_currency}")
        print(f"Average Holding Time: {metrics['average_holding_time_minutes']:.2f} minutes")
        print(f"Trades Per Day: {metrics['trades_per_day']:.2f}")
        print(f"Total Fees: {metrics['total_fees']:.2f} {self.base_currency}")
        print(f"Stop Loss Trades: {metrics['stop_loss_trades']}")
        print(f"Take Profit Trades: {metrics['take_profit_trades']}")
        print(f"Total Signals Generated: {metrics['total_signals']}")
        print(f"Assets Traded: {', '.join(metrics['assets'])}")

        print("==========================================")