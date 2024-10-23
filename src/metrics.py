import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import ast
import os
import logging
from typing import Dict, Any

class MetricsModule:
    def __init__(self, market_data_path: str, base_currency: str, 
                 risk_free_rate: float = 0.01, log_level: int = logging.INFO):
        self.market_data_path = market_data_path
        self.data_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_currency = base_currency
        self.risk_free_rate = risk_free_rate
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Add a handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                '%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(handler)

    def load_data(self) -> Dict[str, Any]:
        """
        Load all necessary data from CSV files.
        """
        try:
            market_data = pd.read_csv(self.market_data_path)
            market_data['timestamp'] = pd.to_datetime(market_data['timestamp'], unit='ms')
            market_data.set_index('timestamp', inplace=True)
            
            signals = pd.read_csv(os.path.join(self.data_directory, 'signals.csv'))
            signals['timestamp'] = pd.to_datetime(signals['timestamp'], unit='ms')
            
            trades = pd.read_csv(os.path.join(self.data_directory, 'trades.csv'))
            trades['entry_timestamp'] = pd.to_datetime(trades['entry_timestamp'], unit='ms')
            trades['exit_timestamp'] = pd.to_datetime(trades['exit_timestamp'], unit='ms')
            
            portfolio_history = pd.read_csv(os.path.join(self.data_directory, 'portfolio.csv'))
            portfolio_history['timestamp'] = pd.to_datetime(portfolio_history['timestamp'], unit='ms')

            # Convert the 'holdings' column in portfolio_history from string to dictionary
            portfolio_history['holdings'] = portfolio_history['holdings'].apply(ast.literal_eval)

            return {
                'market_data': market_data,
                'signals': signals,
                'trades': trades,
                'portfolio_history': portfolio_history
            }
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            self.logger.error(f"Empty CSV file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def calculate_monthly_returns(self, portfolio_history: pd.DataFrame, market_data: pd.DataFrame) -> pd.Series:
        """
        Calculate monthly returns more efficiently.
        """
        self.logger.info("Starting monthly returns calculation...")
        
        # Calculate portfolio values first
        if 'total_value' not in portfolio_history.columns:
            portfolio_history['total_value'] = portfolio_history['holdings'].apply(lambda x: sum(x.values()))
        
        # Set timestamp as index if it isn't already
        if portfolio_history.index.name != 'timestamp':
            portfolio_history.set_index('timestamp', inplace=True)
        
        # Resample to daily first, then monthly for better performance
        daily_values = portfolio_history['total_value'].resample('D').last().ffill()  # Using ffill() instead of fillna
        monthly_values = daily_values.resample('ME').last()  # Using 'ME' instead of 'M'
        monthly_returns = monthly_values.pct_change().dropna()
        
        return monthly_returns
    
    def calculate_sharpe_ratio(self, monthly_returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio based on monthly returns.

        Args:
        monthly_returns (pd.Series): Series of monthly returns.

        Returns:
        float: The annualized Sharpe ratio.
        """
        try:
            if len(monthly_returns) < 2:
                raise ValueError("At least two months of returns are required to calculate Sharpe ratio")

            # Calculate excess returns
            excess_returns = monthly_returns - (self.risk_free_rate / 12)  # Convert annual risk-free rate to monthly

            mean_excess_return = excess_returns.mean()
            std_excess_return = excess_returns.std()

            # Handle near-zero volatility case
            if np.isclose(std_excess_return, 0, atol=1e-8):
                if mean_excess_return > 0:
                    return float('inf')
                elif mean_excess_return < 0:
                    return float('-inf')
                else:
                    return 0.0

            # Calculate annualized Sharpe ratio
            sharpe_ratio = np.sqrt(12) * mean_excess_return / std_excess_return
            
            self.logger.info(f"Sharpe ratio calculated ")
            return sharpe_ratio

        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            raise
    
    def calculate_max_drawdown(self, portfolio_history: pd.DataFrame) -> float:
        """
        Calculate maximum drawdown from portfolio history.

        Args:
        portfolio_history (pd.DataFrame): DataFrame containing portfolio history with 'timestamp' and 'holdings' columns.

        Returns:
        float: The maximum drawdown as a percentage (0 to 1).
        """
        try:
            if len(portfolio_history) < 2:
                raise ValueError("At least two data points are required to calculate maximum drawdown")

            # Calculate portfolio value at each timestamp
            portfolio_values = portfolio_history.apply(
                lambda row: sum(row['holdings'].values()),
                axis=1
            )

            # Calculate running maximum
            running_max = np.maximum.accumulate(portfolio_values)
            
            # Calculate drawdown
            drawdown = (running_max - portfolio_values) / running_max
            
            # Find maximum drawdown
            max_drawdown = drawdown.max()

            self.logger.info(f"Max drawdown calculated ")
            return max_drawdown

        except Exception as e:
            self.logger.error(f"Error calculating maximum drawdown: {e}")
            raise

    def calculate_win_rate(self, trades: pd.DataFrame) -> float:
        """
        Calculate the win rate based on closed trades.

        Args:
            trades (pd.DataFrame): DataFrame containing trade information.

        Returns:
            float: The win rate as a percentage.
        """
        if trades.empty:
            self.logger.warning("No trades provided for win rate calculation.")
            return 0.0

        closed_trades = trades[trades['status'] == 'closed']
        
        if closed_trades.empty:
            self.logger.warning("No closed trades found for win rate calculation.")
            return 0.0

        winning_trades = closed_trades[closed_trades['profit_loss'] > 0]
        win_rate = (len(winning_trades) / len(closed_trades)) * 100

        self.logger.info(f"Win rate calculated: {win_rate:.2f}%")
        return win_rate
    
    def calculate_profit_factor(self, trades: pd.DataFrame) -> float:
        """
        Calculate profit factor from trades data, including fees.

        Args:
            trades (pd.DataFrame): DataFrame containing trade information.

        Returns:
            float: The profit factor. Returns 0 if there are no trades or no losing trades.
        """
        if trades.empty:
            self.logger.warning("No trades provided for profit factor calculation.")
            return 0.0

        closed_trades = trades[trades['status'] == 'closed']
        
        if closed_trades.empty:
            self.logger.warning("No closed trades found for profit factor calculation.")
            return 0.0

        # Calculate total profit and loss, including fees
        winning_trades = closed_trades[closed_trades['profit_loss'] > 0]
        losing_trades = closed_trades[closed_trades['profit_loss'] <= 0]

        gross_profit = winning_trades['profit_loss'].sum()
        gross_loss = abs(losing_trades['profit_loss'].sum())

        if gross_loss == 0:
            self.logger.info("No losing trades found. Profit factor is undefined (returning 0).")
            return 0.0

        profit_factor = gross_profit / gross_loss

        self.logger.info(f"Profit factor calculated: {profit_factor:.2f}")
        return profit_factor

    def calculate_average_trade_duration(self, trades: pd.DataFrame) -> float:
        """
        Calculate average trade duration from trades data.

        Args:
            trades (pd.DataFrame): DataFrame containing trade information.
                Expected columns: 'status', 'entry_timestamp', 'exit_timestamp'

        Returns:
            float: The average trade duration in seconds. Returns 0 if no closed trades.
        """
        if trades.empty:
            self.logger.warning("No trades provided for average trade duration calculation.")
            return 0.0

        closed_trades = trades[(trades['status'] == 'closed') & (trades['exit_timestamp'].notna())].copy()
        
        if closed_trades.empty:
            self.logger.warning("No closed trades found for average trade duration calculation.")
            return 0.0

        closed_trades.loc[:, 'duration'] = (closed_trades['exit_timestamp'] - closed_trades['entry_timestamp']).dt.total_seconds()
        average_duration = closed_trades['duration'].mean()

        self.logger.info(f"Average trade duration calculated: {average_duration:.2f} seconds")
        return average_duration
    
    def calculate_total_fees(self, trades: pd.DataFrame) -> float:
        """
        Calculate total fees from trades data.

        Args:
            trades (pd.DataFrame): DataFrame containing trade information.
                Expected columns: 'entry_fee', 'exit_fee'

        Returns:
            float: The total fees incurred across all trades. Returns 0 if no trades or no fees.
        """
        if trades.empty:
            self.logger.warning("No trades provided for total fees calculation.")
            return 0.0

        entry_fees = trades['entry_fee'].sum()
        exit_fees = trades['exit_fee'].sum()
        total_fees = entry_fees + exit_fees

        self.logger.info(f"Total fees calculated: {total_fees:.2f}")
        return total_fees

    def calculate_total_return(self, portfolio_history: pd.DataFrame) -> float:
        """
        Calculate total return from portfolio history.

        Args:
            portfolio_history (pd.DataFrame): DataFrame containing portfolio history.
                Expected columns: 'timestamp', 'holdings'

        Returns:
            float: The total return as a percentage. Returns 0 if insufficient data.
        """
        if portfolio_history.empty or len(portfolio_history) < 2:
            self.logger.warning("Insufficient data for total return calculation.")
            return 0.0

        # Calculate total value from holdings
        portfolio_history['total_value'] = portfolio_history['holdings'].apply(
            lambda x: sum(x.values())
        )

        initial_value = portfolio_history['total_value'].iloc[0]
        final_value = portfolio_history['total_value'].iloc[-1]

        if initial_value == 0:
            self.logger.warning("Initial portfolio value is zero. Cannot calculate total return.")
            return 0.0

        total_return = (final_value - initial_value) / initial_value * 100

        self.logger.info(f"Total return calculated: {total_return:.2f}%")
        return total_return
    
    def generate_summary(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate metrics summary using pre-loaded data with improved calculations.
        """
        start_time = time.time()
        self.logger.info("Starting summary generation...")

        try:
            portfolio_history = data['portfolio_history'].copy()
            
            # Ensure timestamp is datetime and set as index
            portfolio_history['timestamp'] = pd.to_datetime(portfolio_history['timestamp'])
            portfolio_history.set_index('timestamp', inplace=True)
            portfolio_history = portfolio_history.sort_index()

            # Calculate total_value if not present
            if 'total_value' not in portfolio_history.columns:
                portfolio_history['total_value'] = portfolio_history['holdings'].apply(lambda x: sum(x.values()))

            # Calculate monthly returns
            monthly_returns = self.calculate_monthly_returns(portfolio_history, data['market_data'])
            monthly_returns_dict = {k.strftime('%Y-%m'): v for k, v in monthly_returns.to_dict().items()}

            # Calculate daily returns properly
            daily_returns = portfolio_history['total_value'].pct_change()
            
            # Calculate best/worst returns
            best_return = daily_returns.max() * 100
            worst_return = daily_returns.min() * 100
            
            best_return_date = daily_returns.idxmax().strftime('%Y-%m-%d %H:%M:%S') if best_return > 0 else "N/A"
            worst_return_date = daily_returns.idxmin().strftime('%Y-%m-%d %H:%M:%S') if worst_return < -0.0001 else "N/A"

            # Improved drawdown calculation
            rolling_max = portfolio_history['total_value'].expanding().max()
            drawdowns = (portfolio_history['total_value'] - rolling_max) / rolling_max * 100
            max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0

            # Calculate Sharpe ratio
            excess_returns = daily_returns - (self.risk_free_rate / 252)
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(excess_returns) > 1 else 0

            # Calculate total profit properly
            total_profit = float(portfolio_history['total_value'].iloc[-1] - portfolio_history['total_value'].iloc[0])

            # Build summary dictionary
            summary = {
                'calculation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'base_currency': self.base_currency,
                
                'portfolio_metrics': {
                    'total_return': self.calculate_total_return(portfolio_history),
                    'monthly_returns': monthly_returns_dict,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'best_daily_return': best_return,
                    'best_return_date': best_return_date,
                    'worst_daily_return': worst_return if worst_return < -0.0001 else 0,
                    'worst_return_date': worst_return_date,
                    'initial_value': float(portfolio_history['total_value'].iloc[0]),
                    'final_value': float(portfolio_history['total_value'].iloc[-1]),
                    'volatility': float(daily_returns.std() * np.sqrt(252) * 100),  # Annualized volatility
                    'avg_monthly_return': float(monthly_returns.mean() * 100)
                },
                
                'trade_metrics': {
                    'total_trades': len(data['trades']),
                    'win_rate': self.calculate_win_rate(data['trades']),
                    'profit_factor': self.calculate_profit_factor(data['trades']),
                    'average_trade_duration': self.calculate_average_trade_duration(data['trades']),
                    'total_profit': total_profit,
                    'max_drawdown': max_drawdown,
                    'winning_trades': len(data['trades'][data['trades']['profit_loss'] > 0]),
                    'losing_trades': len(data['trades'][data['trades']['profit_loss'] <= 0]),
                    'avg_profit_per_trade': float(data['trades']['profit_loss'].mean()),
                    'total_fees': float(data['trades']['entry_fee'].sum() + data['trades']['exit_fee'].sum()),
                    'largest_win': float(data['trades']['profit_loss'].max()),
                    'largest_loss': float(data['trades']['profit_loss'].min()),
                    'avg_win': float(data['trades'][data['trades']['profit_loss'] > 0]['profit_loss'].mean()),
                    'avg_loss': float(data['trades'][data['trades']['profit_loss'] < 0]['profit_loss'].mean())
                },
                
                'signal_metrics': {
                    'total_signals': len(data['signals']),
                    'executed_signals': len(data['signals'][data['signals']['status'] == 'executed']),
                    'rejected_signals': len(data['signals'][data['signals']['status'] == 'rejected'])
                },
                
                'time_range': {
                    'start': portfolio_history.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': portfolio_history.index.max().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_days': (portfolio_history.index.max() - portfolio_history.index.min()).days
                }
            }
            
            self.logger.info(f"Summary generated in {time.time() - start_time:.2f} seconds")
            return summary, portfolio_history

        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            raise

    def run(self) -> None:
        """
        Main method to run all calculations and generate output.
        """
        try:
            start_time = time.time()
            self.logger.info("Starting metrics calculation process...")
            
            # Load data once
            data = self.load_data()
            self.logger.info("Data loaded successfully")
            
            # Ensure portfolio values are calculated
            if 'total_value' not in data['portfolio_history'].columns:
                data['portfolio_history']['total_value'] = data['portfolio_history']['holdings'].apply(
                    lambda x: sum(x.values())
                )
            
            # Generate summary and get processed portfolio history
            summary, processed_portfolio = self.generate_summary(data)
            
            # Create output directory
            output_dir = Path(self.data_directory) / 'output'
            output_dir.mkdir(exist_ok=True)
            
            # Save summary to JSON
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_path = output_dir / f'metrics_summary_{timestamp}.json'
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=4)
            
            # Print summary tables to console
            self._print_summary_tables(summary)
            
            # Generate and save charts using processed portfolio history
            self._generate_charts(processed_portfolio, output_dir, timestamp)
            
            self.logger.info(f"Process completed in {time.time() - start_time:.2f} seconds")
            print(f"\nResults saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error during metrics calculation: {str(e)}")
            raise
        
    def _print_summary_tables(self, summary: Dict[str, Any]) -> None:
        """
        Print formatted summary tables to console.
        """
        print("\n=== TRADING METRICS SUMMARY ===\n")
        
        # Portfolio Performance
        print("\nPortfolio Performance:")
        portfolio_table = [
            ["Total Return", f"{summary['portfolio_metrics']['total_return']:.2f}%"],
            ["Initial Value", f"{summary['portfolio_metrics']['initial_value']:.2f} {self.base_currency}"],
            ["Final Value", f"{summary['portfolio_metrics']['final_value']:.2f} {self.base_currency}"],
            ["Sharpe Ratio", f"{summary['portfolio_metrics']['sharpe_ratio']:.2f}"],
            ["Max Drawdown", f"{summary['portfolio_metrics']['max_drawdown']:.2f}%"],
            ["Volatility", f"{summary['portfolio_metrics']['volatility']:.2f}%"],
            ["Avg Monthly Return", f"{summary['portfolio_metrics']['avg_monthly_return']:.2f}%"],
            ["Best Daily Return", f"{summary['portfolio_metrics']['best_daily_return']:.2f}% ({summary['portfolio_metrics']['best_return_date']})"],
            ["Worst Daily Return", f"{summary['portfolio_metrics']['worst_daily_return']:.2f}% ({summary['portfolio_metrics']['worst_return_date']})"]
        ]
        print(tabulate(portfolio_table, tablefmt="grid"))
        
        # Trade Statistics
        print("\nTrade Statistics:")
        trade_table = [
            ["Total Trades", summary['trade_metrics']['total_trades']],
            ["Win Rate", f"{summary['trade_metrics']['win_rate']:.2f}%"],
            ["Profit Factor", f"{summary['trade_metrics']['profit_factor']:.2f}"],
            ["Total Profit", f"{summary['trade_metrics']['total_profit']:.2f} {self.base_currency}"],
            ["Average Trade Duration", f"{summary['trade_metrics']['average_trade_duration']/60:.1f} minutes"],
            ["Largest Win", f"{summary['trade_metrics']['largest_win']:.2f} {self.base_currency}"],
            ["Largest Loss", f"{summary['trade_metrics']['largest_loss']:.2f} {self.base_currency}"],
            ["Average Win", f"{summary['trade_metrics']['avg_win']:.2f} {self.base_currency}"],
            ["Average Loss", f"{summary['trade_metrics']['avg_loss']:.2f} {self.base_currency}"],
            ["Total Fees", f"{summary['trade_metrics']['total_fees']:.2f} {self.base_currency}"]
        ]
        print(tabulate(trade_table, tablefmt="grid"))
        
        # Monthly Returns Table - Horizontal format
        print("\nMonthly Returns:")
        monthly_returns = pd.Series(summary['portfolio_metrics']['monthly_returns'])
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_returns = monthly_returns.sort_index()
        
        # Create two rows: dates and returns
        dates = [date.strftime('%Y-%m') for date in monthly_returns.index]
        returns = [f"{return_val*100:.2f}%" for return_val in monthly_returns.values]
        
        monthly_table = [
            dates,
            returns
        ]
        
        print(tabulate(monthly_table, tablefmt="grid"))

    def _generate_charts(self, portfolio_history: pd.DataFrame, output_dir: Path, timestamp: str):
        """
        Generate improved portfolio value chart with better formatting.
        """
        try:
            chart_start = time.time()
            
            # Reset index if timestamp is in index
            if isinstance(portfolio_history.index, pd.DatetimeIndex):
                portfolio_history = portfolio_history.reset_index()
            
            # Ensure data is sorted by timestamp
            portfolio_history = portfolio_history.sort_values('timestamp')
            
            # Create figure
            plt.figure(figsize=(15, 8), dpi=100)
            
            # Convert timestamp to datetime if needed
            portfolio_history['timestamp'] = pd.to_datetime(portfolio_history['timestamp'])
            
            # Plot main portfolio value line
            plt.plot(portfolio_history['timestamp'], 
                    portfolio_history['total_value'],
                    linewidth=1.5, 
                    color='blue', 
                    alpha=0.8,
                    label='Portfolio Value')
            
            # Format axes
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show one tick per month
            plt.xticks(rotation=45, ha='right')
            
            # Set y-axis limits with some padding
            ymin = portfolio_history['total_value'].min() * 0.95
            ymax = portfolio_history['total_value'].max() * 1.05
            plt.ylim(ymin, ymax)
            
            # Format y-axis with currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.2f} {self.base_currency}'))
            
            # Labels and title
            plt.title('Portfolio Value Over Time', fontsize=14, pad=20)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel(f'Value ({self.base_currency})', fontsize=12)
            
            # Add grid
            plt.grid(True, linestyle='--', alpha=0.3)
            
            # Calculate drawdown
            rolling_max = portfolio_history['total_value'].expanding().max()
            drawdown = (portfolio_history['total_value'] - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()
            max_drawdown_idx = drawdown.idxmin()
            
            if max_drawdown < -0.01:  # Only show if drawdown is meaningful
                max_drawdown_date = portfolio_history.loc[max_drawdown_idx, 'timestamp']
                plt.axvline(x=max_drawdown_date, 
                        color='red', 
                        linestyle='--', 
                        alpha=0.5,
                        label=f'Max Drawdown: {max_drawdown:.2f}%')
                
                # Add maximum drawdown annotation
                plt.annotate(f'Max Drawdown: {max_drawdown:.2f}%',
                            xy=(max_drawdown_date, portfolio_history.loc[max_drawdown_idx, 'total_value']),
                            xytext=(10, -30),
                            textcoords='offset points',
                            ha='left',
                            va='top',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Add legend
            plt.legend(loc='upper left')
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save chart
            chart_path = output_dir / f'portfolio_value_{timestamp}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Chart generated and saved to {chart_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating chart: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    market_data_path = r"D:\StableTrade_dataset\EUTEUR_1m\EUTEUR_1m_final_merged.csv"
    base_currency = "EUR"
    
    # Create instance with DEBUG level logging
    metrics_module = MetricsModule(market_data_path, base_currency, log_level=logging.DEBUG)
    metrics_module.run()