# StableTrade

StableTrade is a scalable backtesting framework designed for trading strategies in the cryptocurrency market. The framework allows users to dynamically switch strategies, preprocess data, and run detailed backtests with trade management, portfolio tracking, and logging of results. It's designed for extensibility and can be enhanced with additional features as your strategies evolve.

## Features

- **Dynamic Strategy Management**: Integrate and run multiple strategies dynamically.
- **Advanced Trade Management**: Execute trades with stop-loss, take-profit, and trailing stop features.
- **Backtest Engine**: Robust backtest engine to evaluate strategies against historical data.
- **Data Preprocessing**: Automatically preprocess market data and calculate required indicators.
- **Detailed Logging**: Logs all trades and portfolio performance for analysis and debugging.
- **Scalability**: Ready to scale with Celery and Redis for distributed processing (optional future feature).

## Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/Rovis91/StableTrade
    cd StableTrade
    ```

2. **Set up Virtual Environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Requirements**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Environment Variables**:

    Copy `.env.example` to `.env` and configure as necessary.

    ```bash
    cp .env.example .env
    ```

5. **(Optional) Set up Celery Workers**:

    For distributed execution with Celery and Redis (if needed for parameter optimization), ensure Redis is running, then start Celery workers with:

    ```bash
    celery -A src.tasks worker --loglevel=info
    ```

## Usage

1. **Data Preprocessing**:

    Before running the backtest, ensure your data is preprocessed. The backtest engine will automatically preprocess the data based on the indicators required by the strategy.

    ```bash
    python main.py --data path_to_data.csv --preprocess
    ```

2. **Run Backtests**:

    Run backtests using preprocessed data with your configured strategy:

    ```bash
    python main.py --data path_to_preprocessed_data.csv --balance 100000 --output backtest_results.json
    ```

3. **Run Unit Tests**:

    To run the unit tests using `pytest`:

    ```bash
    pytest
    ```

## Project Structure

``` txt

StableTrade/
├── data/                        # Folder for raw and preprocessed CSV data
├── src/                         # Source code
│   ├── backtest_engine.py       # Core backtesting logic
│   ├── data_preprocessor.py     # Data loading and preprocessing
│   ├── portfolio.py             # Portfolio management logic
│   ├── trade_manager.py         # Trade management (open/close trades)
│   ├── strategy/                # Folder for strategies
│   │   ├── base_strategy.py     # Abstract base class for all strategies
│   │   ├── depeg_strategy.py    # Example strategy implementation
│   └── tasks.py                 # Celery tasks for distributed execution (optional)
├── tests/                       # Test suite using pytest
│   ├── test_backtest.py         # Unit tests for backtesting engine
├── .env.example                 # Example environment file
├── .gitignore                   # Files to ignore in version control
├── README.md                    # Project description and instructions
├── requirements.txt             # Python dependencies
├── main.py                      # Main script to run backtest or data preprocessing
└── celeryconfig.py              # Celery configuration (optional)
```

## Requirements

- **Python 3.8+**
- **pandas** for data manipulation
- **Redis** and **Celery** (Optional for distributed backtesting)

## Configuration

1. **Environment Variables**:

   Use the `.env` file to configure the necessary environment variables, such as database connections (if required in the future) or Redis for distributed backtesting.

2. **Backtest Parameters**:

   The parameters for running the backtest, such as the initial balance, slippage, and fees, can be adjusted through the command-line interface or by modifying the `main.py` file directly.

## Future Enhancements

- **Distributed Backtesting**: Scale backtests across multiple workers using Celery and Redis.
- **Parameter Optimization**: Implement grid search and optimization techniques to fine-tune strategy parameters.
- **Additional Strategy Support**: Easily add new strategies or trading rules through the strategy folder.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
