# StableTrade

StableTrade is a scalable backtesting framework for cryptocurrency trading strategies. It allows dynamic strategy switching, distributed parameter optimization using Celery and Redis, and tracks performance metrics like profit and Sharpe ratio.

## Features

- **Dynamic Strategy Switching**: Easily swap between different trading strategies.
- **Distributed Backtesting**: Utilize Celery and Redis to distribute backtests over multiple workers.
- **Parameter Grid Search**: Test different parameter combinations using grid search.
- **Logging & Results**: Logs performance metrics and saves results in JSON format.

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

5. **Run Celery Workers**:

    Ensure Redis is running, then start Celery workers with:

    ```bash
    celery -A src.tasks worker --loglevel=info
    ```

## Usage

1. **Run Backtests**:

    To run backtests using the parameter grid search:

    ```bash
    python main.py --data path_to_data.csv --balance 1000 --commission 0.001 --output results.json
    ```

2. **Run Unit Tests**:

    To run the unit tests using `pytest`:

    ```bash
    pytest
    ```

## Project Structure

``` txt

StableTrade/
├── data/                        # Folder for CSV data
├── src/                         # Source code
│   ├── backtest.py              # Core backtesting logic
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── metrics.py               # Performance metrics (Sharpe Ratio, etc.)
│   ├── optimization.py          # Parameter optimization logic (Grid Search)
│   ├── strategy.py              # Strategy implementation
│   ├── tasks.py                 # Celery tasks for distributed execution
│   ├── utils.py                 # Utility functions (logging, saving results)
├── tests/                       # Test suite using pytest
│   ├── test_backtest.py         # Unit tests for backtesting
├── .env.example                 # Example environment file
├── .gitignore                   # Files to ignore in version control
├── README.md                    # Project description and instructions
├── requirements.txt             # Python dependencies
├── main.py                      # Main script to run backtest or optimization
└── celeryconfig.py              # Celery configuration
```

## Requirements

- Python 3.8+
- Redis
- Celery
- pandas

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
