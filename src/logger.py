import logging
import os
from typing import Optional

# Dictionary to store log levels for different components
LOG_LEVELS = {}

def set_log_levels(levels: dict) -> None:
    """
    Set log levels for different components of the application.

    This function updates the global `LOG_LEVELS` dictionary to change the 
    logging level for specific components.

    Args:
        levels (dict): A dictionary mapping component names to their respective 
                       log levels. Valid levels are 'DEBUG', 'INFO', 'WARNING', 
                       'ERROR', 'CRITICAL'.

    Example:
        set_log_levels({'component1': 'DEBUG', 'component2': 'ERROR'})
    """
    global LOG_LEVELS
    LOG_LEVELS.update(levels)

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Sets up a logger with the specified name and logging level.

    Creates and configures a logger for a given component. The logger outputs
    to both console and a file if `log_file` is provided. The logger's level 
    is determined by the global `LOG_LEVELS` or defaults to 'INFO'.

    Args:
        name (str): The name of the logger, typically the component name.
        log_file (str, optional): Path to the log file for logging output.
                                  If None, logging is only done to the console.

    Returns:
        logging.Logger: The configured logger instance.

    Example:
        logger = setup_logger('component1', '/var/log/component1.log')
        logger.info("Logger setup complete")
    """
    # Determine the log level from the global LOG_LEVELS dictionary
    log_level = LOG_LEVELS.get(name, 'INFO')
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Create or get the logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Remove any existing handlers to avoid duplicate logs
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create console handler and set level to log_level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # If a log file is specified, add a file handler if not already added
    if log_file:
        # Check if the file handler already exists
        file_handler_exists = any(
            isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_file)
            for handler in logger.handlers
        )
        if not file_handler_exists:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    # Avoid duplicate logs by disabling propagation
    logger.propagate = False

    return logger
