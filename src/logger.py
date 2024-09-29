import logging
import os

# Global dictionary to store log levels for different components
LOG_LEVELS = {}

def set_log_levels(levels):
    """
    Set log levels for different components.
    
    Args:
        levels (dict): A dictionary mapping component names to their log levels.
    """
    global LOG_LEVELS
    LOG_LEVELS.update(levels)

def setup_logger(name, log_file=None):
    """
    Sets up a logger with the specified name and logging level.
    
    Args:
        name (str): The name of the logger.
        log_file (str): Path to a file to log to. If None, logs will only be printed to console.
        
    Returns:
        logging.Logger: Configured logger.
    """
    # Determine the log level from the global LOG_LEVELS dictionary
    log_level = LOG_LEVELS.get(name, 'INFO')
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Remove any existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create console handler and set level to log_level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # If a log file is specified, add a file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Avoid duplicate logs
    logger.propagate = False

    return logger