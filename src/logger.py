import logging
import os

def setup_logger(name=__name__, log_level=None, log_file=None):
    """
    Sets up a logger with the specified name and logging level.
    
    Args:
        name (str): The name of the logger.
        log_level (str): The logging level (e.g., 'DEBUG', 'INFO'). Defaults to 'INFO'.
        log_file (str): Path to a file to log to. If None, logs will only be printed to console.
        
    Returns:
        logging.Logger: Configured logger.
    """
    # Determine the log level from the environment or default to INFO
    log_level = log_level or os.getenv('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level, logging.INFO)
    
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create console handler and set level to log_level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # If a log file is specified, add a file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Avoid duplicate logs
    logger.propagate = False

    return logger
