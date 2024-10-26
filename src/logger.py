import logging
import os
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime
import glob

# Dictionary to store log levels and configurations for different components
LOG_LEVELS: Dict[str, str] = {}
LOG_CONFIG = {
    'save_logs': True,  # Global flag to enable/disable file logging
    'max_log_files': 10,  # Maximum number of log files to keep per component
    'log_dir': 'logs'  # Default directory for log files
}

def set_log_config(save_logs: bool = True, max_log_files: int = 10, log_dir: str = 'logs') -> None:
    """
    Configure global logging settings.

    Args:
        save_logs: Whether to save logs to files
        max_log_files: Maximum number of log files to keep per component
        log_dir: Directory to store log files
    """
    LOG_CONFIG.update({
        'save_logs': save_logs,
        'max_log_files': max_log_files,
        'log_dir': log_dir
    })

def set_log_levels(levels: dict) -> None:
    """
    Set log levels for different components.

    Args:
        levels: Dictionary mapping component names to log levels
    """
    global LOG_LEVELS
    LOG_LEVELS.update(levels)

def cleanup_old_logs(component_name: str, max_files: int) -> None:
    """
    Clean up old log files keeping only the most recent ones.

    Args:
        component_name: Name of the component
        max_files: Maximum number of files to keep
    """
    log_dir = Path(LOG_CONFIG['log_dir'])
    log_pattern = f"{component_name}_*.log"
    log_files = glob.glob(str(log_dir / log_pattern))
    
    # Sort files by modification time
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Remove old files
    for old_file in log_files[max_files:]:
        try:
            os.remove(old_file)
        except OSError as e:
            print(f"Error removing old log file {old_file}: {e}")

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Sets up a logger with the specified name and logging level.

    Args:
        name: The name of the logger
        log_file: Optional specific log file path

    Returns:
        logging.Logger: Configured logger instance
    """
    # Determine log level
    log_level = LOG_LEVELS.get(name, 'INFO')
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Create or get logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if enabled
    if LOG_CONFIG['save_logs']:
        # Create logs directory if it doesn't exist
        log_dir = Path(LOG_CONFIG['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate log file name if not provided
        if not log_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = str(log_dir / f"{name}_{timestamp}.log")

        # Cleanup old logs before adding new one
        cleanup_old_logs(name, LOG_CONFIG['max_log_files'])

        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation
    logger.propagate = False

    return logger