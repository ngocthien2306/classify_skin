import logging
import os
from datetime import datetime

def get_logger(name, log_dir='logs'):
    """
    Create a logger instance
    
    Args:
        name (str): Logger name (usually __name__ from the calling module)
        log_dir (str): Directory to store log files
        
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger instance
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    log_file = os.path.join(
        log_dir, 
        f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger