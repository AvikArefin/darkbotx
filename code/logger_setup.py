import logging
import os
from rich.logging import RichHandler

def setup_logger(name: str, log_file: str = "run.log") -> logging.Logger:
    """Sets up a logger with Rich console output and standard file logging."""
    
    logger = logging.getLogger(name)
    # Set base level to INFO to capture everything for the file
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # 1. File Handler (Standard text)
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        fh = logging.FileHandler(log_file, mode='a')
        file_formatter = logging.Formatter(
            "%(filename)s [%(levelname)s] %(asctime)s.%(msecs)03d %(funcName)s:%(lineno)d %(message)s",
            datefmt="%H:%M:%S %d-%m-%y"
        )
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

        # 2. Rich Console Handler (Filtered to WARNING and above)
        # Rich automatically handles colors, time, and level formatting.
        rh = RichHandler(
            level=logging.DEBUG, 
            rich_tracebacks=True, 
            markup=True
        )
        # We give Rich a simpler format because it adds its own timestamp and level blocks
        rich_formatter = logging.Formatter("%(message)s")
        rh.setFormatter(rich_formatter)
        
        logger.addHandler(rh)

    return logger
