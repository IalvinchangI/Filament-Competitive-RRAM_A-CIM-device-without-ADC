import logging

class LoggingColor():
    """
    Logging Color Utility
    ===========================================================================
    A utility class for handling terminal output colors and configuring 
    standardized loggers with colored formatting.

    Key Responsibilities:
    1. Color Management: Provides ANSI escape codes for terminal coloring.
    2. Logger Configuration: Instantiates and configures standardized loggers 
       to prevent duplicate handlers and enforce a consistent output format.
    
    ===========================================================================
    """

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    ORANGE = "\033[38;5;208m"
    CYAN = "\033[36m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    INFO = RESET
    WARNING = YELLOW
    ERROR = RED
    
    TITLE = BOLD + ORANGE
    SUCCESS = GREEN
    FAILED = RED

    @classmethod
    def color_text(cls, text: str, color: str) -> str:
        """
        Wrap the provided text with the specified color code.

        Args:
            text (str): The text to colorize.
            color (str): The ANSI color code (e.g., LoggingColor.GREEN).

        Returns:
            str: The formatted string with start and reset color codes.
        """
        return f"{color}{text}{cls.RESET}"
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Configure and retrieve a logger instance with color-coded formatting.

        Ensures the logger is set up with a StreamHandler and a custom 
        date/message format, preventing duplicate handlers if called multiple times.

        Args:
            name (str): The name of the logger (usually __name__).

        Returns:
            logging.Logger: A configured logger instance.
        """
        log_format = (
            f"{LoggingColor.TITLE}[%(asctime)s] [%(name)s]{LoggingColor.RESET} "
            f"%(levelname)s: %(message)s"
        )

        logger = logging.getLogger(name)
        logger.propagate = False

        if logger.hasHandlers():
            return logger

        handler = logging.StreamHandler()
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        return logger
