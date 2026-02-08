import logging

class LoggingColor():
    """
    A utility class for handling terminal output colors and configuring 
    standardized loggers with colored formatting.
    """

    # --- ANSI Escape Codes for Terminal Colors ---
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    ORANGE = "\033[38;5;208m"  # Extended 256-color code for Orange
    RESET = "\033[0m"          # Resets all attributes (color, bold, etc.)
    BOLD = "\033[1m"

    # --- Semantic Color Mapping ---
    # Maps specific logging levels or statuses to their corresponding colors
    INFO = RESET
    WARNING = YELLOW
    ERROR = RED
    
    # Custom formatting styles
    TITLE = BOLD + ORANGE
    SUCCESS = GREEN
    FAILED = RED

    @classmethod
    def color_text(cls, text, color):
        """
        Wraps the provided text with the specified color code and resets it afterwards.

        Args:
            text (str): The text to colorize.
            color (str): The ANSI color code (e.g., LoggingColor.GREEN).

        Returns:
            str: The formatted string with start and reset color codes.
        """
        return f"{color}{text}{cls.RESET}"
    
    @classmethod
    def get_logger(cls, name: str):
        """
        Configures and retrieves a logger instance with a specific color-coded format.
        
        This method ensures the logger is set up with a StreamHandler and a 
        custom date/message format. It avoids adding duplicate handlers if 
        the logger is initialized multiple times.

        Args:
            name (str): The name of the logger (usually __name__).

        Returns:
            logging.Logger: A configured logger instance.
        """
        # Define the log format: [Timestamp] [LoggerName] Level: Message
        # Uses TITLE color for the metadata prefix
        log_format = (
            f"{LoggingColor.TITLE}[%(asctime)s] [%(name)s]{LoggingColor.RESET} "
            f"%(levelname)s: %(message)s"
        )

        logger = logging.getLogger(name)
        logger.propagate = False

        # Check if handlers already exist to prevent duplicate log outputs
        # (A common issue when get_logger is called multiple times)
        if logger.hasHandlers():
            return logger

        # Create console handler
        handler = logging.StreamHandler()
        
        # Set formatter with specific date format (YYYY-MM-DD HH:MM:SS)
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        
        # Attach handler to logger
        logger.addHandler(handler)
        
        # Default logging level is set to DEBUG
        logger.setLevel(logging.DEBUG)

        return logger
