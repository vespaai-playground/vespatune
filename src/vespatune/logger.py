"""
VespaTune Logger - A beautiful, rich-enabled logging system.
"""

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme


# Custom theme for VespaTune
VESPATUNE_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow bold",
        "error": "red bold",
        "debug": "dim white",
        "success": "green bold",
        "timestamp": "dim cyan",
        "message": "white",
        "highlight": "magenta bold",
    }
)


class VespatuneLogger:
    """
    A rich-enabled logger for VespaTune with beautiful formatting.

    Features:
    - Colored output with rich formatting
    - Configurable log levels
    - Optional file logging
    - Success messages support
    - Contextual logging with panels
    """

    def __init__(
        self,
        name: str = "vespatune",
        level: int = logging.INFO,
        show_time: bool = True,
        show_path: bool = False,
        rich_tracebacks: bool = True,
    ):
        self.name = name
        self.console = Console(theme=VESPATUNE_THEME, stderr=True)
        self._level = level
        self._show_time = show_time

        # Create the underlying Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers.clear()

        # Rich handler for beautiful console output
        rich_handler = RichHandler(
            console=self.console,
            show_time=show_time,
            show_path=show_path,
            rich_tracebacks=rich_tracebacks,
            tracebacks_show_locals=True,
            markup=True,
            log_time_format="[%X]",
        )
        rich_handler.setLevel(level)
        self._logger.addHandler(rich_handler)

        # Prevent propagation to root logger
        self._logger.propagate = False

    def set_level(self, level: int) -> None:
        """Set the logging level."""
        self._level = level
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)

    def enable_file_logging(
        self,
        filepath: str,
        level: Optional[int] = None,
        fmt: str = "%(asctime)s | %(levelname)-8s | %(message)s",
    ) -> None:
        """Enable logging to a file."""
        file_handler = logging.FileHandler(filepath)
        file_handler.setLevel(level or self._level)
        file_handler.setFormatter(logging.Formatter(fmt))
        self._logger.addHandler(file_handler)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log a debug message."""
        self._logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Log an info message."""
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Log a warning message."""
        self._logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Log an error message."""
        self._logger.error(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs) -> None:
        """Log an exception with traceback."""
        self._logger.exception(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """Log a critical message."""
        self._logger.critical(message, *args, **kwargs)

    def success(self, message: str) -> None:
        """Log a success message with a checkmark."""
        text = Text()
        text.append("✓ ", style="success")
        text.append(message, style="success")
        self.console.print(text)

    def panel(
        self,
        message: str,
        title: Optional[str] = None,
        style: str = "cyan",
        expand: bool = False,
    ) -> None:
        """Display a message in a panel."""
        self.console.print(Panel(message, title=title, style=style, expand=expand))

    def header(self, message: str, style: str = "bold magenta") -> None:
        """Print a header-style message."""
        self.console.print()
        self.console.rule(f"[{style}]{message}[/{style}]")
        self.console.print()

    def step(self, step_num: int, message: str, total: Optional[int] = None) -> None:
        """Log a step in a process."""
        if total:
            prefix = f"[cyan][{step_num}/{total}][/cyan]"
        else:
            prefix = f"[cyan][{step_num}][/cyan]"
        self.console.print(f"{prefix} {message}")

    def highlight(self, message: str) -> None:
        """Print a highlighted message."""
        self.console.print(f"[highlight]→ {message}[/highlight]")

    def divider(self, style: str = "dim") -> None:
        """Print a divider line."""
        self.console.rule(style=style)


def get_logger(
    name: str = "vespatune",
    level: int = logging.INFO,
    show_time: bool = True,
    show_path: bool = False,
) -> VespatuneLogger:
    """
    Get or create a VespaTune logger instance.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        show_time: Whether to show timestamps
        show_path: Whether to show file paths

    Returns:
        VespatuneLogger instance
    """
    return VespatuneLogger(
        name=name,
        level=level,
        show_time=show_time,
        show_path=show_path,
    )


# Default logger instance for module-level imports
logger = get_logger()
