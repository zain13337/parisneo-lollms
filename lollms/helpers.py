import traceback
from ascii_colors import ASCIIColors
from enum import Enum
class NotificationType(Enum):
    """Notification types."""
    
    NOTIF_ERROR = 0
    """This is an error notification."""
    
    NOTIF_SUCCESS = 1
    """This is a success notification."""

    NOTIF_INFO = 2
    """This is an information notification."""

    NOTIF_WARNING = 3
    """This is a warining notification."""

class NotificationDisplayType(Enum):
    """Notification display types."""
    
    TOAST = 0
    """This is a toast."""
    
    MESSAGE_BOX = 1
    """This is a message box."""

def get_trace_exception(ex):
    """
    Traces an exception (useful for debug) and returns the full trace of the exception
    """
    # Catch the exception and get the traceback as a list of strings
    traceback_lines = traceback.format_exception(type(ex), ex, ex.__traceback__)

    # Join the traceback lines into a single string
    traceback_text = ''.join(traceback_lines)        
    return traceback_text

def trace_exception(ex):
    """
    Traces an exception (useful for debug)
    """
    ASCIIColors.error(get_trace_exception(ex))

