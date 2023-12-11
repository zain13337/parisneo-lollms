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


class LoLLMsCom:
    def __init__(self) -> None:
        pass
    def InfoMessage(self, content, duration:int=4, client_id=None, verbose:bool=True):
        self.notify(
                content, 
                notification_type=NotificationType.NOTIF_SUCCESS, 
                duration=duration, 
                client_id=client_id, 
                display_type=NotificationDisplayType.MESSAGE_BOX,
                verbose=verbose
            )

    def info(self, content, duration:int=4, client_id=None, verbose:bool=True):
        self.notify(
                content, 
                notification_type=NotificationType.NOTIF_SUCCESS, 
                duration=duration, 
                client_id=client_id, 
                display_type=NotificationDisplayType.TOAST,
                verbose=verbose
            )

    def warning(self, content, duration:int=4, client_id=None, verbose:bool=True):
        self.notify(
                content, 
                notification_type=NotificationType.NOTIF_WARNING, 
                duration=duration, 
                client_id=client_id, 
                display_type=NotificationDisplayType.TOAST,
                verbose=verbose
            )

    def success(self, content, duration:int=4, client_id=None, verbose:bool=True):
        self.notify(
                content, 
                notification_type=NotificationType.NOTIF_SUCCESS, 
                duration=duration, 
                client_id=client_id, 
                display_type=NotificationDisplayType.TOAST,
                verbose=verbose
            )
        
    def error(self, content, duration:int=4, client_id=None, verbose:bool=True):
        self.notify(
                content, 
                notification_type=NotificationType.NOTIF_ERROR, 
                duration=duration, 
                client_id=client_id, 
                display_type=NotificationDisplayType.TOAST,
                verbose = verbose
            )
        

    def notify(
                self, 
                content, 
                notification_type:NotificationType=NotificationType.NOTIF_SUCCESS, 
                duration:int=4, 
                client_id=None, 
                display_type:NotificationDisplayType=NotificationDisplayType.TOAST,
                verbose=True
            ):
        if verbose:
            if notification_type==NotificationType.NOTIF_SUCCESS:
                ASCIIColors.success(content)
            elif notification_type==NotificationType.NOTIF_INFO:
                ASCIIColors.info(content)
            elif notification_type==NotificationType.NOTIF_WARNING:
                ASCIIColors.warning(content)
            else:
                ASCIIColors.red(content)