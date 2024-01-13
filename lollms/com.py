from ascii_colors import ASCIIColors
import socketio
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

    YESNO_MESSAGE = 2
    """This is a yes not messagebox."""

    SHOW_BLOCKING_MESSAGE = 3
    """This shows a blocking messagebox."""

    HIDE_BLOCKING_MESSAGE = 4
    """This hides a blocking messagebox."""


class LoLLMsCom:
    def __init__(self, socketio:socketio.AsyncServer=None) -> None:
        self.socketio= socketio

        
    def InfoMessage(self, content, client_id=None, verbose:bool=True):
        self.notify(
                content, 
                notification_type=NotificationType.NOTIF_SUCCESS, 
                duration=0, 
                client_id=client_id, 
                display_type=NotificationDisplayType.MESSAGE_BOX,
                verbose=verbose
            )
    def ShowBlockingMessage(self, content, client_id=None, verbose:bool=True):
        self.notify(
                content, 
                notification_type=NotificationType.NOTIF_SUCCESS, 
                duration=0, 
                client_id=client_id, 
                display_type=NotificationDisplayType.SHOW_BLOCKING_MESSAGE,
                verbose=verbose
            )        
        
    def HideBlockingMessage(self, client_id=None, verbose:bool=True):
        self.notify(
                "", 
                notification_type=NotificationType.NOTIF_SUCCESS, 
                duration=0, 
                client_id=client_id, 
                display_type=NotificationDisplayType.HIDE_BLOCKING_MESSAGE,
                verbose=verbose
            )        



    def YesNoMessage(self, content, duration:int=4, client_id=None, verbose:bool=True):
        infos={
            "wait":True,
            "result":False
        }
        @self.socketio.on('yesNoRes')
        def yesnores(result):
            infos["result"] = result["yesRes"]
            infos["wait"]=False

        self.notify(
                content, 
                notification_type=NotificationType.NOTIF_SUCCESS, 
                duration=duration, 
                client_id=client_id, 
                display_type=NotificationDisplayType.YESNO_MESSAGE,
                verbose=verbose
            )
        # wait
        ASCIIColors.yellow("Waiting for yes no question to be answered")
        while infos["wait"]:
            self.socketio.sleep(1)
        return infos["result"]

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
                content:str, 
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