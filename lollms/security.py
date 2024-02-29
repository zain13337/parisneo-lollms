from fastapi import HTTPException
from ascii_colors import ASCIIColors
from pathlib import Path


def sanitize_path(path:str, allow_absolute_path:False, error_text="Absolute database path detected", exception_text="Detected an attempt of path traversal. Are you kidding me?"):
    if(".." in path):
        ASCIIColors.warning(error_text)
        raise exception_text
    if (not allow_absolute_path) and Path(path).is_absolute():
        ASCIIColors.warning(error_text)
        raise exception_text
    
def sanitize_path_from_endpoint(path:str, error_text="A suspected LFI attack detected. The path sent to the server has .. in it!", exception_text="Invalid path!"):
    if (".." in path or Path(path).is_absolute()):
        ASCIIColors.error(error_text)
        raise HTTPException(status_code=400, detail=exception_text)

def forbid_remote_access(lollmsElfServer):
    if lollmsElfServer.config.host!="localhost" and lollmsElfServer.config.host!="127.0.0.1":
        return {"status":False,"error":"Code execution is blocked when the server is exposed outside for very obvious reasons!"}

