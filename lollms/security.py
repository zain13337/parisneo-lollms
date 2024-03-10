from fastapi import HTTPException
from ascii_colors import ASCIIColors
from pathlib import Path
from typing import List
import os
import re

def sanitize_path(path:str, allow_absolute_path:bool=False, error_text="Absolute database path detected", exception_text="Detected an attempt of path traversal. Are you kidding me?"):
    if path is None:
        return path
    if(".." in path):
        ASCIIColors.warning(error_text)
        raise Exception(exception_text)
    if (not allow_absolute_path) and Path(path).is_absolute():
        ASCIIColors.warning(error_text)
        raise Exception(exception_text)
    return path
    
def sanitize_path_from_endpoint(path:str, error_text="A suspected LFI attack detected. The path sent to the server has .. in it!", exception_text="Invalid path!"):
    if path is None:
        return path
    if (".." in path or Path(path).is_absolute()):
        ASCIIColors.error(error_text)
        raise HTTPException(status_code=400, detail=exception_text)
    return path


def sanitize_path_from_endpoint(path: str, error_text="A suspected LFI attack detected. The path sent to the server has suspicious elements in it!", exception_text="Invalid path!"):
    if path is None:
        return path
    
    # Regular expression to detect patterns like "...." and multiple forward slashes
    suspicious_patterns = re.compile(r'(\.\.+)|(/+/)')
    
    if suspicious_patterns.search(path) or Path(path).is_absolute():
        ASCIIColors.error(error_text)
        raise HTTPException(status_code=400, detail=exception_text)
    
    return path

def forbid_remote_access(lollmsElfServer):
    if lollmsElfServer.config.host!="localhost" and lollmsElfServer.config.host!="127.0.0.1":
        raise Exception("This functionality is forbidden if the server is exposed")

def validate_path(path, allowed_paths:List[str|Path]):
    # Convert the path to an absolute path
    abs_path = os.path.realpath(str(path))

    # Iterate over the allowed paths
    for allowed_path in allowed_paths:
        # Convert the allowed path to an absolute path
        abs_allowed_path = os.path.realpath(allowed_path)

        # Check if the absolute path starts with the absolute allowed path
        if abs_path.startswith(abs_allowed_path):
            return True

    # If the path is not within any of the allowed paths, return False
    return False


if __name__=="__main__":
    sanitize_path_from_endpoint("main")
    sanitize_path_from_endpoint("cat/main")
    print("Main passed")
    sanitize_path_from_endpoint(".../user")
    print("hi")