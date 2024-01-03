from fastapi import APIRouter
import pkg_resources
from lollms.server.elf_server import LOLLMSElfServer
from ascii_colors import ASCIIColors

router = APIRouter()
lollmsWebUI = LOLLMSElfServer.get_instance()

@router.get("/version")
def get_version():
    """
    Get the version of the lollms package.

    Returns:
        dict: A dictionary containing the version of the lollms package.
    """
    version = pkg_resources.get_distribution('lollms').version
    ASCIIColors.yellow("Lollms version: " + version)
    return {"version": version}