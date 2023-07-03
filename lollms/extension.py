# File : extension.py
# Author : ParisNeo
# Description :
# This is the main class to be imported by the extension
# it gives your code access to the model, the callback functions, the model conditionning etc
from lollms.helpers import ASCIIColors
from lollms.config import BaseConfig
from enum import Enum


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms-webui"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"


class EXTENSION_TYPE(Enum):
    EXTENSION_TYPE_STAND_ALONE = 0 # An extension that has access to the current personality and model but has an independant 


class Extension():
    def __init__(self, metadata_file_path:str, app) -> None:
        self.app = app
        self.metadata_file_path = metadata_file_path
        self.config = BaseConfig(file_path=metadata_file_path)
        self.config.load_config()

    def install(self):
        """
        Installation procedure (to be implemented)
        """
        ASCIIColors.blue("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
        ASCIIColors.red(f"Installing {self.binding_folder_name}")
        ASCIIColors.blue("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")

    def get_ui():
        """
        Get user interface of the extension
        """
        return "<p>This is a ui extension template</p>"