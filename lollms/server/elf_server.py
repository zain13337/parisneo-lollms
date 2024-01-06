"""
File: lollms_web_ui.py
Author: ParisNeo
Description: Singleton class for the LoLLMS web UI.

This class provides a singleton instance of the LoLLMS web UI, allowing access to its functionality and data across multiple endpoints.
"""

from lollms.app import LollmsApplication
from lollms.main_config import LOLLMSConfig
from lollms.paths import LollmsPaths
from pathlib import Path

class LOLLMSElfServer(LollmsApplication):
    __instance = None

    @staticmethod
    def build_instance(
        config: LOLLMSConfig,
        lollms_paths: LollmsPaths,
        load_binding=True,
        load_model=True,
        load_voice_service=True,
        load_sd_service=True,
        try_select_binding=False,
        try_select_model=False,
        callback=None,
        socketio = None
    ):
        if LOLLMSElfServer.__instance is None:
            LOLLMSElfServer(
                config,
                lollms_paths,
                load_binding=load_binding,
                load_model=load_model,
                load_sd_service=load_sd_service,
                load_voice_service=load_voice_service,
                try_select_binding=try_select_binding,
                try_select_model=try_select_model,
                callback=callback,
                socketio=socketio
            )
        return LOLLMSElfServer.__instance
    @staticmethod
    def get_instance():
        return LOLLMSElfServer.__instance

    def __init__(
        self,
        config: LOLLMSConfig,
        lollms_paths: LollmsPaths,
        load_binding=True,
        load_model=True,
        load_voice_service=True,
        load_sd_service=True,
        try_select_binding=False,
        try_select_model=False,
        callback=None,
        socketio=None
    ) -> None:
        super().__init__(
            "LOLLMSElfServer",
            config,
            lollms_paths,
            load_binding=load_binding,
            load_model=load_model,
            load_sd_service=load_sd_service,
            load_voice_service=load_voice_service,
            try_select_binding=try_select_binding,
            try_select_model=try_select_model,
            callback=callback,
            socketio=socketio
        )
        if LOLLMSElfServer.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            LOLLMSElfServer.__instance = self

    # Other methods and properties of the LoLLMSWebUI singleton class
    def find_extension(self, path:Path, filename:str, exts:list)->Path:
        for ext in exts:
            full_path = path/(filename+ext)
            if full_path.exists():
                return full_path
        return None
