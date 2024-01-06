"""
project: lollms
file: lollms_extensions_infos.py 
author: ParisNeo
description: 
    This module contains a set of FastAPI routes that provide information about the Lord of Large Language and Multimodal Systems (LoLLMs) Web UI
    application. These routes are specific to handling extensions related operations.

"""
from fastapi import APIRouter, Request
from pydantic import BaseModel
import pkg_resources
from lollms.server.elf_server import LOLLMSElfServer
from ascii_colors import ASCIIColors
from lollms.utilities import load_config, trace_exception
from pathlib import Path
from typing import List
import psutil
import yaml


router = APIRouter()
lollmsElfServer = LOLLMSElfServer.get_instance()

# --------------------- Listing -------------------------------
