# Title LollmsComfyUI
# Licence: GPL-3.0
# Author : Paris Neo
# Forked from comfyanonymous's Comfyui nodes system
# check it out : https://github.com/comfyanonymous/ComfyUI
# Here is a copy of the LICENCE https://github.com/comfyanonymous/ComfyUI/blob/main/LICENSE
# All rights are reserved

from pathlib import Path
import sys
from lollms.app import LollmsApplication
from lollms.paths import LollmsPaths
from lollms.config import TypedConfig, ConfigTemplate, BaseConfig
import time
import io
import sys
import requests
import os
import base64
import subprocess
import time
import json
import platform
from dataclasses import dataclass
from PIL import Image, PngImagePlugin
from enum import Enum
from typing import List, Dict, Any

from ascii_colors import ASCIIColors, trace_exception
from lollms.paths import LollmsPaths
from lollms.utilities import git_pull, show_yes_no_dialog, run_script_in_env, create_conda_env, run_python_script_in_env
import subprocess
import shutil
from tqdm import tqdm


def verify_comfyui(lollms_paths:LollmsPaths):
    # Clone repository
    root_dir = lollms_paths.personal_path
    shared_folder = root_dir/"shared"
    comfyui_folder = shared_folder / "comfyui"
    return comfyui_folder.exists()

def download_file(url, folder_path, local_filename):
    # Make sure 'folder_path' exists
    folder_path.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
        with open(folder_path / local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
                progress_bar.update(len(chunk))
        progress_bar.close()

    return local_filename

def install_comfyui(lollms_app:LollmsApplication):
    root_dir = lollms_app.lollms_paths.personal_path
    shared_folder = root_dir/"shared"
    comfyui_folder = shared_folder / "comfyui"
    if comfyui_folder.exists():
        if show_yes_no_dialog("warning!","I have detected that there is a previous installation of stable diffusion.\nShould I remove it and continue installing?"):
            shutil.rmtree(comfyui_folder)
        elif show_yes_no_dialog("warning!","Continue installation?"):
            ASCIIColors.cyan("Installing comfyui conda environment with python 3.10")
            create_conda_env("comfyui","3.10")
            ASCIIColors.cyan("Done")
            return
        else:
            return

    subprocess.run(["git", "clone", "https://github.com/ParisNeo/ComfyUI.git", str(comfyui_folder)])
    subprocess.run(["git", "clone", "https://github.com/ParisNeo/ComfyUI-Manager.git", str(comfyui_folder/"custom_nodes/ComfyUI-Manager")])
    if show_yes_no_dialog("warning!","Do you want to install a model from civitai?\nIsuggest dreamshaper xl."):
        download_file("https://civitai.com/api/download/models/351306", comfyui_folder/"models/checkpoints","dreamshaperXL_v21TurboDPMSDE.safetensors")
    create_conda_env("comfyui","3.10")
    if lollms_app.config.hardware_mode in ["nvidia", "nvidia-tensorcores"]:
        run_python_script_in_env("comfyui", "-m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121")
    if lollms_app.config.hardware_mode in ["amd", "amd-noavx"]:
        run_python_script_in_env("comfyui", "-m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7")
    elif lollms_app.config.hardware_mode in ["cpu", "cpu-noavx"]:
        run_python_script_in_env("comfyui", "-m pip install --pre torch torchvision torchaudio")
    run_python_script_in_env("comfyui", f"-m pip install -r {comfyui_folder}/requirements.txt")
    
    lollms_app.comfyui = LollmsComfyUI(lollms_app)
    ASCIIColors.green("Comfyui installed successfully")


def get_comfyui(lollms_paths:LollmsPaths):
    root_dir = lollms_paths.personal_path
    shared_folder = root_dir/"shared"
    comfyui_folder = shared_folder / "comfyui"
    comfyui_script_path = comfyui_folder / "main.py"
    git_pull(comfyui_folder)
    
    if comfyui_script_path.exists():
        ASCIIColors.success("comfyui found.")
        ASCIIColors.success("Loading source file...",end="")
        # use importlib to load the module from the file path
        from lollms.services.comfyui.lollms_comfyui import LollmsComfyUI
        ASCIIColors.success("ok")
        return LollmsComfyUI

class LollmsComfyUI:
    has_controlnet = False
    def __init__(
                    self, 
                    app:LollmsApplication, 
                    wm = "Artbot", 
                    max_retries=50,
                    sampler="Euler a",
                    steps=20,               
                    use_https=False,
                    username=None,
                    password=None,
                    comfyui_base_url=None,
                    share=False,
                    wait_for_service=True
                    ):
        if comfyui_base_url=="" or comfyui_base_url=="http://127.0.0.1:8188/":
            comfyui_base_url = None
        # Get the current directory
        lollms_paths = app.lollms_paths
        self.app = app
        root_dir = lollms_paths.personal_path
        
        self.wm = wm
        # Store the path to the script
        if comfyui_base_url is None:
            self.comfyui_base_url = "http://127.0.0.1:8188/"
            if not verify_comfyui(lollms_paths):
                install_comfyui(app.lollms_paths)
        else:
            self.comfyui_base_url = comfyui_base_url

        self.comfyui_url = self.comfyui_base_url+"/comfyuiapi/v1"
        shared_folder = root_dir/"shared"
        self.comfyui_folder = shared_folder / "comfyui"
        self.output_dir = root_dir / "outputs/comfyui"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ASCIIColors.red(" _      ____  _      _      __  __  _____  _____                 __             _ ")
        ASCIIColors.red("| |    / __ \| |    | |    |  \/  |/ ____|/ ____|               / _|           (_)")
        ASCIIColors.red("| |   | |  | | |    | |    | \  / | (___ | |     ___  _ __ ___ | |_ _   _ _   _ _ ")
        ASCIIColors.red("| |   | |  | | |    | |    | |\/| |\___ \| |    / _ \| '_ ` _ \|  _| | | | | | | |")
        ASCIIColors.red("| |___| |__| | |____| |____| |  | |____) | |___| (_) | | | | | | | | |_| | |_| | |")
        ASCIIColors.red("|______\____/|______|______|_|  |_|_____/ \_____\___/|_| |_| |_|_|  \__, |\__,_|_|")
        ASCIIColors.red("                                     ______                         __/ |         ")
        ASCIIColors.red("                                    |______|                       |___/          ")

        ASCIIColors.red(" Forked from comfyanonymous's Comfyui nodes system")
        ASCIIColors.red(" Integration in lollms by ParisNeo")

        if not self.wait_for_service(1,False) and comfyui_base_url is None:
            ASCIIColors.info("Loading lollms_comfyui")
            if platform.system() == "Windows":
                ASCIIColors.info("Running on windows")
                script_path = self.comfyui_folder / "main.py"

                if share:
                    run_python_script_in_env("comfyui", str(script_path), cwd=self.comfyui_folder)
                    # subprocess.Popen("conda activate " + str(script_path) +" --share", cwd=self.comfyui_folder)
                else:
                    run_python_script_in_env("comfyui", str(script_path), cwd=self.comfyui_folder)
                    # subprocess.Popen(script_path, cwd=self.comfyui_folder)
            else:
                ASCIIColors.info("Running on linux/MacOs")
                script_path = str(self.comfyui_folder / "lollms_comfyui.sh")
                ASCIIColors.info(f"launcher path: {script_path}")
                ASCIIColors.info(f"comfyui path: {self.comfyui_folder}")

                if share:
                    run_script_in_env("comfyui","bash " + script_path +" --share", cwd=self.comfyui_folder)
                    # subprocess.Popen("conda activate " + str(script_path) +" --share", cwd=self.comfyui_folder)
                else:
                    run_script_in_env("comfyui","bash " + script_path, cwd=self.comfyui_folder)
                ASCIIColors.info("Process done")
                ASCIIColors.success("Launching Comfyui succeeded")

        # Wait until the service is available at http://127.0.0.1:8188//
        if wait_for_service:
            self.wait_for_service(max_retries=max_retries)
        else:
            ASCIIColors.warning("We are not waiting for the SD service to be up.\nThis means that you may need to wait a bit before you can use it.")

        self.default_sampler = sampler
        self.default_steps = steps

        self.session = requests.Session()

        if username and password:
            self.set_auth(username, password)
        else:
            self.check_controlnet()

    def wait_for_service(self, max_retries = 50, show_warning=True):
        url = f"{self.comfyui_base_url}"
        # Adjust this value as needed
        retries = 0

        while retries < max_retries or max_retries<0:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print("Service is available.")
                    if self.app is not None:
                        self.app.success("Comfyui Service is now available.")
                    return True
            except requests.exceptions.RequestException:
                pass

            retries += 1
            time.sleep(1)
        if show_warning:
            print("Service did not become available within the given time.")
            if self.app is not None:
                self.app.error("Comfyui Service did not become available within the given time.")
        return False