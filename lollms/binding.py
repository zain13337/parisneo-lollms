######
# Project       : GPT4ALL-UI
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Supported by Nomic-AI
# license       : Apache 2.0
# Description   : 
# This is an interface class for GPT4All-ui bindings.
######
from pathlib import Path
from typing import Callable
from lollms.helpers import BaseConfig, ASCIIColors
from lollms.paths import LollmsPaths

import inspect
import yaml
import sys
from tqdm import tqdm
import urllib.request
import importlib
import shutil
import subprocess


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"


import yaml

DEFAULT_CONFIG = {
    # =================== Lord Of Large Language Models Configuration file =========================== 
    "version": 5,
    "binding_name": "llama_cpp_official",
    "model_name": "Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_0.bin",

    # Host information
    "host": "localhost",
    "port": 9600,

    # Genreration parameters 
    "seed": -1,
    "n_predict": 1024,
    "ctx_size": 2048,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 0.95,
    "repeat_last_n": 40,
    "repeat_penalty": 1.2,

    "n_threads": 8,

    #Personality parameters
    "personalities": ["english/generic/lollms"],
    "active_personality_id": 0,
    "override_personality_model_parameters": False, #if true the personality parameters are overriden by those of the configuration (may affect personality behaviour) 

    "user_name": "user",

}


class LOLLMSConfig(BaseConfig):
    def __init__(self, file_path=None, lollms_paths:LollmsPaths = None):
        super().__init__(["file_path", "config", "lollms_paths"])
                
        if file_path:
            self.file_path = Path(file_path)
        else:
            self.file_path = None

        if file_path is not None:
            self.load_config(file_path)
        else:
            self.config = DEFAULT_CONFIG.copy()

        if lollms_paths is None:
            self.lollms_paths = LollmsPaths()
        else:
            self.lollms_paths = lollms_paths
        

    @staticmethod
    def autoload(lollms_paths, config_path:str=None):
        # Configuration loading part
        original_cfg_path = lollms_paths.default_cfg_path
        if config_path is None:
            local = lollms_paths.personal_configuration_path / "local_config.yaml"
            if not local.exists():
                shutil.copy(original_cfg_path, local)
            cfg_path = local
        else:
            cfg_path = config_path

        if cfg_path.exists():
            original_config = LOLLMSConfig(original_cfg_path,lollms_paths)
            config = LOLLMSConfig(cfg_path,lollms_paths)
            if "version" not in config or int(config["version"])<int(original_config["version"]):
                #Upgrade old configuration files to new format
                ASCIIColors.error("Configuration file is very old.\nReplacing with default configuration")
                _, added, removed = config.sync_cfg(original_config)
                print(f"Added entries : {added}, removed entries:{removed}")
                config.save_config(cfg_path)
        else:
            config = LOLLMSConfig()
        
        return config
            
    def sync_cfg(self, default_config):
        """Syncs a configuration with the default configuration

        Args:
            default_config (_type_): _description_
            config (_type_): _description_

        Returns:
            _type_: _description_
        """
        added_entries = []
        removed_entries = []

        # Ensure all fields from default_config exist in config
        for key, value in default_config.config.items():
            if key not in self:
                self[key] = value
                added_entries.append(key)

        # Remove fields from config that don't exist in default_config
        for key in list(self.config.keys()):
            if key not in default_config.config:
                del self.config[key]
                removed_entries.append(key)

        self["version"]=default_config["version"]
        
        return self, added_entries, removed_entries

    def get_model_path_infos(self):
        return f"personal_models_path: {self.lollms_paths.personal_models_path}\nBinding name:{self.binding_name}\nModel name:{self.model_name}"

    def get_personality_path_infos(self):
        return f"personalities_zoo_path: {self.lollms_paths.personalities_zoo_path}\nPersonalities:{self.personalities}\nActive personality id:{self.active_personality_id}"

    def get_model_full_path(self):
        try:
            return self.lollms_paths.personal_models_path/self.binding_name/self.model_name
        except:
            return None
    def check_model_existance(self):
        try:
            model_path = self.lollms_paths.personal_models_path/self.binding_name/self.model_name
            return model_path.exists()
        except Exception as ex:
            print(f"Exception in checking model existance: {ex}")
            return False 
           
    def download_model(self, url, binding, callback = None):
        folder_path = self.lollms_paths.personal_models_path/self.binding_name
        model_name  = url.split("/")[-1]
        model_full_path = (folder_path / model_name)
        if binding is not None and hasattr(binding,'download_model'):
            binding.download_model(url, model_full_path, callback)
        else:

            # Check if file already exists in folder
            if model_full_path.exists():
                print("File already exists in folder")
            else:
                # Create folder if it doesn't exist
                folder_path.mkdir(parents=True, exist_ok=True)
                progress_bar = tqdm(total=None, unit="B", unit_scale=True, desc=f"Downloading {url.split('/')[-1]}")
                # Define callback function for urlretrieve
                def report_progress(block_num, block_size, total_size):
                    progress_bar.total=total_size
                    progress_bar.update(block_size)
                # Download file from URL to folder
                try:
                    urllib.request.urlretrieve(url, folder_path / url.split("/")[-1], reporthook=report_progress if callback is None else callback)
                    print("File downloaded successfully!")
                except Exception as e:
                    print("Error downloading file:", e)
                    sys.exit(1)

    def reference_model(self, path):
        path = str(path).replace("\\","/")
        folder_path = self.lollms_paths.personal_models_path/self.binding_name
        model_name  = path.split("/")[-1]+".reference"
        model_full_path = (folder_path / model_name)

        # Check if file already exists in folder
        if model_full_path.exists():
            print("File already exists in folder")
        else:
            # Create folder if it doesn't exist
            folder_path.mkdir(parents=True, exist_ok=True)
            with open(model_full_path,"w") as f:
                f.write(path)
            print("Reference created, please make sure you don't delete the file or you will have broken link")


class BindingInstaller:
    def __init__(self, config: LOLLMSConfig) -> None:
        self.config = config

    def reinstall_pytorch_with_cuda(self):
        result = subprocess.run(["pip", "install", "--upgrade", "torch", "torchvision", "torchaudio", "--no-cache-dir", "--index-url", "https://download.pytorch.org/whl/cu117"])
        if result.returncode != 0:
            ASCIIColors.warning("Couldn't find Cuda build tools on your PC. Reverting to CPU.")
            result = subprocess.run(["pip", "install", "--upgrade", "torch", "torchvision", "torchaudio", "--no-cache-dir"])
            if result.returncode != 0:
                ASCIIColors.error("Couldn't install pytorch !!")
            else:
                ASCIIColors.error("Pytorch installed successfully!!")


class LLMBinding:
   
    file_extension='*.bin'
    binding_path = Path(__file__).parent
    def __init__(self, config:LOLLMSConfig, inline:bool) -> None:
        self.config = config
        self.inline = inline

    def load_config_file(self, path):
        """
        Load the content of local_config.yaml file.

        The function reads the content of the local_config.yaml file and returns it as a Python dictionary.

        Args:
            None

        Returns:
            dict: A dictionary containing the loaded data from the local_config.yaml file.
        """     
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        return data


    def generate(self, 
                 prompt:str,                  
                 n_predict: int = 128,
                 callback: Callable[[str], None] = None,
                 verbose: bool = False,
                 **gpt_params ):
        """Generates text out of a prompt
        This should ber implemented by child class

        Args:
            prompt (str): The prompt to use for generation
            n_predict (int, optional): Number of tokens to prodict. Defaults to 128.
            callback (Callable[[str], None], optional): A callback function that is called everytime a new text element is generated. Defaults to None.
            verbose (bool, optional): If true, the code will spit many informations about the generation process. Defaults to False.
        """
        pass
    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        return prompt.split(" ")

    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        return " ".join(tokens_list)

    @staticmethod
    def list_models(config:dict, root_path="."):
        """Lists the models for this binding
        """
        root_path = Path(root_path)
        models_dir =(root_path/'models')/config["binding_name"]  # replace with the actual path to the models folder
        return [f.name for f in models_dir.glob(LLMBinding.file_extension)]

    @staticmethod    
    def install_binding(binding_path, config:LOLLMSConfig):
        install_file_name = "install.py"
        install_script_path = binding_path / install_file_name
        if install_script_path.exists():
            module_name = install_file_name[:-3]  # Remove the ".py" extension
            module_spec = importlib.util.spec_from_file_location(module_name, str(install_script_path))
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            if hasattr(module, "Install"):
                module.Install(config)

    # To implement by children
    # @staticmethod
    # def get_available_models():

