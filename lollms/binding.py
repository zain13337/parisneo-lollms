######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Supported by Nomic-AI
# license       : Apache 2.0
# Description   : 
# This is an interface class for lollms bindings.
######
from pathlib import Path
from typing import Callable
from lollms.paths import LollmsPaths
from lollms.helpers import ASCIIColors

import yaml
from tqdm import tqdm
import importlib
import subprocess
from lollms.config import TypedConfig
from lollms.main_config import LOLLMSConfig


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"


import yaml



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
    def __init__(
                    self,
                    binding_dir:Path,
                    lollms_paths:LollmsPaths,
                    config:LOLLMSConfig, 
                    binding_config:TypedConfig,
                    force_install:bool=False
                ) -> None:
        self.binding_dir            = binding_dir
        self.binding_folder_name    = binding_dir.stem
        self.lollms_paths           = lollms_paths
        self.config                 = config
        self.binding_config         = binding_config

        self.configuration_file_path = lollms_paths.personal_configuration_path/f"binding_{self.binding_folder_name}.yaml"
        self.binding_config.config.file_path = self.configuration_file_path
        if not self.configuration_file_path.exists() or force_install:
            self.install()
            self.binding_config.config.save_config()
        else:
            self.load_binding_config()

        self.models_folder = config.lollms_paths.personal_models_path / self.binding_folder_name
        self.models_folder.mkdir(parents=True, exist_ok=True)



    def install(self):
        """
        Installation procedure (to be implemented)
        """
        ASCIIColors.blue("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
        ASCIIColors.red(f"Installing {self.binding_folder_name}")
        ASCIIColors.blue("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")


    def get_model_path(self):
        """
        Retrieves the path of the model based on the configuration.

        If the model name ends with ".reference", it reads the model path from a file.
        Otherwise, it constructs the model path based on the configuration.

        Returns:
            str: The path of the model.
        """
        if self.config.model_name.endswith(".reference"):
            with open(str(self.lollms_paths.personal_models_path / f"{self.binding_folder_name}/{self.config.model_name}"), 'r') as f:
                model_path = Path(f.read())
        else:
            model_path = Path(self.lollms_paths.personal_models_path / f"{self.binding_folder_name}/{self.config.model_name}")
        return model_path

    

    def load_binding_config(self):
        """
        Load the content of local_config.yaml file.

        The function reads the content of the local_config.yaml file and returns it as a Python dictionary.

        Args:
            None

        Returns:
            dict: A dictionary containing the loaded data from the local_config.yaml file.
        """     
        self.binding_config.config.load_config()
        self.binding_config.sync()

    def save_config_file(self, path):
        """
        Load the content of local_config.yaml file.

        The function reads the content of the local_config.yaml file and returns it as a Python dictionary.

        Args:
            None

        Returns:
            dict: A dictionary containing the loaded data from the local_config.yaml file.
        """     
        self.binding_config.config.save_config(self.configuration_file_path)

    


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

