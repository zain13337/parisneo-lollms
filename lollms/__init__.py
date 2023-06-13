

__author__ = "ParisNeo"
__github__ = "https://github.com/ParisNeo/lollms"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"


from lollms.binding import LLMBinding, LOLLMSConfig
from lollms.personality import AIPersonality, MSG_TYPE
from lollms.helpers import ASCIIColors
from lollms.paths import LollmsPaths
#from lollms.binding import  LLMBinding
import importlib
from pathlib import Path


from pathlib import Path

def reset_all_installs():
    ASCIIColors.info("Removeing .install files to force reinstall")
    folder_path = Path(__file__).parent
    path = Path(folder_path)
    
    ASCIIColors.info(f"Searching files from {path}")
    for file_path in path.rglob("*.installed"):
        file_path.unlink()
        ASCIIColors.info(f"Deleted file: {file_path}")





class BindingBuilder:
    def build_binding(self, bindings_path: Path, cfg: LOLLMSConfig, force_reinstall=False)->LLMBinding:
        binding_path = Path(bindings_path) / cfg["binding_name"]
        # first find out if there is a requirements.txt file
        install_file_name = "install.py"
        install_script_path = binding_path / install_file_name
        if install_script_path.exists():
            module_name = install_file_name[:-3]  # Remove the ".py" extension
            module_spec = importlib.util.spec_from_file_location(module_name, str(install_script_path))
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            if hasattr(module, "Install"):
                module.Install(cfg, force_reinstall=force_reinstall)
        # define the full absolute path to the module
        absolute_path = binding_path.resolve()
        # infer the module name from the file path
        module_name = binding_path.stem
        # use importlib to load the module from the file path
        loader = importlib.machinery.SourceFileLoader(module_name, str(absolute_path / "__init__.py"))
        binding_module = loader.load_module()
        binding_class = getattr(binding_module, binding_module.binding_name)
        return binding_class
    

class ModelBuilder:
    def __init__(self, binding_class:LLMBinding, config:LOLLMSConfig):
        self.binding_class = binding_class
        self.model = None
        self.build_model(config) 

    def build_model(self, cfg: LOLLMSConfig):
        self.model = self.binding_class(cfg)

    def get_model(self):
        return self.model


class PersonalityBuilder:
    def __init__(self, lollms_paths:LollmsPaths, config:LOLLMSConfig, model:LLMBinding):
        self.config = config
        self.lollms_paths = lollms_paths
        self.model = model


    def build_personality(self, force_reinstall=False):
        if len(self.config["personalities"][self.config["active_personality_id"]].split("/"))==3:
            self.personality = AIPersonality(self.lollms_paths, self.lollms_paths.personalities_zoo_path / self.config["personalities"][self.config["active_personality_id"]], self.model, force_reinstall= force_reinstall)
        else:
            self.personality = AIPersonality(self.lollms_paths, self.config["personalities"][self.config["active_personality_id"]], self.model, is_relative_path=False, force_reinstall= force_reinstall)
        return self.personality
    
    def get_personality(self):
        return self.personality

