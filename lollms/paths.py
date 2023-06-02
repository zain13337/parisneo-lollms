from pathlib import Path
import shutil
from lollms.helpers import ASCIIColors
from lollms.helpers import BaseConfig

lollms_path = Path(__file__).parent
lollms_default_cfg_path = lollms_path / "configs/config.yaml"
lollms_bindings_zoo_path = lollms_path / "bindings_zoo"
lollms_personalities_zoo_path = lollms_path / "personalities_zoo"


# Now we speify the personal folders
class LollmsPaths:
    def __init__(self, lollms_path=None, personal_path=None, custom_default_cfg_path=None):
        if lollms_path is None:
            lollms_path = Path(__file__).parent
        else:
            lollms_path = Path(lollms_path)
        if personal_path is None:
            personal_path = Path.home() / "Documents/lollms"
        else:
            personal_path = Path(personal_path)
        
        if custom_default_cfg_path is not None:
            self.default_cfg_path = Path(custom_default_cfg_path)
        else:
            self.default_cfg_path = lollms_path / "configs/config.yaml"
        self.bindings_zoo_path = lollms_path / "bindings_zoo"
        self.personalities_zoo_path = lollms_path / "personalities_zoo"

        self.personal_path = personal_path
        self.personal_configuration_path = personal_path / "configs"
        self.personal_models_path = personal_path / "models"

        self.create_directories()
        self.copy_default_config()

    def change_personal_path(self, path):
        self.personal_path = path

    def create_directories(self):
        self.personal_path.mkdir(parents=True, exist_ok=True)
        self.personal_configuration_path.mkdir(parents=True, exist_ok=True)
        self.personal_models_path.mkdir(parents=True, exist_ok=True)

    def copy_default_config(self):
        local_config_path = self.personal_configuration_path / "local_config.yaml"
        if not local_config_path.exists():
            shutil.copy(self.default_cfg_path, str(local_config_path))


    @staticmethod
    def find_paths(force_local=False, custom_default_cfg_path=None):
        lollms_path = Path(__file__).parent
        global_paths_cfg = Path("./global_paths_cfg.yaml")
        if global_paths_cfg.exists():
            try:
                cfg = BaseConfig()
                cfg.load_config(global_paths_cfg)
                lollms_path = cfg.lollms_path
                lollms_personal_path = cfg.lollms_personal_path
                return LollmsPaths(lollms_path, lollms_personal_path, custom_default_cfg_path=custom_default_cfg_path)
            except Exception as ex:
                print(f"{ASCIIColors.color_red}Global paths configuration file found but seems to be corrupted{ASCIIColors.color_reset}")
                print("Couldn't find your personal data path!")
                cfg.lollms_path = Path(__file__).parent
                cfg.lollms_personal_path = input("Please specify the folder where your configuration files, your models and your custom personalities need to be stored:")
                cfg.save_config(global_paths_cfg)
                lollms_path = cfg.lollms_path
                lollms_personal_path = cfg.lollms_personal_path
                return LollmsPaths(lollms_path, lollms_personal_path, custom_default_cfg_path=custom_default_cfg_path)
        else:
            # if the app is not forcing a specific path, then try to find out if the default installed library has specified a default path
            global_paths_cfg = lollms_path/"global_paths_cfg.yaml"
            if global_paths_cfg.exists():
                try:
                    cfg = BaseConfig()
                    cfg.load_config(global_paths_cfg)
                    lollms_path = cfg.lollms_path
                    lollms_personal_path = cfg.lollms_personal_path
                    return LollmsPaths(lollms_path, lollms_personal_path, custom_default_cfg_path=custom_default_cfg_path)
                except Exception as ex:
                    print(f"{ASCIIColors.color_red}Global paths configuration file found but seems to be corrupted{ASCIIColors.color_reset}")
                    print("Couldn't find your personal data path!")
                    cfg.lollms_path = Path(__file__).parent
                    cfg.lollms_personal_path = input("Please specify the folder where your configuration files, your models and your custom personalities need to be stored:")
                    cfg.save_config(global_paths_cfg)
                    lollms_path = cfg.lollms_path
                    lollms_personal_path = cfg.lollms_personal_path
                    return LollmsPaths(lollms_path, lollms_personal_path, custom_default_cfg_path=custom_default_cfg_path)
            else: # First time 
                print(f"{ASCIIColors.color_green}Welcome! It seems this is your first use of the new lollms app.{ASCIIColors.color_reset}")
                print(f"To make it clear where your data are stored, we now give the user the choice where to put its data.")
                print(f"This allows you to mutualize models which are heavy, between multiple lollms compatible apps.")
                print(f"You can change this at any tome using the lollms-update_path script or by simply change the content of the global_paths_cfg.yaml file.")
                print(f"Please provide a folder to store your configurations files, your models and your personal data (database, custom personalities etc).")
                cfg = BaseConfig(config={
                    "lollms_path":str(Path(__file__).parent),
                    "lollms_personal_path":str(Path.home()/"Documents/lollms")
                })

                cfg.lollms_personal_path = input(f"Folder path: ({cfg.lollms_personal_path}):")
                if cfg.lollms_personal_path=="":
                    cfg.lollms_personal_path = str(Path.home()/"Documents/lollms")

                print(f"Selected: {cfg.lollms_personal_path}")
                pp= Path(cfg.lollms_personal_path)
                if not pp.exists():
                    try:
                        pp.mkdir(parents=True)
                    except:
                        print(f"{ASCIIColors.color_red}It seams there is an error in the path you rovided{ASCIIColors.color_reset}")
                        return None
                if force_local:
                    global_paths_cfg = Path("./global_paths_cfg.yaml")
                else:
                    global_paths_cfg = lollms_path/"global_paths_cfg.yaml"
                cfg.save_config(global_paths_cfg)
                    
                return LollmsPaths(cfg.lollms_path, cfg.lollms_personal_path, custom_default_cfg_path=custom_default_cfg_path)
            
            
    @staticmethod     
    def reset_configs():
        lollms_path = Path(__file__).parent
        global_paths_cfg = Path("./global_paths_cfg.yaml")
        if global_paths_cfg.exists():
            ASCIIColors.error("Resetting local settings")
            global_paths_cfg.unlink()
            return
        global_paths_cfg = lollms_path/"global_paths_cfg.yaml"
        if global_paths_cfg.exists():
            ASCIIColors.error("Resetting global settings")
            global_paths_cfg.unlink()


# Try to find out if the application has a global paths config
# If the application has a local configuration file that points us to the paths configuration then load it
"""
global_paths_cfg = Path("./global_paths_cfg.yaml")
if global_paths_cfg.exists():
    cfg = BaseConfig()
    cfg.load_config(global_paths_cfg)
    try:
        lollms_personal_path = cfg.global_path
    except Exception as ex:
        print("Couldn't find your global path!")
        cfg.global_path = input("Please specify the folder where your configuration files, your models and your custom personalities need to be stored:")
        lollms_personal_path = cfg.global_path
        cfg.save_config(global_paths_cfg)
else:
    # if the app is not forcing a specific path, then try to find out if the default installed library has specified a default path
    global_paths_cfg = lollms_path/"global_paths_cfg.yaml"
    if global_paths_cfg.exists():
        cfg = BaseConfig()
        cfg.load_config(global_paths_cfg)
        try:
            lollms_personal_path = cfg.global_path
        except Exception as ex:
            print("Couldn't find your global path!")
            cfg.global_path = input("Please specify the folder where your configuration files, your models and your custom personalities need to be stored:")
            lollms_personal_path = cfg.global_path
            cfg.save_config(global_paths_cfg)

    lollms_personal_path = Path.home()/"Documents/lollms"

lollms_personal_configuration_path = lollms_personal_path/"configs"
lollms_personal_models_path = lollms_personal_path/"models"

lollms_personal_path.mkdir(parents=True, exist_ok=True)
lollms_personal_configuration_path.mkdir(parents=True, exist_ok=True)
lollms_personal_models_path.mkdir(parents=True, exist_ok=True)

if not(lollms_personal_configuration_path/"local_config.yaml").exists():
    shutil.copy(lollms_path / "configs/config.yaml", str(lollms_personal_configuration_path/"local_config.yaml"))


"""
