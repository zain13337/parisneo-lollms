from lollms.main_config import LOLLMSConfig
from lollms.helpers import ASCIIColors
from lollms.paths import LollmsPaths
from lollms.personality import PersonalityBuilder
from lollms.binding import LLMBinding, BindingBuilder, ModelBuilder
from lollms.config import InstallOption
import traceback
from lollms.terminal import MainMenu

class LollmsApplication:
    def __init__(self, app_name:str, config:LOLLMSConfig, lollms_paths:LollmsPaths) -> None:
        """
        Creates a LOLLMS Application
        """
        self.app_name               = app_name
        self.config                 = config
        self.lollms_paths           = lollms_paths

        self.menu                   = MainMenu(self)
        self.mounted_personalities  = []
        self.personality            = None

        if self.config.binding_name is None:
            ASCIIColors.warning(f"No binding selected")
            ASCIIColors.info("Please select a valid model or install a new one from a url")
            self.menu.select_binding()

        try:
            self.binding            = self.load_binding()
        except Exception as ex:
            ASCIIColors.error(f"Failed to load binding.\nReturned exception: {ex}")

        if self.binding is not None:
            if self.config.model_name is None:
                ASCIIColors.warning(f"No model selected")
                print("Please select a valid binding")
                self.menu.select_model()
            try:
                self.model          = self.load_model()
            except Exception as ex:
                ASCIIColors.error(f"Failed to load model.\nReturned exception: {ex}")
        else:
            self.binding = None
        self.mount_personalities()


    def trace_exception(self, ex):
        # Catch the exception and get the traceback as a list of strings
        traceback_lines = traceback.format_exception(type(ex), ex, ex.__traceback__)

        # Join the traceback lines into a single string
        traceback_text = ''.join(traceback_lines)        
        ASCIIColors.error(traceback_text)

    def load_binding(self):
        try:
            binding = BindingBuilder().build_binding(self.config, self.lollms_paths)
        except Exception as ex:
            print(ex)
            print(f"Couldn't find binding. Please verify your configuration file at {self.configuration_path} or use the next menu to select a valid binding")
            print(f"Trying to reinstall binding")
            binding = BindingBuilder().build_binding(self.config, self.lollms_paths,installation_option=InstallOption.FORCE_INSTALL)
        return binding    
    
    def load_model(self):
        try:
            model = ModelBuilder(self.binding).get_model()
            for personality in self.mounted_personalities:
                if personality is not None:
                    personality.model = model
        except Exception as ex:
            ASCIIColors.error(f"Couldn't load model. Please verify your configuration file at {self.lollms_paths.personal_configuration_path} or use the next menu to select a valid model")
            ASCIIColors.error(f"Binding returned this exception : {ex}")
            ASCIIColors.error(f"{self.config.get_model_path_infos()}")
            print("Please select a valid model or install a new one from a url")
            model = None

        return model


    def mount_personality(self, id:int):
        try:
            self.personality = PersonalityBuilder(self.lollms_paths, self.config, self.model).build_personality(id)
            self.config.active_personality_id=len(self.config.personalities)-1
            if self.personality.model is not None:
                self.cond_tk = self.personality.model.tokenize(self.personality.personality_conditioning)
                self.n_cond_tk = len(self.cond_tk)
                ASCIIColors.success(f"Personality  {self.personality.name} mounted successfully")
            else:
                ASCIIColors.success(f"Personality  {self.personality.name} mounted successfully but no model is selected")

        except Exception as ex:
            ASCIIColors.error(f"Couldn't load personality. Please verify your configuration file at {self.lollms_paths.personal_configuration_path} or use the next menu to select a valid personality")
            ASCIIColors.error(f"Binding returned this exception : {ex}")
            self.trace_exception(ex)
            ASCIIColors.error(f"{self.config.get_personality_path_infos()}")
            self.personality = None
        self.mounted_personalities.append(self.personality)
        return self.personality
    
    def mount_personalities(self):
        self.mounted_personalities = []
        to_remove = []
        for i in range(len(self.config["personalities"])):
            p = self.mount_personality(i)
            if p is None:
                to_remove.append(i)
        to_remove.sort(reverse=True)
        for i in to_remove:
            self.unmount_personality(i)


    def unmount_personality(self, id:int)->bool:
        if id<len(self.config.personalities):
            del self.config.personalities[id]
            del self.mounted_personalities[id]
            if self.config.active_personality_id>=id:
                self.config.active_personality_id-=1

            self.config.save_config()
            return True
        else:
            return False


    def select_personality(self, id:int):
        if id<len(self.config.personalities):
            self.config.active_personality_id = id
            self.personality = self.mount_personalities[id]
            self.config.save_config()
            return True
        else:
            return False


    def load_personality(self):
        try:
            personality = PersonalityBuilder(self.lollms_paths, self.config, self.model).build_personality()
        except Exception as ex:
            ASCIIColors.error(f"Couldn't load personality. Please verify your configuration file at {self.configuration_path} or use the next menu to select a valid personality")
            ASCIIColors.error(f"Binding returned this exception : {ex}")
            ASCIIColors.error(f"{self.config.get_personality_path_infos()}")
            print("Please select a valid model or install a new one from a url")
            personality = None
        return personality

    @staticmethod   
    def reset_paths(lollms_paths:LollmsPaths):
        lollms_paths.resetPaths()

    @staticmethod   
    def reset_all_installs(lollms_paths:LollmsPaths):
        ASCIIColors.info("Removeing all configuration files to force reinstall")
        ASCIIColors.info(f"Searching files from {lollms_paths.personal_configuration_path}")
        for file_path in lollms_paths.personal_configuration_path.iterdir():
            if file_path.name!=f"{lollms_paths.tool_prefix}local_config.yaml" and file_path.suffix.lower()==".yaml":
                file_path.unlink()
                ASCIIColors.info(f"Deleted file: {file_path}")
