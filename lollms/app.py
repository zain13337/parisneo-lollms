from lollms.main_config import LOLLMSConfig
from lollms.paths import LollmsPaths
from lollms.personality import PersonalityBuilder
from lollms.binding import LLMBinding, BindingBuilder, ModelBuilder
from lollms.extension import LOLLMSExtension, ExtensionBuilder
from lollms.config import InstallOption
from lollms.helpers import trace_exception
from lollms.terminal import MainMenu
from lollms.utilities import PromptReshaper
from safe_store import TextVectorizer, VectorizationMethod, VisualizationMethod
from typing import Callable
from ascii_colors import ASCIIColors
from pathlib import Path
from datetime import datetime
from functools import partial
import subprocess
import importlib
import sys

class LollmsApplication:
    def __init__(
                    self, 
                    app_name:str, 
                    config:LOLLMSConfig, 
                    lollms_paths:LollmsPaths, 
                    load_binding=True, 
                    load_model=True, 
                    try_select_binding=False, 
                    try_select_model=False,
                    callback=None,
                    notification_callback:Callable=None
                ) -> None:
        """
        Creates a LOLLMS Application
        """
        self.app_name               = app_name
        self.config                 = config
        self.lollms_paths           = lollms_paths

        self.menu                   = MainMenu(self, callback)
        self.mounted_personalities  = []
        self.personality            = None

        self.mounted_extensions     = []
        self.notification_callback  = notification_callback
        self.binding                = None
        self.model:LLMBinding       = None
        self.long_term_memory       = None

        try:
            if config.auto_update:
                # Clone the repository to the target path
                if self.lollms_paths.lollms_core_path.exists():
                    ASCIIColors.info("Lollms_core found in the app space.\nPulling last lollms_core")
                    subprocess.run(["git", "-C", self.lollms_paths.lollms_core_path, "pull"])            
                if self.lollms_paths.safe_store_path.exists():
                    ASCIIColors.info("safe_store_path found in the app space.\nPulling last safe_store_path")
                    subprocess.run(["git", "-C", self.lollms_paths.safe_store_path, "pull"])            
                # Pull the repository if it already exists
                
                ASCIIColors.info("Bindings zoo found in your personal space.\nPulling last personalities zoo")
                subprocess.run(["git", "-C", self.lollms_paths.bindings_zoo_path, "pull"])            
                # Pull the repository if it already exists
                ASCIIColors.info("Personalities zoo found in your personal space.\nPulling last personalities zoo")
                subprocess.run(["git", "-C", self.lollms_paths.personalities_zoo_path, "pull"])            
                # Pull the repository if it already exists
                ASCIIColors.info("Extensions zoo found in your personal space.\nPulling last Extensions zoo")
                subprocess.run(["git", "-C", self.lollms_paths.extensions_zoo_path, "pull"])            
                # Pull the repository if it already exists
                ASCIIColors.info("Models zoo found in your personal space.\nPulling last Models zoo")
                subprocess.run(["git", "-C", self.lollms_paths.models_zoo_path, "pull"])            
        except Exception as ex:
            ASCIIColors.error("Couldn't pull zoos. Please contact the main dev on our discord channel and report the problem.")
            trace_exception(ex)

        if self.config.binding_name is None:
            ASCIIColors.warning(f"No binding selected")
            if try_select_binding:
                ASCIIColors.info("Please select a valid model or install a new one from a url")
                self.menu.select_binding()
        else:
            if load_binding:
                try:
                    ASCIIColors.info(f">Loading binding {self.config.binding_name}. Please wait ...")
                    self.binding            = self.load_binding()
                except Exception as ex:
                    ASCIIColors.error(f"Failed to load binding.\nReturned exception: {ex}")
                    trace_exception(ex)

                if self.binding is not None:
                    ASCIIColors.success(f"Binding {self.config.binding_name} loaded successfully.")
                    if load_model:
                        if self.config.model_name is None:
                            ASCIIColors.warning(f"No model selected")
                            if try_select_model:
                                print("Please select a valid model")
                                self.menu.select_model()
                                
                        if self.config.model_name is not None:
                            try:
                                ASCIIColors.info(f">Loading model {self.config.model_name}. Please wait ...")
                                self.model          = self.load_model()
                                if self.model is not None:
                                    ASCIIColors.success(f"Model {self.config.model_name} loaded successfully.")

                            except Exception as ex:
                                ASCIIColors.error(f"Failed to load model.\nReturned exception: {ex}")
                                trace_exception(ex)
                else:
                    ASCIIColors.warning(f"Couldn't load binding {self.config.binding_name}.")
            
        self.mount_personalities()
        self.mount_extensions()

    def build_long_term_skills_memory(self):
        db_path:Path = self.lollms_paths.personal_databases_path/self.config.db_path.split(".")[0]
        db_path.mkdir(exist_ok=True, parents=True)
        self.long_term_memory = TextVectorizer(
                vectorization_method=VectorizationMethod.TFIDF_VECTORIZER,
                model=self.model,
                database_path=db_path/"skills_memory.json",
                save_db=True,
                data_visualization_method=VisualizationMethod.PCA,
            )
        return self.long_term_memory
    
    def process_chunk(
                        self, 
                        chunk:str, 
                        message_type,
                        parameters:dict=None, 
                        metadata:list=None, 
                        personality=None
                    ):
        
        pass

    def default_callback(self, chunk, type, generation_infos:dict):
        if generation_infos["nb_received_tokens"]==0:
            self.start_time = datetime.now()
        dt =(datetime.now() - self.start_time).seconds
        if dt==0:
            dt=1
        spd = generation_infos["nb_received_tokens"]/dt
        ASCIIColors.green(f"Received {generation_infos['nb_received_tokens']} tokens (speed: {spd:.2f}t/s)              ",end="\r",flush=True) 
        sys.stdout = sys.__stdout__
        sys.stdout.flush()
        if chunk:
            generation_infos["generated_text"] += chunk
        antiprompt = self.personality.detect_antiprompt(generation_infos["generated_text"])
        if antiprompt:
            ASCIIColors.warning(f"\nDetected hallucination with antiprompt: {antiprompt}")
            generation_infos["generated_text"] = self.remove_text_from_string(generation_infos["generated_text"],antiprompt)
            return False
        else:
            generation_infos["nb_received_tokens"] += 1
            generation_infos["first_chunk"]=False
            # if stop generation is detected then stop
            if not self.cancel_gen:
                return True
            else:
                self.cancel_gen = False
                ASCIIColors.warning("Generation canceled")
                return False
                
    def learn_from_discussion(self, title, discussion, nb_gen=None, callback=None):
        if self.config.summerize_discussion:
            prompt = f"!@>discussion:\n--\n!@>title:{title}!@>content:\n{discussion}\n--\n!@>question:What should we learn from this discussion?!@>Summerizer: Here is a summary of the most important informations extracted from the discussion:\n"
            if nb_gen is None:
                nb_gen = self.config.max_summary_size
            generated = ""
            gen_infos={
                "nb_received_tokens":0,
                "generated_text":"",
                "processing":False,
                "first_chunk": True,
            }
            if callback is None:
                callback = partial(self.default_callback, generation_infos=gen_infos)
            self.model.generate(prompt, nb_gen, callback)
            if self.config.debug:
                ASCIIColors.yellow(gen_infos["generated_text"])
            return gen_infos["generated_text"]
        else:
            return  discussion
    
    def remove_text_from_string(self, string, text_to_find):
        """
        Removes everything from the first occurrence of the specified text in the string (case-insensitive).

        Parameters:
        string (str): The original string.
        text_to_find (str): The text to find in the string.

        Returns:
        str: The updated string.
        """
        index = string.lower().find(text_to_find.lower())

        if index != -1:
            string = string[:index]

        return string

    def safe_generate(self, full_discussion:str, n_predict=None, callback: Callable[[str, int, dict], bool]=None, placeholder={}, place_holders_to_sacrifice=[], debug=False):
        """safe_generate

        Args:
            full_discussion (string): A prompt or a long discussion to use for generation
            callback (_type_, optional): A callback to call for each received token. Defaults to None.

        Returns:
            str: Model output
        """
        full_discussion = PromptReshaper(full_discussion).build(placeholder, self.model.tokenize, self.model.detokenize, max_nb_tokens=self.config.ctx_size-n_predict, place_holders_to_sacrifice=place_holders_to_sacrifice )
        if debug:
            ASCIIColors.yellow(full_discussion)
        if n_predict == None:
            n_predict =self.personality.model_n_predicts
        self.bot_says = ""
        if self.personality.processor is not None and self.personality.processor_cfg["custom_workflow"]:
            ASCIIColors.info("processing...")
            generated_text = self.personality.processor.run_workflow(full_discussion.split("!@>")[-1] if "!@>" in full_discussion else full_discussion, previous_discussion_text=self.personality.personality_conditioning+fd, callback=callback)
        else:
            ASCIIColors.info("generating...")
            generated_text = self.personality.model.generate(full_discussion, n_predict=n_predict, callback=callback)
        return generated_text

    def notify(self, content, is_success=True, client_id=None):
        if self.notification_callback:
            return self.notification_callback(content, is_success, client_id)

        if is_success:
            ASCIIColors.yellow(content)
        else:
            ASCIIColors.red(content)

    def load_binding(self):
        try:
            binding = BindingBuilder().build_binding(self.config, self.lollms_paths, notification_callback=self.notify)
            return binding    
        except Exception as ex:
            print(ex)
            print(f"Couldn't find binding. Please verify your configuration file at {self.lollms_paths.personal_configuration_path}/local_configs.yaml or use the next menu to select a valid binding")
            print(f"Trying to reinstall binding")
            try:
                binding = BindingBuilder().build_binding(self.config, self.lollms_paths,installation_option=InstallOption.FORCE_INSTALL)
            except Exception as ex:
                ASCIIColors.error("Couldn't reinstall model")
                trace_exception(ex)
            return None    

    
    def load_model(self):
        try:
            model = ModelBuilder(self.binding).get_model()
            for personality in self.mounted_personalities:
                if personality is not None:
                    personality.model = model
        except Exception as ex:
            ASCIIColors.error(f"Couldn't load model. Please verify your configuration file at {self.lollms_paths.personal_configuration_path} or use the next menu to select a valid model")
            ASCIIColors.error(f"Binding returned this exception : {ex}")
            trace_exception(ex)
            ASCIIColors.error(f"{self.config.get_model_path_infos()}")
            print("Please select a valid model or install a new one from a url")
            model = None

        return model


    def mount_extension(self, id:int, callback=None):
        try:
            extension = ExtensionBuilder().build_extension(self.config["extensions"][id], self.lollms_paths, self)
            self.mounted_extensions.append(extension)
            return extension
        except Exception as ex:
            ASCIIColors.error(f"Couldn't load extension. Please verify your configuration file at {self.lollms_paths.personal_configuration_path} or use the next menu to select a valid personality")
            trace_exception(ex)
        return None


    def mount_personality(self, id:int, callback=None):
        try:
            personality = PersonalityBuilder(self.lollms_paths, self.config, self.model, self, callback=callback).build_personality(id)
            if personality.model is not None:
                self.cond_tk = personality.model.tokenize(personality.personality_conditioning)
                self.n_cond_tk = len(self.cond_tk)
                ASCIIColors.success(f"Personality  {personality.name} mounted successfully")
            else:
                if personality.selected_language is not None:
                    ASCIIColors.success(f"Personality  {personality.name} : {personality.selected_language} mounted successfully but no model is selected")
                else:
                    ASCIIColors.success(f"Personality  {personality.name} mounted successfully but no model is selected")
        except Exception as ex:
            ASCIIColors.error(f"Couldn't load personality. Please verify your configuration file at {self.lollms_paths.personal_configuration_path} or use the next menu to select a valid personality")
            ASCIIColors.error(f"Binding returned this exception : {ex}")
            trace_exception(ex)
            ASCIIColors.error(f"{self.config.get_personality_path_infos()}")
            if id == self.config.active_personality_id:
                self.config.active_personality_id=len(self.config.personalities)-1
            personality = None
        
        self.mounted_personalities.append(personality)
        return personality
    
    def mount_personalities(self, callback = None):
        self.mounted_personalities = []
        to_remove = []
        for i in range(len(self.config["personalities"])):
            p = self.mount_personality(i, callback = None)
            if p is None:
                to_remove.append(i)
        to_remove.sort(reverse=True)
        for i in to_remove:
            self.unmount_personality(i)

        if self.config.active_personality_id>=0 and self.config.active_personality_id<len(self.mounted_personalities):
            self.personality = self.mounted_personalities[self.config.active_personality_id]
        else:
            self.config["personalities"].insert(0, "generic/lollms")
            self.mount_personality(0, callback = None)
            self.config.active_personality_id = 0
            self.personality = self.mounted_personalities[self.config.active_personality_id]

    def mount_extensions(self, callback = None):
        self.mounted_extensions = []
        to_remove = []
        for i in range(len(self.config["extensions"])):
            p = self.mount_extension(i, callback = None)
            if p is None:
                to_remove.append(i)
        to_remove.sort(reverse=True)
        for i in to_remove:
            self.unmount_extension(i)


    def set_personalities_callbacks(self, callback: Callable[[str, int, dict], bool]=None):
        for personality in self.mount_personalities:
            personality.setCallback(callback)

    def unmount_extension(self, id:int)->bool:
        if id<len(self.config.extensions):
            del self.config.extensions[id]
            if id>=0 and id<len(self.mounted_extensions):
                del self.mounted_extensions[id]
            self.config.save_config()
            return True
        else:
            return False

            
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
            self.personality = self.mounted_personalities[id]
            self.config.save_config()
            return True
        else:
            return False


    def load_personality(self, callback=None):
        try:
            personality = PersonalityBuilder(self.lollms_paths, self.config, self.model, self, callback=callback).build_personality()
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
