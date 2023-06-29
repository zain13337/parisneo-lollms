from lollms.config import InstallOption
from lollms.binding import BindingBuilder, ModelBuilder
from lollms.personality import MSG_TYPE, PersonalityBuilder
from lollms.main_config import LOLLMSConfig
from lollms.helpers import ASCIIColors
from lollms.paths import LollmsPaths

from pathlib import Path
from tqdm import tqdm
import pkg_resources
import argparse
import yaml
import sys

class LollmsApplication:
    def __init__(self, config:LOLLMSConfig, lollms_paths:LollmsPaths) -> None:
        self.config         = config
        self.lollms_paths   = lollms_paths
    def load_binding(self):
        pass
    def load_model(self):
        pass
    def load_personality(self):
        pass

def reset_paths(lollms_paths:LollmsPaths):
    lollms_paths.resetPaths()

def reset_all_installs(lollms_paths:LollmsPaths):
    ASCIIColors.info("Removeing all configuration files to force reinstall")
    ASCIIColors.info(f"Searching files from {lollms_paths.personal_configuration_path}")
    for file_path in lollms_paths.personal_configuration_path.iterdir():
        if file_path.name!=f"{lollms_paths.tool_prefix}local_config.yaml" and file_path.suffix.lower()==".yaml":
            file_path.unlink()
            ASCIIColors.info(f"Deleted file: {file_path}")

class MainMenu:
    def __init__(self, lollms_app:LollmsApplication):
        self.binding_infs = []
        self.lollms_app = lollms_app

    def show_logo(self):
        print(f"{ASCIIColors.color_bright_yellow}")
        print("      ___       ___           ___       ___       ___           ___      ")
        print("     /\__\     /\  \         /\__\     /\__\     /\__\         /\  \     ")
        print("    /:/  /    /::\  \       /:/  /    /:/  /    /::|  |       /::\  \    ")
        print("   /:/  /    /:/\:\  \     /:/  /    /:/  /    /:|:|  |      /:/\ \  \   ")
        print("  /:/  /    /:/  \:\  \   /:/  /    /:/  /    /:/|:|__|__   _\:\~\ \  \  ")
        print(" /:/__/    /:/__/ \:\__\ /:/__/    /:/__/    /:/ |::::\__\ /\ \:\ \ \__\ ")
        print(" \:\  \    \:\  \ /:/  / \:\  \    \:\  \    \/__/~~/:/  / \:\ \:\ \/__/ ")
        print("  \:\  \    \:\  /:/  /   \:\  \    \:\  \         /:/  /   \:\ \:\__\   ")
        print("   \:\  \    \:\/:/  /     \:\  \    \:\  \       /:/  /     \:\/:/  /   ")
        print("    \:\__\    \::/  /       \:\__\    \:\__\     /:/  /       \::/  /    ")
        print("     \/__/     \/__/         \/__/     \/__/     \/__/         \/__/     ")




        print(f"{ASCIIColors.color_reset}")
        print(f"{ASCIIColors.color_red}Version: {ASCIIColors.color_green}{pkg_resources.get_distribution('lollms').version}")
        print(f"{ASCIIColors.color_red}By : {ASCIIColors.color_green}ParisNeo")
        print(f"{ASCIIColors.color_reset}")

    def show_commands_list(self):
        print()
        print("Commands:")
        print(f"   {ASCIIColors.color_red}├{ASCIIColors.color_reset} menu: shows main menu")        
        print(f"   {ASCIIColors.color_red}├{ASCIIColors.color_reset} help: shows this info")        
        print(f"   {ASCIIColors.color_red}├{ASCIIColors.color_reset} reset: resets the context")
        print(f"   {ASCIIColors.color_red}├{ASCIIColors.color_reset} <empty prompt>: forces the model to continue generating")
        print(f"   {ASCIIColors.color_red}├{ASCIIColors.color_reset} context_infos: current context size and space left before cropping")
        print(f"   {ASCIIColors.color_red}├{ASCIIColors.color_reset} start_log: starts logging the discussion to a text file")
        print(f"   {ASCIIColors.color_red}├{ASCIIColors.color_reset} stop_log: stops logging the discussion to a text file")
        print(f"   {ASCIIColors.color_red}├{ASCIIColors.color_reset} send_file: uploads a file to the AI")
        print(f"   {ASCIIColors.color_red}└{ASCIIColors.color_reset} exit: exists the console")
        
        if self.lollms_app.personality.help !="":
            print(f"Personality help:")
            print(f"{self.lollms_app.personality.help}")
            
        

    def show_menu(self, options):
        print("Menu:")
        for index, option in enumerate(options):
            print(f"{ASCIIColors.color_green}{index + 1} -{ASCIIColors.color_reset} {option}")
        choice = input("Enter your choice: ")
        return int(choice) if choice.isdigit() else -1

    def select_binding(self):
        bindings_list = []
        print()
        print(f"{ASCIIColors.color_green}Current binding: {ASCIIColors.color_reset}{self.lollms_app.config['binding_name']}")
        for p in self.lollms_app.lollms_paths.bindings_zoo_path.iterdir():
            if p.is_dir() and not p.stem.startswith("."):
                with open(p/"binding_card.yaml", "r") as f:
                    card = yaml.safe_load(f)
                with open(p/"models.yaml", "r") as f:
                    models = yaml.safe_load(f)
                is_installed = (self.lollms_app.lollms_paths.personal_configuration_path/f"binding_{p.name}.yaml").exists()
                entry=f"{ASCIIColors.color_green if is_installed else ''}{'*' if self.lollms_app.config['binding_name']==card['name'] else ''} {card['name']} (by {card['author']})"
                bindings_list.append(entry)
                entry={
                    "name":p.name,
                    "card":card,
                    "models":models,
                    "installed": is_installed
                }
                self.binding_infs.append(entry)
        bindings_list += ["Back"]
        choice = self.show_menu(bindings_list)
        if 1 <= choice <= len(bindings_list)-1:
            print(f"You selected binding: {ASCIIColors.color_green}{self.binding_infs[choice - 1]['name']}{ASCIIColors.color_reset}")
            self.lollms_app.config['binding_name']=self.binding_infs[choice - 1]['name']
            self.lollms_app.load_binding()
            self.lollms_app.config['model_name']=None
            self.lollms_app.config.save_config()
        elif choice <= len(bindings_list):
            return
        else:
            print("Invalid choice!")

    def select_model(self):
        print()
        print(f"{ASCIIColors.color_green}Current binding: {ASCIIColors.color_reset}{self.lollms_app.config['binding_name']}")
        print(f"{ASCIIColors.color_green}Current model: {ASCIIColors.color_reset}{self.lollms_app.config['model_name']}")

        models_dir:Path = (self.lollms_app.lollms_paths.personal_models_path/self.lollms_app.config['binding_name'])
        models_dir.mkdir(parents=True, exist_ok=True)

        models_list = [v for v in self.lollms_app.binding.list_models(self.lollms_app.config)] + ["Install model", "Change binding", "Back"]
        choice = self.show_menu(models_list)
        if 1 <= choice <= len(models_list)-3:
            print(f"You selected model: {ASCIIColors.color_green}{models_list[choice - 1]}{ASCIIColors.color_reset}")
            self.lollms_app.config['model_name']=models_list[choice - 1]
            self.lollms_app.config.save_config()
            self.lollms_app.load_model()
        elif choice <= len(models_list)-2:
            self.install_model()
        elif choice <= len(models_list)-1:
            self.select_binding()
            self.select_model()
        elif choice <= len(models_list):
            return
        else:
            print("Invalid choice!")

    def install_model(self):

        models_list = ["Install model from internet","Install model from local file","Back"]
        choice = self.show_menu(models_list)
        if 1 <= choice <= len(models_list)-2:
            url = input("Give a URL to the model to be downloaded :")
            def progress_callback(blocks, block_size, total_size):
                tqdm_bar.total=total_size
                tqdm_bar.update(block_size)

            # Usage example
            with tqdm(total=100, unit="%", desc="Download Progress", ncols=80) as tqdm_bar:
                self.lollms_app.config.download_model(url,self.lollms_app.binding, progress_callback)
            self.select_model()
        elif choice <= len(models_list)-1:
            path = Path(input("Give a path to the model to be used on your PC:"))
            if path.exists():
                self.lollms_app.config.reference_model(path)
            self.select_model()
        elif choice <= len(models_list):
            return
        else:
            print("Invalid choice!")

    def select_personality(self):
        print()
        print(f"{ASCIIColors.color_green}Current personality: {ASCIIColors.color_reset}{self.lollms_app.config['personalities'][self.lollms_app.config['active_personality_id']]}")
        personality_languages = [p.stem for p in self.lollms_app.lollms_paths.personalities_zoo_path.iterdir() if p.is_dir()] + ["Back"]
        print("Select language")
        choice = self.show_menu(personality_languages)
        if 1 <= choice <= len(personality_languages)-1:
            language = personality_languages[choice - 1]
            print(f"You selected language: {ASCIIColors.color_green}{language}{ASCIIColors.color_reset}")
            personality_categories = [p.stem for p in (self.lollms_app.lollms_paths.personalities_zoo_path/language).iterdir() if p.is_dir()]+["Back"]
            print("Select category")
            choice = self.show_menu(personality_categories)
            if 1 <= choice <= len(personality_categories)-1:
                category = personality_categories[choice - 1]
                print(f"You selected category: {ASCIIColors.color_green}{category}{ASCIIColors.color_reset}")
                personality_names = [p.stem for p in (self.lollms_app.lollms_paths.personalities_zoo_path/language/category).iterdir() if p.is_dir()]+["Back"]
                print("Select personality")
                choice = self.show_menu(personality_names)
                if 1 <= choice <= len(personality_names)-1:
                    name = personality_names[choice - 1]
                    print(f"You selected personality: {ASCIIColors.color_green}{name}{ASCIIColors.color_reset}")
                    self.lollms_app.config["personalities"]=[f"{language}/{category}/{name}"]
                    self.lollms_app.load_personality()
                    self.lollms_app.config.save_config()
                    print("Personality saved successfully!")
                elif 1 <= choice <= len(personality_names):
                    return
                else:
                    print("Invalid choice!")
            elif 1 <= choice <= len(personality_categories):
                return
            else:
                print("Invalid choice!")
        elif 1 <= choice <= len(personality_languages):
            return
        else:
            print("Invalid choice!")

    def reinstall_binding(self):
        lollms_app = self.lollms_app
        try:
            lollms_app.binding = BindingBuilder().build_binding(lollms_app.config, lollms_app.lollms_paths,InstallOption.FORCE_INSTALL)
        except Exception as ex:
            print(ex)
            print(f"Couldn't find binding. Please verify your configuration file at {lollms_app.config.file_path} or use the next menu to select a valid binding")
            self.select_binding()
    
    def reinstall_personality(self):
        lollms_app = self.lollms_app
        try:
            lollms_app.personality = PersonalityBuilder(lollms_app.lollms_paths, lollms_app.config, lollms_app.model, installation_option=InstallOption.FORCE_INSTALL).build_personality()
        except Exception as ex:
            ASCIIColors.error(f"Couldn't load personality. Please verify your configuration file at {lollms_app.configuration_path} or use the next menu to select a valid personality")
            ASCIIColors.error(f"Binding returned this exception : {ex}")
            ASCIIColors.error(f"{lollms_app.config.get_personality_path_infos()}")
            print("Please select a valid model or install a new one from a url")
            self.select_model()

    def main_menu(self):
        while True:
            print("\nMain Menu:")
            print(f"{ASCIIColors.color_green}1 -{ASCIIColors.color_reset} Select Binding")
            print(f"{ASCIIColors.color_green}2 -{ASCIIColors.color_reset} Select Model")
            print(f"{ASCIIColors.color_green}3 -{ASCIIColors.color_reset} Select Personality")
            print(f"{ASCIIColors.color_green}4 -{ASCIIColors.color_reset} Reinstall current Binding")
            print(f"{ASCIIColors.color_green}5 -{ASCIIColors.color_reset} Reinstall current Personality")
            print(f"{ASCIIColors.color_green}6 -{ASCIIColors.color_reset} Reset all installs")        
            print(f"{ASCIIColors.color_green}7 -{ASCIIColors.color_reset} Reset paths")        
            print(f"{ASCIIColors.color_green}0 -{ASCIIColors.color_reset} Back to app")
            print(f"{ASCIIColors.color_green}-1 -{ASCIIColors.color_reset} Help")
            print(f"{ASCIIColors.color_green}-2 -{ASCIIColors.color_reset} Exit app")
            choice = input("Enter your choice: ").strip()
            if choice == "1":
                self.select_binding()
            elif choice == "2":
                self.select_model()
            elif choice == "3":
                self.select_personality()
            elif choice == "4":
                self.reinstall_binding()
            elif choice == "5":
                self.reinstall_personality()
            elif choice == "6":
                reset_all_installs()
            elif choice == "6":
                reset_paths()
                
            elif choice == "0":
                print("Back to main app...")
                break
            elif choice == "-1":
                self.show_commands_list()
            elif choice == "-2":
                print("Bye")
                sys.exit(0)
            else:
                print("Invalid choice! Try again.")

class Conversation(LollmsApplication):
    def __init__(
                    self, 
                    configuration_path:str|Path=None, 
                    show_logo:bool=True, 
                    show_commands_list:bool=False, 
                    show_personality_infos:bool=True,
                    show_model_infos:bool=True,
                    show_welcome_message:bool=True
                ):
        
        # Fore it to be a path
        self.configuration_path = configuration_path
        self.is_logging = False
        self.log_file_path = ""
        
        self.bot_says = ""
        # get paths
        lollms_paths = LollmsPaths.find_paths(force_local=False, tool_prefix="lollms_server_")

        # Configuration loading part
        config = LOLLMSConfig.autoload(lollms_paths, configuration_path)

        super().__init__(config, lollms_paths=lollms_paths)

        # Build menu
        self.menu = MainMenu(self)


        # load binding
        self.load_binding()
        
        if self.config.model_name is None:
            self.menu.select_model()

        # Load model
        self.load_model()
        # cfg.binding_name = llm_binding.binding_folder_name
        # cfg.model_name = model_name

        # Load personality
        try:
            self.load_personality()
        except Exception as ex:
            print(f"No personality selected. Please select one from the zoo. {ex}")
            self.menu.select_personality()

        if show_logo:
            self.menu.show_logo()
        if show_commands_list:
            self.menu.show_commands_list()
            
        if show_personality_infos:
            print()
            print(f"{ASCIIColors.color_green}Current personality : {ASCIIColors.color_reset}{self.personality}")
            print(f"{ASCIIColors.color_green}Version : {ASCIIColors.color_reset}{self.personality.version}")
            print(f"{ASCIIColors.color_green}Author : {ASCIIColors.color_reset}{self.personality.author}")
            print(f"{ASCIIColors.color_green}Description : {ASCIIColors.color_reset}{self.personality.personality_description}")
            print()

        if show_model_infos:
            print()
            print(f"{ASCIIColors.color_green}Current binding : {ASCIIColors.color_reset}{self.config['binding_name']}")
            print(f"{ASCIIColors.color_green}Current model : {ASCIIColors.color_reset}{self.config['model_name']}")
            print(f"{ASCIIColors.color_green}Personal data path : {ASCIIColors.color_reset}{self.lollms_paths.personal_path}")
            
            print()


        # If there is a disclaimer, show it
        if self.personality.disclaimer != "":
            print(f"\n{ASCIIColors.color_red}Disclaimer")
            print(self.personality.disclaimer)
            print(f"{ASCIIColors.color_reset}")

        if show_welcome_message and self.personality.welcome_message:
            ASCIIColors.red(self.personality.name+": ", end="")
            print(self.personality.welcome_message)
            
    def ask_override_file(self):
        user_input = input("Would you like to override the existing file? (Y/N): ")
        user_input = user_input.lower()
        if user_input == "y" or user_input == "yes":
            print("File will be overridden.")
            return True
        elif user_input == "n" or user_input == "no":
            print("File will not be overridden.")
            return False
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")
            # Call the function again recursively to prompt the user for valid input
            return self.ask_override_file() 
                   
    def start_log(self, file_name):
        if Path(file_name).is_absolute():
            self.log_file_path = Path(file_name)
        else:
            home_dir = self.lollms_paths.personal_log_path
            home_dir.mkdir(parents=True, exist_ok=True)
            self.log_file_path = home_dir/file_name
            if self.log_file_path.exists():
                if not self.ask_override_file():
                    print("Canceled")
                    return
        try:
            with(open(self.log_file_path, "w") as f):
                self.header = f"""------------------------
Log file for lollms discussion                        
Participating personalities:
{self.config['personalities']}                  
------------------------
"""
                f.write(self.header)
            self.is_logging = True
            return True
        except:
            return False            

    def log(self, text, append=False):
        try:
            with(open(self.log_file_path, "a" if append else "w") as f):
                f.write(text) if append else f.write(self.header+self.personality.personality_conditioning+text)
            return True
        except:
            return False            

    def stop_log(self):
        self.is_logging = False           



    def load_binding(self):
        if self.config.binding_name is None:
            print(f"No bounding selected")
            print("Please select a valid model or install a new one from a url")
            self.menu.select_binding()
            # cfg.download_model(url)
        else:
            try:
                self.binding = BindingBuilder().build_binding(self.config, self.lollms_paths)
            except Exception as ex:
                print(ex)
                print(f"Couldn't find binding. Please verify your configuration file at {self.configuration_path} or use the next menu to select a valid binding")
                print(f"Trying to reinstall binding")
                self.binding = BindingBuilder().build_binding(self.config, self.lollms_paths,installation_option=InstallOption.FORCE_INSTALL)
                self.menu.select_binding()

    def load_model(self):
        try:
            self.model = ModelBuilder(self.binding).get_model()
        except Exception as ex:
            ASCIIColors.error(f"Couldn't load model. Please verify your configuration file at {self.configuration_path} or use the next menu to select a valid model")
            ASCIIColors.error(f"Binding returned this exception : {ex}")
            ASCIIColors.error(f"{self.config.get_model_path_infos()}")
            print("Please select a valid model or install a new one from a url")
            self.menu.select_model()


    def load_personality(self):
        try:
            self.personality = PersonalityBuilder(self.lollms_paths, self.config, self.model).build_personality()
        except Exception as ex:
            ASCIIColors.error(f"Couldn't load personality. Please verify your configuration file at {self.configuration_path} or use the next menu to select a valid personality")
            ASCIIColors.error(f"Binding returned this exception : {ex}")
            ASCIIColors.error(f"{self.config.get_personality_path_infos()}")
            print("Please select a valid model or install a new one from a url")
            self.menu.select_model()
        self.cond_tk = self.personality.model.tokenize(self.personality.personality_conditioning)
        self.n_cond_tk = len(self.cond_tk)

    def reset_context(self):
        if self.personality.include_welcome_message_in_disucssion:
            full_discussion = (
                self.personality.ai_message_prefix +
                self.personality.welcome_message +
                self.personality.link_text
            )
        else:
            full_discussion = ""
        return full_discussion
        
    def safe_generate(self, full_discussion:str, n_predict=None, callback=None):
        """safe_generate

        Args:
            full_discussion (string): A prompt or a long discussion to use for generation
            callback (_type_, optional): A callback to call for each received token. Defaults to None.

        Returns:
            str: Model output
        """
        if n_predict == None:
            n_predict =self.personality.model_n_predicts
        tk = self.personality.model.tokenize(full_discussion)
        n_tokens = len(tk)
        fd = self.personality.model.detokenize(tk[-min(self.config.ctx_size-self.n_cond_tk,n_tokens):])
        self.bot_says = ""
        output = self.personality.model.generate(self.personality.personality_conditioning+fd, n_predict=n_predict, callback=callback)
        return output
    
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
        
    def start_conversation(self):
        full_discussion = self.reset_context()
        while True:
            try:
                prompt = input(f"{ASCIIColors.color_green}You: {ASCIIColors.color_reset}")
                if prompt == "exit":
                    return
                if prompt == "menu":
                    self.menu.main_menu()
                    continue
                if prompt == "reset":
                    self.reset_context()
                    print(f"{ASCIIColors.color_red}Context reset issued{ASCIIColors.color_reset}")
                    continue
                if prompt == "start_log":
                    fp = input("Please enter a log file path (ex: log.txt): ")               
                    self.start_log(fp)
                    print(f"{ASCIIColors.color_red}Started logging to : {self.log_file_path}{ASCIIColors.color_reset}")
                    continue
                if prompt == "stop_log":
                    self.stop_log()
                    print(f"{ASCIIColors.color_red}Log stopped{ASCIIColors.color_reset}")
                    continue    
                if prompt == "send_file":
                    if self.personality.processor is None:
                        print(f"{ASCIIColors.color_red}This personality doesn't support file reception{ASCIIColors.color_reset}")
                        continue
                    fp = input("Please enter a file path: ")
                    # Remove double quotes using string slicing
                    if fp.startswith('"') and fp.endswith('"'):
                        fp = fp[1:-1]
                    if self.personality.processor.add_file(fp):
                        print(f"{ASCIIColors.color_green}File imported{ASCIIColors.color_reset}")
                    else:
                        print(f"{ASCIIColors.color_red}Couldn't load file{ASCIIColors.color_reset}")
                    continue    
                            
                   
                if prompt == "context_infos":
                    tokens = self.personality.model.tokenize(full_discussion)
                    print(f"{ASCIIColors.color_green}Current context has {len(tokens)} tokens/ {self.config.ctx_size}{ASCIIColors.color_reset}")
                    continue                
                              
                if prompt != '':
                    if self.personality.processor is not None and self.personality.processor_cfg["process_model_input"]:
                        preprocessed_prompt = self.personality.processor.process_model_input(prompt)
                    else:
                        preprocessed_prompt = prompt
                    
                    if self.personality.processor is not None and self.personality.processor_cfg["custom_workflow"]:
                        full_discussion += (
                            self.personality.user_message_prefix +
                            preprocessed_prompt
                        )
                    else:
                        full_discussion += (
                            self.personality.user_message_prefix +
                            preprocessed_prompt +
                            self.personality.link_text +
                            self.personality.ai_message_prefix
                        )

                def callback(text, type:MSG_TYPE=None):
                    if type == MSG_TYPE.MSG_TYPE_CHUNK:
                        # Replace stdout with the default stdout
                        sys.stdout = sys.__stdout__
                        print(text, end="", flush=True)
                        bot_says = self.bot_says + text
                        antiprompt = self.personality.detect_antiprompt(bot_says)
                        if antiprompt:
                            self.bot_says = self.remove_text_from_string(bot_says,antiprompt)
                            print("Detected hallucination")
                            return False
                        else:
                            self.bot_says = bot_says
                            return True

                tk = self.personality.model.tokenize(full_discussion)
                n_tokens = len(tk)
                fd = self.personality.model.detokenize(tk[-min(self.config.ctx_size-self.n_cond_tk,n_tokens):])
                
                print(f"{ASCIIColors.color_red}{self.personality.name}:{ASCIIColors.color_reset}", end='', flush=True)
                if self.personality.processor is not None and self.personality.processor_cfg["custom_workflow"]:
                    output = self.personality.processor.run_workflow(prompt, previous_discussion_text=self.personality.personality_conditioning+fd, callback=callback)
                    print(output)

                else:
                    output = self.personality.model.generate(self.personality.personality_conditioning+fd, n_predict=self.personality.model_n_predicts, callback=callback)
                full_discussion += output.strip()
                print()

                if self.personality.processor is not None and self.personality.processor_cfg["process_model_output"]: 
                    output = self.personality.processor.process_model_output(output)

                self.log(full_discussion)
                
            except KeyboardInterrupt:
                print("Keyboard interrupt detected.\nBye")
                break

        print("Done")
        print(f"{self.personality}")

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='App Description')

    # Add the configuration path argument
    parser.add_argument('--configuration_path', default=None,
                        help='Path to the configuration file')


    parser.add_argument('--reset_personal_path', action='store_true', help='Reset the personal path')
    parser.add_argument('--reset_config', action='store_true', help='Reset the configurations')
    parser.add_argument('--reset_installs', action='store_true', help='Reset all installation status')

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.reset_installs:
        reset_all_installs()

    if args.reset_personal_path:
        LollmsPaths.reset_configs()
    
    if args.reset_config:
        lollms_paths = LollmsPaths.find_paths(tool_prefix="lollms_server_")
        cfg_path = lollms_paths.personal_configuration_path / f"{lollms_paths.tool_prefix}local_config.yaml"
        try:
            cfg_path.unlink()
            ASCIIColors.success("LOLLMS configuration reset successfully")
        except:
            ASCIIColors.success("Couldn't reset LOLLMS configuration")


    # Parse the command-line arguments
    args = parser.parse_args()
            
    configuration_path = args.configuration_path
        
    lollms_app = Conversation(configuration_path=configuration_path, show_commands_list=True)
    lollms_app.start_conversation()
    

if __name__ == "__main__":
    main()