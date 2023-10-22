from lollms.main_config import LOLLMSConfig
from ascii_colors import ASCIIColors

from lollms.paths import LollmsPaths
from pathlib import Path
import argparse
from lollms.app import LollmsApplication
from typing import Callable



class Settings(LollmsApplication):
    def __init__(
                    self, 
                    configuration_path:str|Path=None, 
                    show_logo:bool=True, 
                    show_commands_list:bool=False, 
                    show_personality_infos:bool=True,
                    show_model_infos:bool=True,
                    show_welcome_message:bool=True
                ):

        # get paths
        lollms_paths = LollmsPaths.find_paths(force_local=False, tool_prefix="lollms_server_")
        # Load maoin configuration
        config = LOLLMSConfig.autoload(lollms_paths)

        super().__init__("lollms-settings", config, lollms_paths, load_model=False)

        if show_logo:
            self.menu.show_logo()
            
        if show_personality_infos:
            try:
                print()
                print(f"{ASCIIColors.color_green}Current personality : {ASCIIColors.color_reset}{self.personality}")
                print(f"{ASCIIColors.color_green}Version : {ASCIIColors.color_reset}{self.personality.version}")
                print(f"{ASCIIColors.color_green}Author : {ASCIIColors.color_reset}{self.personality.author}")
                print(f"{ASCIIColors.color_green}Description : {ASCIIColors.color_reset}{self.personality.personality_description}")
                print()
            except:
                pass
        if show_model_infos:
            try:
                print()
                print(f"{ASCIIColors.color_green}Current binding : {ASCIIColors.color_reset}{self.config['binding_name']}")
                print(f"{ASCIIColors.color_green}Current model : {ASCIIColors.color_reset}{self.config['model_name']}")
                print()
            except:
                pass

        self.menu.main_menu()
            
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
            home_dir = self.lollms_paths.personal_path/"logs"
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
        
    def safe_generate(self, full_discussion:str, n_predict=None, callback: Callable[[str, int, dict], bool]=None):
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

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='App Description')

    # Add the configuration path argument
    parser.add_argument('--configuration_path', default=None,
                        help='Path to the configuration file')

    parser.add_argument('--reset_personal_path', action='store_true', help='Reset the personal path')
    parser.add_argument('--reset_config', action='store_true', help='Reset the configurations')
    parser.add_argument('--reset_installs', action='store_true', help='Reset all installation status')
    parser.add_argument('--default_cfg_path', type=str, default=None, help='Reset all installation status')



    # Parse the command-line arguments
    args = parser.parse_args()

    if args.reset_installs:
        LollmsApplication.reset_all_installs()

    if args.reset_personal_path:
        LollmsPaths.reset_configs()
    
    if args.reset_config:
        lollms_paths = LollmsPaths.find_paths(custom_default_cfg_path=args.default_cfg_path, tool_prefix="lollms_server_")
        cfg_path = lollms_paths.personal_configuration_path / f"{lollms_paths.tool_prefix}local_config.yaml"
        try:
            cfg_path.unlink()
            ASCIIColors.success("LOLLMS configuration reset successfully")
        except:
            ASCIIColors.success("Couldn't reset LOLLMS configuration")

    configuration_path = args.configuration_path
        
    Settings(configuration_path=configuration_path, show_commands_list=True)
    

if __name__ == "__main__":
    main()