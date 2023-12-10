from lollms.main_config import LOLLMSConfig
from ascii_colors import ASCIIColors

from lollms.paths import LollmsPaths
from pathlib import Path
import argparse
from lollms.app import LollmsApplication
from typing import Callable

from lollms.config import BaseConfig
from lollms.binding import BindingBuilder, InstallOption
from tqdm import tqdm


class Settings(LollmsApplication):
    def __init__(
                    self, 
                    lollms_paths:LollmsPaths=None,
                    configuration_path:str|Path=None, 
                    show_logo:bool=True, 
                    show_commands_list:bool=False, 
                    show_personality_infos:bool=True,
                    show_model_infos:bool=True,
                    show_welcome_message:bool=True
                ):

        # get paths
        if lollms_paths is None:
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

    def start(self):
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
    parser = argparse.ArgumentParser(description='The lollms-settings app is used to configure multiple aspects of the lollms project. Lollms is a multi bindings LLM service that serves multiple LLM models that can generate text out of a prompt.')

    # Add the configuration path argument
    parser.add_argument('--configuration_path', default=None,
                        help='Path to the configuration file')

    parser.add_argument('--reset_personal_path', action='store_true', help='Reset the personal path')
    parser.add_argument('--reset_config', action='store_true', help='Reset the configurations')
    parser.add_argument('--reset_installs', action='store_true', help='Reset all installation status')
    parser.add_argument('--default_cfg_path', type=str, default=None, help='Reset all installation status')


    parser.add_argument('--tool_prefix', type=str, default="lollms_server_", help='A prefix to define what tool is being used (default lollms_server_)\nlollms applications prefixes:\n lollms server: lollms_server_\nlollms-elf: lollms_elf_\nlollms-webui: ""\nlollms-discord-bot: lollms_discord_')
    parser.add_argument('--set_personal_folder_path', type=str, default=None, help='Forces installing and selecting a specific binding')
    parser.add_argument('--install_binding', type=str, default=None, help='Forces installing and selecting a specific binding')
    parser.add_argument('--install_model', type=str, default=None, help='Forces installing and selecting a specific model')
    parser.add_argument('--select_model', type=str, default=None, help='Forces selecting a specific model')
    parser.add_argument('--mount_personalities', nargs='+', type=str, default=None, help='Forces mounting a list of personas')
    parser.add_argument('--set_personal_foldrer', type=str, default=None, help='Forces setting personal folder to a specific value')

    parser.add_argument('--silent', action='store_true', help='This will operate in scilent mode, no menu will be shown')

    # Parse the command-line arguments
    args = parser.parse_args()

    tool_prefix = args.tool_prefix

    if args.reset_installs:
        LollmsApplication.reset_all_installs()

    if args.reset_personal_path:
        LollmsPaths.reset_configs()
    
    if args.reset_config:
        lollms_paths = LollmsPaths.find_paths(custom_default_cfg_path=args.default_cfg_path, tool_prefix=tool_prefix, force_personal_path=args.set_personal_folder_path)
        cfg_path = lollms_paths.personal_configuration_path / f"{lollms_paths.tool_prefix}local_config.yaml"
        try:
            cfg_path.unlink()
            ASCIIColors.success("LOLLMS configuration reset successfully")
        except:
            ASCIIColors.success("Couldn't reset LOLLMS configuration")
    else:
        lollms_paths = LollmsPaths.find_paths(force_local=False, tool_prefix=tool_prefix, force_personal_path=args.set_personal_folder_path)

    configuration_path = args.configuration_path
        
    if args.set_personal_folder_path:
        cfg = BaseConfig(config={
            "lollms_path":str(Path(__file__).parent),
            "lollms_personal_path":args.set_personal_folder_path
        })
        ASCIIColors.green(f"Selected personal path: {cfg.lollms_personal_path}")
        pp= Path(cfg.lollms_personal_path)
        if not pp.exists():
            try:
                pp.mkdir(parents=True)
                ASCIIColors.green(f"Personal path set to {pp}")
            except:
                print(f"{ASCIIColors.color_red}It seams there is an error in the path you rovided{ASCIIColors.color_reset}")
        global_paths_cfg_path = Path.home()/f"{tool_prefix}global_paths_cfg.yaml"
        lollms_paths = LollmsPaths.find_paths(force_local=False, custom_global_paths_cfg_path=global_paths_cfg_path, tool_prefix=tool_prefix)
        cfg.save_config(global_paths_cfg_path)


    settings_app = Settings(configuration_path=configuration_path, lollms_paths=lollms_paths, show_commands_list=True)

    if args.install_binding:
        settings_app.config.binding_name= args.install_binding
        settings_app.binding = BindingBuilder().build_binding(settings_app.config, settings_app.lollms_paths,InstallOption.FORCE_INSTALL, app=settings_app)
        settings_app.config.save_config()


    if args.install_model:
        if not settings_app.binding:
            settings_app.binding = BindingBuilder().build_binding(settings_app.config, settings_app.lollms_paths,InstallOption.FORCE_INSTALL, app=settings_app)
        
        def progress_callback(blocks, block_size, total_size):
            tqdm_bar.total=total_size
            tqdm_bar.update(block_size)

        # Usage example
        with tqdm(total=100, unit="%", desc="Download Progress", ncols=80) as tqdm_bar:
            settings_app.config.download_model(args.install_model,settings_app.binding, progress_callback)
        
        settings_app.config.model_name = args.install_model.split("/")[-1]
        settings_app.model = settings_app.binding.build_model()
        settings_app.config.save_config()

    if args.select_model:
        settings_app.config.model_name = args.select_model
        if not settings_app.binding:
            settings_app.binding = BindingBuilder().build_binding(settings_app.config, settings_app.lollms_paths,InstallOption.FORCE_INSTALL, app=settings_app)
        settings_app.model = settings_app.binding.build_model()
        settings_app.config.save_config()

    if args.mount_personalities:
        for entry in args.mount_personalities:
            try:
                settings_app.config.personalities.append(entry)
                settings_app.mount_personality(-1)
                ASCIIColors.green(f"Personality : {entry} mounted")
            except:
                settings_app.config.personalities.pop()
                ASCIIColors.red(f"Personality : {entry} couldn't be mounted")
                
    if not args.silent:
        settings_app.start()

    

if __name__ == "__main__":
    main()