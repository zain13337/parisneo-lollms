from lollms.config import InstallOption
from lollms.binding import BindingBuilder, ModelBuilder
from lollms.personality import MSG_TYPE, PersonalityBuilder
from lollms.main_config import LOLLMSConfig
from lollms.helpers import ASCIIColors
from lollms.paths import LollmsPaths
from lollms.app import LollmsApplication
from lollms.terminal import MainMenu

from typing import Callable
from pathlib import Path
import argparse
import yaml
import time
import sys


class Trainer(LollmsApplication):
    def __init__(
                    self, 
                    configuration_path:str|Path=None, 
                    show_logo:bool=True, 
                    show_time_elapsed:bool = False
                ):
        # Fore it to be a path
        self.configuration_path = configuration_path
        self.is_logging = False
        self.log_file_path = ""
        self.show_time_elapsed = show_time_elapsed
        self.bot_says = ""

        # get paths
        lollms_paths = LollmsPaths.find_paths(force_local=False, tool_prefix="lollms_server_")

        # Configuration loading part
        config = LOLLMSConfig.autoload(lollms_paths, configuration_path)

        super().__init__("lollms-console",config, lollms_paths)

        if show_logo:
            self.menu.show_logo()
            
    def start_training(self):
        print(self.model)
    
def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='App Description')

    # Add the configuration path argument
    parser.add_argument('--configuration_path', default=None,
                        help='Path to the configuration file')

    # Parse the command-line arguments
    args = parser.parse_args()


    # Parse the command-line arguments
    args = parser.parse_args()
            
    configuration_path = args.configuration_path
        
    lollms_app = Trainer(configuration_path=configuration_path)
    lollms_app.start_training()
    

if __name__ == "__main__":
    main()
