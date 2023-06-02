from pathlib import Path
import yaml



class ASCIIColors:
    # Reset
    color_reset = '\u001b[0m'

    # Regular colors
    color_black = '\u001b[30m'
    color_red = '\u001b[31m'
    color_green = '\u001b[32m'
    color_yellow = '\u001b[33m'
    color_blue = '\u001b[34m'
    color_magenta = '\u001b[35m'
    color_cyan = '\u001b[36m'
    color_white = '\u001b[37m'
    color_orange = '\u001b[38;5;202m'

    # Bright colors
    color_bright_black = '\u001b[30;1m'
    color_bright_red = '\u001b[31;1m'
    color_bright_green = '\u001b[32;1m'
    color_bright_yellow = '\u001b[33;1m'
    color_bright_blue = '\u001b[34;1m'
    color_bright_magenta = '\u001b[35;1m'
    color_bright_cyan = '\u001b[36;1m'
    color_bright_white = '\u001b[37;1m'
    color_bright_orange = '\u001b[38;5;208m'

    @staticmethod
    def print(text, color=color_bright_red):
        print(f"{color}{text}{ASCIIColors.color_reset}")

    @staticmethod
    def warning(text):
        print(f"{ASCIIColors.color_bright_orange}{text}{ASCIIColors.color_reset}")

    @staticmethod
    def error(text):
        print(f"{ASCIIColors.color_bright_red}{text}{ASCIIColors.color_reset}")

    @staticmethod
    def success(text):
        print(f"{ASCIIColors.color_green}{text}{ASCIIColors.color_reset}")

    @staticmethod
    def info(text):
        print(f"{ASCIIColors.color_blue}{text}{ASCIIColors.color_reset}")

class BaseConfig():
    def __init__(self, exceptional_keys=[], config = None):
        self.exceptional_keys = exceptional_keys 
        self.config = config


    def to_dict(self):
        return self.config
    
    def __getitem__(self, key):
        if self.config is None:
            raise ValueError("No configuration loaded.")
        return self.config[key]

    def __getattr__(self, key):
        if key == "exceptional_keys":
            return super().__getattribute__(key)
        if key in self.exceptional_keys+ ["config"] or key.startswith("__"):
            return super().__getattribute__(key)
        else:
            if self.config is None:
                raise ValueError("No configuration loaded.")
            return self.config[key]


    def __setattr__(self, key, value):
        if key == "exceptional_keys":
            return super().__setattr__(key, value)
        if key in self.exceptional_keys+ ["config"] or key.startswith("__"):
            super().__setattr__(key, value)
        else:
            if self.config is None:
                raise ValueError("No configuration loaded.")
            self.config[key] = value

    def __setitem__(self, key, value):
        if self.config is None:
            raise ValueError("No configuration loaded.")
        self.config[key] = value
    
    def __contains__(self, item):
        if self.config is None:
            raise ValueError("No configuration loaded.")
        return item in self.config

    def load_config(self, file_path:Path=None):
        if file_path is None:
            file_path = self.file_path
        with open(file_path, 'r', encoding='utf-8') as stream:
            self.config = yaml.safe_load(stream)

    def save_config(self, file_path:Path=None):
        if file_path is None:
            file_path = self.file_path
        if self.config is None:
            raise ValueError("No configuration loaded.")
        with open(file_path, "w") as f:
            yaml.dump(self.config, f)
