
from lollms.helpers import ASCIIColors
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lollms.app import LollmsApplication

from lollms.binding import BindingBuilder
from lollms.config import InstallOption
from lollms.personality import PersonalityBuilder

from tqdm import tqdm
import pkg_resources
from pathlib import Path
import yaml
import sys

class MainMenu:
    def __init__(self, lollms_app:'LollmsApplication'):
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
        
        if self.lollms_app.personality:
            if self.lollms_app.personality.help !="":
                print(f"Personality help:")
                print(f"{self.lollms_app.personality.help}")
            
        

    def show_menu(self, options, title="Menu:", selection:int=None):
        ASCIIColors.yellow(title)
        for index, option in enumerate(options):
            if selection is not None and index==selection:
                print(f"{ASCIIColors.color_green}{index + 1} - {option}{ASCIIColors.color_reset}")
            else:
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

    def mount_personality(self):
        print()
        ASCIIColors.red(f"Mounted personalities:")
        for i,personality in enumerate(self.lollms_app.config['personalities']):
            if i==self.lollms_app.config['active_personality_id']:
                ASCIIColors.green(personality)
            else:
                ASCIIColors.yellow(personality)
        personality_languages = [p.stem for p in self.lollms_app.lollms_paths.personalities_zoo_path.iterdir() if p.is_dir() and not p.name.startswith(".")] + ["Back"]
        print("Select language")
        choice = self.show_menu(personality_languages)
        if 1 <= choice <= len(personality_languages)-1:
            language = personality_languages[choice - 1]
            print(f"You selected language: {ASCIIColors.color_green}{language}{ASCIIColors.color_reset}")
            personality_categories = [p.stem for p in (self.lollms_app.lollms_paths.personalities_zoo_path/language).iterdir() if p.is_dir() and not p.name.startswith(".")]+["Back"]
            print("Select category")
            choice = self.show_menu(personality_categories)
            if 1 <= choice <= len(personality_categories)-1:
                category = personality_categories[choice - 1]
                print(f"You selected category: {ASCIIColors.color_green}{category}{ASCIIColors.color_reset}")
                personality_names = [p.stem for p in (self.lollms_app.lollms_paths.personalities_zoo_path/language/category).iterdir() if p.is_dir() and not p.name.startswith(".")]+["Back"]
                print("Select personality")
                choice = self.show_menu(personality_names)
                if 1 <= choice <= len(personality_names)-1:
                    name = personality_names[choice - 1]
                    print(f"You selected personality: {ASCIIColors.color_green}{name}{ASCIIColors.color_reset}")
                    self.lollms_app.config["personalities"].append(f"{language}/{category}/{name}")
                    self.lollms_app.mount_personality(len(self.lollms_app.config["personalities"])-1)
                    self.lollms_app.config.save_config()
                    print("Personality mounted successfully!")
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

    def vew_mounted_personalities(self):
        ASCIIColors.info("Here is the list of mounted personalities")
        entries = self.lollms_app.config['personalities']
        for id, entry in enumerate(entries):
            if id != self.lollms_app.config.active_personality_id:
                ASCIIColors.yellow(f"{id+1} - {entry}")
            else:
                ASCIIColors.green(f"{id+1} - {entry}")
        self.show_menu(["Back"])


    def unmount_personality(self):
        ASCIIColors.info("Select personality to unmount")
        entries = self.lollms_app.config['personalities']+["Back"]
        try:
            choice = int(self.show_menu(entries, self.lollms_app.config['active_personality_id']))-1
            if choice<len(entries)-1:
                self.lollms_app.unmount_personality(choice)
        except Exception as ex:
            ASCIIColors.error(f"Couldn't uhnmount personality.\nGot this exception:{ex}")

    def select_personality(self):
        ASCIIColors.info("Select personality to activate")
        entries = self.lollms_app.config['personalities']+["Back"]
        try:
            choice = int(self.show_menu(entries, self.lollms_app.config['active_personality_id']))-1
            if choice<len(entries)-1:
                self.lollms_app.select_personality(choice)
        except Exception as ex:
            ASCIIColors.error(f"Couldn't set personality.\nGot this exception:{ex}")

    def reinstall_binding(self):
        lollms_app = self.lollms_app
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

            try:
                lollms_app.binding = BindingBuilder().build_binding(lollms_app.config, lollms_app.lollms_paths,InstallOption.FORCE_INSTALL)
            except Exception as ex:
                print(ex)
                print(f"Couldn't find binding. Please verify your configuration file at {lollms_app.config.file_path} or use the next menu to select a valid binding")
                self.select_binding()

            self.lollms_app.config['model_name']=None
            self.lollms_app.config.save_config()
        elif choice <= len(bindings_list):
            return
        else:
            print("Invalid choice!")        


    
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
            print(f"{ASCIIColors.color_green}3 -{ASCIIColors.color_reset} View mounted Personalities")
            print(f"{ASCIIColors.color_green}4 -{ASCIIColors.color_reset} Mount Personality")
            print(f"{ASCIIColors.color_green}5 -{ASCIIColors.color_reset} Unmount Personality")
            print(f"{ASCIIColors.color_green}6 -{ASCIIColors.color_reset} Select Personality")
            print(f"{ASCIIColors.color_green}7 -{ASCIIColors.color_reset} Reinstall Binding")
            print(f"{ASCIIColors.color_green}8 -{ASCIIColors.color_reset} Reinstall current Personality")
            print(f"{ASCIIColors.color_green}9 -{ASCIIColors.color_reset} Reset all installs")        
            print(f"{ASCIIColors.color_green}10 -{ASCIIColors.color_reset} Reset paths")        
            print(f"{ASCIIColors.color_green}11 -{ASCIIColors.color_reset} Back to app")
            print(f"{ASCIIColors.color_green}12 -{ASCIIColors.color_reset} Help")
            print(f"{ASCIIColors.color_green}13 -{ASCIIColors.color_reset} Exit app")
            choice = input("Enter your choice: ").strip()
            if choice == "1":
                self.select_binding()
            elif choice == "2":
                self.select_model()
            elif choice == "3":
                self.vew_mounted_personalities()
            elif choice == "4":
                self.mount_personality()
            elif choice == "5":
                self.unmount_personality()
            elif choice == "6":
                self.select_personality()
            elif choice == "7":
                self.reinstall_binding()
            elif choice == "8":
                self.reinstall_personality()
            elif choice == "9":
                self.lollms_app.reset_all_installs()
            elif choice == "10":
                self.lollms_app.reset_paths()
                
            elif choice == "11":
                print("Back to main app...")
                break
            elif choice == "12":
                self.show_commands_list()
            elif choice == "13":
                print("Bye")
                sys.exit(0)
            else:
                print("Invalid choice! Try again.")

