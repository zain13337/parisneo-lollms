"""
project: lollms
file: lollms_configuration_infos.py 
author: ParisNeo
description: 
    This module contains a set of FastAPI routes that provide information about the Lord of Large Language and Multimodal Systems (LoLLMs) Web UI
    application. These routes are specific to configurations

"""
from fastapi import APIRouter, Request
from pydantic import BaseModel
import pkg_resources
from lollms.server.elf_server import LOLLMSElfServer
from lollms.binding import BindingBuilder, InstallOption
from ascii_colors import ASCIIColors
from lollms.utilities import load_config, trace_exception, gc
from pathlib import Path
from typing import List
import json

class SettingsInfos(BaseModel):
    setting_name:str
    setting_value:str

router = APIRouter()
lollmsElfServer = LOLLMSElfServer.get_instance()


# ----------------------------------- Settings -----------------------------------------
@router.post("/update_setting")
def update_setting(data:SettingsInfos):
    setting_name = data.setting_name

    ASCIIColors.info(f"Requested updating of setting {data.setting_name} to {data.setting_value}")
    if setting_name== "temperature":
        lollmsElfServer.config["temperature"]=float(data.setting_value)
    elif setting_name== "n_predict":
        lollmsElfServer.config["n_predict"]=int(data.setting_value)
    elif setting_name== "top_k":
        lollmsElfServer.config["top_k"]=int(data.setting_value)
    elif setting_name== "top_p":
        lollmsElfServer.config["top_p"]=float(data.setting_value)
        
    elif setting_name== "repeat_penalty":
        lollmsElfServer.config["repeat_penalty"]=float(data.setting_value)
    elif setting_name== "repeat_last_n":
        lollmsElfServer.config["repeat_last_n"]=int(data.setting_value)

    elif setting_name== "n_threads":
        lollmsElfServer.config["n_threads"]=int(data.setting_value)
    elif setting_name== "ctx_size":
        lollmsElfServer.config["ctx_size"]=int(data.setting_value)

    elif setting_name== "personality_folder":
        lollmsElfServer.personality_name=data.setting_value
        if len(lollmsElfServer.config["personalities"])>0:
            if lollmsElfServer.config["active_personality_id"]<len(lollmsElfServer.config["personalities"]):
                lollmsElfServer.config["personalities"][lollmsElfServer.config["active_personality_id"]] = f"{lollmsElfServer.personality_category}/{lollmsElfServer.personality_name}"
            else:
                lollmsElfServer.config["active_personality_id"] = 0
                lollmsElfServer.config["personalities"][lollmsElfServer.config["active_personality_id"]] = f"{lollmsElfServer.personality_category}/{lollmsElfServer.personality_name}"
            
            if lollmsElfServer.personality_category!="custom_personalities":
                personality_fn = lollmsElfServer.lollms_paths.personalities_zoo_path/lollmsElfServer.config["personalities"][lollmsElfServer.config["active_personality_id"]]
            else:
                personality_fn = lollmsElfServer.lollms_paths.personal_personalities_path/lollmsElfServer.config["personalities"][lollmsElfServer.config["active_personality_id"]].split("/")[-1]
            lollmsElfServer.personality.load_personality(personality_fn)
        else:
            lollmsElfServer.config["personalities"].append(f"{lollmsElfServer.personality_category}/{lollmsElfServer.personality_name}")
    elif setting_name== "override_personality_model_parameters":
        lollmsElfServer.config["override_personality_model_parameters"]=bool(data.setting_value)

    elif setting_name== "binding_name":
        if lollmsElfServer.config['binding_name']!= data.setting_value:
            print(f"New binding selected : {data.setting_value}")
            lollmsElfServer.config["binding_name"]=data.setting_value
            try:
                if lollmsElfServer.binding:
                    lollmsElfServer.binding.destroy_model()
                lollmsElfServer.binding = None
                lollmsElfServer.model = None
                for per in lollmsElfServer.mounted_personalities:
                    per.model = None
                gc.collect()
                lollmsElfServer.binding = BindingBuilder().build_binding(lollmsElfServer.config, lollmsElfServer.lollms_paths, InstallOption.INSTALL_IF_NECESSARY, lollmsCom=lollmsElfServer)
                lollmsElfServer.model = None
                lollmsElfServer.config.save_config()
                ASCIIColors.green("Binding loaded successfully")
            except Exception as ex:
                ASCIIColors.error(f"Couldn't build binding: [{ex}]")
                trace_exception(ex)
                return {"status":False, 'error':str(ex)}
        else:
            if lollmsElfServer.config["debug"]:
                print(f"Configuration {data.setting_name} set to {data.setting_value}")
            return {'setting_name': data.setting_name, "status":True}


    elif setting_name == "model_name":
        ASCIIColors.yellow(f"Changing model to: {data.setting_value}")
        lollmsElfServer.config["model_name"]=data.setting_value
        lollmsElfServer.config.save_config()
        try:
            lollmsElfServer.model = None
            for per in lollmsElfServer.mounted_personalities:
                if per is not None:
                    per.model = None
            lollmsElfServer.model = lollmsElfServer.binding.build_model()
            if lollmsElfServer.model is not None:
                ASCIIColors.yellow("New model OK")
            for per in lollmsElfServer.mounted_personalities:
                per.model = lollmsElfServer.model
        except Exception as ex:
            trace_exception(ex)
            lollmsElfServer.InfoMessage(f"It looks like you we couldn't load the model.\nHere is the error message:\n{ex}")


    else:
        if data.setting_name in lollmsElfServer.config.config.keys():
            lollmsElfServer.config[data.setting_name] = data.setting_value
        else:
            if lollmsElfServer.config["debug"]:
                print(f"Configuration {data.setting_name} couldn't be set to {data.setting_value}")
            return {'setting_name': data.setting_name, "status":False}

    if lollmsElfServer.config["debug"]:
        print(f"Configuration {data.setting_name} set to {data.setting_value}")
        
    ASCIIColors.success(f"Configuration {data.setting_name} updated")
    if lollmsElfServer.config.auto_save:
        lollmsElfServer.config.save_config()
    # Tell that the setting was changed
    return {'setting_name': data.setting_name, "status":True}
