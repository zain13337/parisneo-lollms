"""
project: lollms
file: lollms_binding_files_server.py 
author: ParisNeo
description: 
    This module contains a set of FastAPI routes that provide information about the Lord of Large Language and Multimodal Systems (LoLLMs) Web UI
    application. These routes are specific to serving files

"""
from fastapi import APIRouter, Request
from fastapi import HTTPException
from pydantic import BaseModel
import pkg_resources
from lollms.server.elf_server import LOLLMSElfServer
from fastapi.responses import FileResponse
from lollms.binding import BindingBuilder, InstallOption
from ascii_colors import ASCIIColors
from lollms.personality import MSG_TYPE, AIPersonality
from lollms.utilities import load_config, trace_exception, gc, terminate_thread, run_async
from pathlib import Path
from typing import List
import socketio
from datetime import datetime
from functools import partial
import shutil
import threading
import os

lollmsElfServer = LOLLMSElfServer.get_instance()


# ----------------------------------- events -----------------------------------------
def add_events(sio:socketio):
    @sio.on('install_model')
    def install_model(sid, data):
        room_id = sid            
        def install_model_():
            print("Install model triggered")
            model_path = data["path"].replace("\\","/")

            if data["type"].lower() in model_path.lower():
                model_type:str=data["type"]
            else:
                mtt = None
                for mt in lollmsElfServer.binding.models_dir_names:
                    if mt.lower() in  model_path.lower():
                        mtt = mt
                        break
                if mtt:
                    model_type = mtt
                else:
                    model_type:str=lollmsElfServer.binding.models_dir_names[0]

            progress = 0
            installation_dir = lollmsElfServer.binding.searchModelParentFolder(model_path.split('/')[-1], model_type)
            if model_type=="gptq" or  model_type=="awq" or model_type=="transformers":
                parts = model_path.split("/")
                if len(parts)==2:
                    filename = parts[1]
                else:
                    filename = parts[4]
                installation_path = installation_dir / filename
            elif model_type=="gpt4all":
                filename = data["variant_name"]
                model_path = "http://gpt4all.io/models/gguf/"+filename
                installation_path = installation_dir / filename
            else:
                filename = Path(model_path).name
                installation_path = installation_dir / filename
            print("Model install requested")
            print(f"Model path : {model_path}")

            model_name = filename
            binding_folder = lollmsElfServer.config["binding_name"]
            model_url = model_path
            signature = f"{model_name}_{binding_folder}_{model_url}"
            try:
                lollmsElfServer.download_infos[signature]={
                    "start_time":datetime.now(),
                    "total_size":lollmsElfServer.binding.get_file_size(model_path),
                    "downloaded_size":0,
                    "progress":0,
                    "speed":0,
                    "cancel":False
                }
                
                if installation_path.exists():
                    print("Error: Model already exists. please remove it first")
                    run_async( partial(sio.emit,'install_progress',{
                                                        'status': False,
                                                        'error': f'model already exists. Please remove it first.\nThe model can be found here:{installation_path}',
                                                        'model_name' : model_name,
                                                        'binding_folder' : binding_folder,
                                                        'model_url' : model_url,
                                                        'start_time': lollmsElfServer.download_infos[signature]['start_time'].strftime("%Y-%m-%d %H:%M:%S"),
                                                        'total_size': lollmsElfServer.download_infos[signature]['total_size'],
                                                        'downloaded_size': lollmsElfServer.download_infos[signature]['downloaded_size'],
                                                        'progress': lollmsElfServer.download_infos[signature]['progress'],
                                                        'speed': lollmsElfServer.download_infos[signature]['speed'],
                                                    }, room=room_id
                                )
                    )
                    return
                
                run_async( partial(sio.emit,'install_progress',{
                                                'status': True,
                                                'progress': progress,
                                                'model_name' : model_name,
                                                'binding_folder' : binding_folder,
                                                'model_url' : model_url,
                                                'start_time': lollmsElfServer.download_infos[signature]['start_time'].strftime("%Y-%m-%d %H:%M:%S"),
                                                'total_size': lollmsElfServer.download_infos[signature]['total_size'],
                                                'downloaded_size': lollmsElfServer.download_infos[signature]['downloaded_size'],
                                                'progress': lollmsElfServer.download_infos[signature]['progress'],
                                                'speed': lollmsElfServer.download_infos[signature]['speed'],

                                                }, room=room_id)
                )
                
                def callback(downloaded_size, total_size):
                    progress = (downloaded_size / total_size) * 100
                    now = datetime.now()
                    dt = (now - lollmsElfServer.download_infos[signature]['start_time']).total_seconds()
                    speed = downloaded_size/dt
                    lollmsElfServer.download_infos[signature]['downloaded_size'] = downloaded_size
                    lollmsElfServer.download_infos[signature]['speed'] = speed

                    if progress - lollmsElfServer.download_infos[signature]['progress']>2:
                        lollmsElfServer.download_infos[signature]['progress'] = progress
                        run_async( partial(sio.emit,'install_progress',{
                                                        'status': True,
                                                        'model_name' : model_name,
                                                        'binding_folder' : binding_folder,
                                                        'model_url' : model_url,
                                                        'start_time': lollmsElfServer.download_infos[signature]['start_time'].strftime("%Y-%m-%d %H:%M:%S"),
                                                        'total_size': lollmsElfServer.download_infos[signature]['total_size'],
                                                        'downloaded_size': lollmsElfServer.download_infos[signature]['downloaded_size'],
                                                        'progress': lollmsElfServer.download_infos[signature]['progress'],
                                                        'speed': lollmsElfServer.download_infos[signature]['speed'],
                                                        }, room=room_id)
                        )
                    
                    if lollmsElfServer.download_infos[signature]["cancel"]:
                        raise Exception("canceled")
                        
                    
                if hasattr(lollmsElfServer.binding, "download_model"):
                    try:
                        lollmsElfServer.binding.download_model(model_path, installation_path, callback)
                    except Exception as ex:
                        ASCIIColors.warning(str(ex))
                        trace_exception(ex)
                        run_async( partial(sio.emit,'install_progress',{
                                    'status': False,
                                    'error': 'canceled',
                                    'model_name' : model_name,
                                    'binding_folder' : binding_folder,
                                    'model_url' : model_url,
                                    'start_time': lollmsElfServer.download_infos[signature]['start_time'].strftime("%Y-%m-%d %H:%M:%S"),
                                    'total_size': lollmsElfServer.download_infos[signature]['total_size'],
                                    'downloaded_size': lollmsElfServer.download_infos[signature]['downloaded_size'],
                                    'progress': lollmsElfServer.download_infos[signature]['progress'],
                                    'speed': lollmsElfServer.download_infos[signature]['speed'],
                                }, room=room_id
                            )
                        )
                        del lollmsElfServer.download_infos[signature]
                        try:
                            if installation_path.is_dir():
                                shutil.rmtree(installation_path)
                            else:
                                installation_path.unlink()
                        except Exception as ex:
                            trace_exception(ex)
                            ASCIIColors.error(f"Couldn't delete file. Please try to remove it manually.\n{installation_path}")
                        return

                else:
                    try:
                        lollmsElfServer.download_file(model_path, installation_path, callback)
                    except Exception as ex:
                        ASCIIColors.warning(str(ex))
                        trace_exception(ex)
                        run_async( partial(sio.emit,'install_progress',{
                                    'status': False,
                                    'error': 'canceled',
                                    'model_name' : model_name,
                                    'binding_folder' : binding_folder,
                                    'model_url' : model_url,
                                    'start_time': lollmsElfServer.download_infos[signature]['start_time'].strftime("%Y-%m-%d %H:%M:%S"),
                                    'total_size': lollmsElfServer.download_infos[signature]['total_size'],
                                    'downloaded_size': lollmsElfServer.download_infos[signature]['downloaded_size'],
                                    'progress': lollmsElfServer.download_infos[signature]['progress'],
                                    'speed': lollmsElfServer.download_infos[signature]['speed'],
                                }, room=room_id
                           )
                        )
                        del lollmsElfServer.download_infos[signature]
                        installation_path.unlink()
                        return                        
                run_async( partial(sio.emit,'install_progress',{
                                                'status': True, 
                                                'error': '',
                                                'model_name' : model_name,
                                                'binding_folder' : binding_folder,
                                                'model_url' : model_url,
                                                'start_time': lollmsElfServer.download_infos[signature]['start_time'].strftime("%Y-%m-%d %H:%M:%S"),
                                                'total_size': lollmsElfServer.download_infos[signature]['total_size'],
                                                'downloaded_size': lollmsElfServer.download_infos[signature]['downloaded_size'],
                                                'progress': 100,
                                                'speed': lollmsElfServer.download_infos[signature]['speed'],
                                                }, room=room_id)
                )
                del lollmsElfServer.download_infos[signature]
            except Exception as ex:
                trace_exception(ex)
                run_async( partial(sio.emit,'install_progress',{
                            'status': False,
                            'error': str(ex),
                            'model_name' : model_name,
                            'binding_folder' : binding_folder,
                            'model_url' : model_url,
                            'start_time': '',
                            'total_size': 0,
                            'downloaded_size': 0,
                            'progress': 0,
                            'speed': 0,
                        }, room=room_id
                    )                    
                )
        tpe = threading.Thread(target=install_model_, args=())
        tpe.start()

    @sio.on('uninstall_model')
    def uninstall_model(sid, data):
        model_path = data['path']
        model_type:str=data.get("type","ggml")
        installation_dir = lollmsElfServer.binding.searchModelParentFolder(model_path)
        
        binding_folder = lollmsElfServer.config["binding_name"]
        if model_type=="gptq" or  model_type=="awq":
            filename = model_path.split("/")[4]
            installation_path = installation_dir / filename
        else:
            filename = Path(model_path).name
            installation_path = installation_dir / filename
        model_name = filename

        if not installation_path.exists():
            run_async( partial(sio.emit,'uninstall_progress',{
                                                'status': False,
                                                'error': 'The model does not exist',
                                                'model_name' : model_name,
                                                'binding_folder' : binding_folder
                                            }, room=sid)
            )
        try:
            if not installation_path.exists():
                # Try to find a version
                model_path = installation_path.name.lower().replace("-ggml","").replace("-gguf","")
                candidates = [m for m in installation_dir.iterdir() if model_path in m.name]
                if len(candidates)>0:
                    model_path = candidates[0]
                    installation_path = model_path
                    
            if installation_path.is_dir():
                shutil.rmtree(installation_path)
            else:
                installation_path.unlink()
            run_async( partial(sio.emit,'uninstall_progress',{
                                                'status': True, 
                                                'error': '',
                                                'model_name' : model_name,
                                                'binding_folder' : binding_folder
                                            }, room=sid)
            )
        except Exception as ex:
            trace_exception(ex)
            ASCIIColors.error(f"Couldn't delete {installation_path}, please delete it manually and restart the app")
            run_async( partial(sio.emit,'uninstall_progress',{
                                                'status': False, 
                                                'error': f"Couldn't delete {installation_path}, please delete it manually and restart the app",
                                                'model_name' : model_name,
                                                'binding_folder' : binding_folder
                                            }, room=sid)
            )


    @sio.on('cancel_install')
    def cancel_install(sid,data):
        try:
            model_name = data["model_name"]
            binding_folder = data["binding_folder"]
            model_url = data["model_url"]
            signature = f"{model_name}_{binding_folder}_{model_url}"
            lollmsElfServer.download_infos[signature]["cancel"]=True

            run_async( partial(sio.emit,'canceled', {
                                            'status': True
                                            },
                                            room=sid 
                                ) 
            )           
        except Exception as ex:
            trace_exception(ex)
            sio.emit('canceled', {
                                            'status': False,
                                            'error':str(ex)
                                            },
                                            room=sid 
                                )            
