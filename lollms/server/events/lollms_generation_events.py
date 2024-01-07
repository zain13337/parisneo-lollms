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
from lollms.utilities import load_config, trace_exception, gc
from pathlib import Path
from typing import List
import socketio
import os

router = APIRouter()
lollmsElfServer = LOLLMSElfServer.get_instance()


# ----------------------------------- events -----------------------------------------
def add_events(sio:socketio):
    @sio.on('cancel_generation')
    def cancel_generation(sid, environ):
        client_id = sid
        lollmsElfServer.cancel_gen = True
        #kill thread
        ASCIIColors.error(f'Client {sid} requested cancelling generation')
        terminate_thread(lollmsElfServer.connections[client_id]['generation_thread'])
        ASCIIColors.error(f'Client {sid} canceled generation')
        lollmsElfServer.busy=False
    
    
    @sio.on('cancel_text_generation')
    def cancel_text_generation(sid, environ):
        client_id = sid
        lollmsElfServer.connections[client_id]["requested_stop"]=True
        print(f"Client {client_id} requested canceling generation")
        lollmsElfServer.socketio.emit("generation_canceled", {"message":"Generation is canceled."}, room=client_id)
        lollmsElfServer.socketio.sleep(0)
        lollmsElfServer.busy = False


    # A copy of the original lollms-server generation code needed for playground
    @sio.on('generate_text')
    def handle_generate_text(sid, environ, data):
        client_id = sid
        lollmsElfServer.cancel_gen = False
        ASCIIColors.info(f"Text generation requested by client: {client_id}")
        if lollmsElfServer.busy:
            lollmsElfServer.socketio.emit("busy", {"message":"I am busy. Come back later."}, room=client_id)
            lollmsElfServer.socketio.sleep(0)
            ASCIIColors.warning(f"OOps request {client_id}  refused!! Server busy")
            return
        def generate_text():
            lollmsElfServer.busy = True
            try:
                model = lollmsElfServer.model
                lollmsElfServer.connections[client_id]["is_generating"]=True
                lollmsElfServer.connections[client_id]["requested_stop"]=False
                prompt          = data['prompt']
                tokenized = model.tokenize(prompt)
                personality_id  = data.get('personality', -1)

                n_crop          = data.get('n_crop', len(tokenized))
                if n_crop!=-1:
                    prompt          = model.detokenize(tokenized[-n_crop:])

                n_predicts      = data["n_predicts"]
                parameters      = data.get("parameters",{
                    "temperature":lollmsElfServer.config["temperature"],
                    "top_k":lollmsElfServer.config["top_k"],
                    "top_p":lollmsElfServer.config["top_p"],
                    "repeat_penalty":lollmsElfServer.config["repeat_penalty"],
                    "repeat_last_n":lollmsElfServer.config["repeat_last_n"],
                    "seed":lollmsElfServer.config["seed"]
                })

                if personality_id==-1:
                    # Raw text generation
                    lollmsElfServer.answer = {"full_text":""}
                    def callback(text, message_type: MSG_TYPE, metadata:dict={}):
                        if message_type == MSG_TYPE.MSG_TYPE_CHUNK:
                            ASCIIColors.success(f"generated:{len(lollmsElfServer.answer['full_text'].split())} words", end='\r')
                            if text is not None:
                                lollmsElfServer.answer["full_text"] = lollmsElfServer.answer["full_text"] + text
                                lollmsElfServer.socketio.emit('text_chunk', {'chunk': text, 'type':MSG_TYPE.MSG_TYPE_CHUNK.value}, room=client_id)
                                lollmsElfServer.socketio.sleep(0)
                        if client_id in lollmsElfServer.connections:# Client disconnected                      
                            if lollmsElfServer.connections[client_id]["requested_stop"]:
                                return False
                            else:
                                return True
                        else:
                            return False                            

                    tk = model.tokenize(prompt)
                    n_tokens = len(tk)
                    fd = model.detokenize(tk[-min(lollmsElfServer.config.ctx_size-n_predicts,n_tokens):])

                    try:
                        ASCIIColors.print("warming up", ASCIIColors.color_bright_cyan)
                        
                        generated_text = model.generate(fd, 
                                                        n_predict=n_predicts, 
                                                        callback=callback,
                                                        temperature = parameters["temperature"],
                                                        top_k = parameters["top_k"],
                                                        top_p = parameters["top_p"],
                                                        repeat_penalty = parameters["repeat_penalty"],
                                                        repeat_last_n = parameters["repeat_last_n"],
                                                        seed = parameters["seed"],                                           
                                                        )
                        ASCIIColors.success(f"\ndone")

                        if client_id in lollmsElfServer.connections:
                            if not lollmsElfServer.connections[client_id]["requested_stop"]:
                                # Emit the generated text to the client
                                lollmsElfServer.socketio.emit('text_generated', {'text': generated_text}, room=client_id)                
                                lollmsElfServer.socketio.sleep(0)
                    except Exception as ex:
                        lollmsElfServer.socketio.emit('generation_error', {'error': str(ex)}, room=client_id)
                        ASCIIColors.error(f"\ndone")
                    lollmsElfServer.busy = False
                else:
                    try:
                        personality: AIPersonality = lollmsElfServer.personalities[personality_id]
                        ump = lollmsElfServer.config.discussion_prompt_separator +lollmsElfServer.config.user_name.strip() if lollmsElfServer.config.use_user_name_in_discussions else lollmsElfServer.personality.user_message_prefix
                        personality.model = model
                        cond_tk = personality.model.tokenize(personality.personality_conditioning)
                        n_cond_tk = len(cond_tk)
                        # Placeholder code for text generation
                        # Replace this with your actual text generation logic
                        print(f"Text generation requested by client: {client_id}")

                        lollmsElfServer.answer["full_text"] = ''
                        full_discussion_blocks = lollmsElfServer.connections[client_id]["full_discussion_blocks"]

                        if prompt != '':
                            if personality.processor is not None and personality.processor_cfg["process_model_input"]:
                                preprocessed_prompt = personality.processor.process_model_input(prompt)
                            else:
                                preprocessed_prompt = prompt
                            
                            if personality.processor is not None and personality.processor_cfg["custom_workflow"]:
                                full_discussion_blocks.append(ump)
                                full_discussion_blocks.append(preprocessed_prompt)
                        
                            else:

                                full_discussion_blocks.append(ump)
                                full_discussion_blocks.append(preprocessed_prompt)
                                full_discussion_blocks.append(personality.link_text)
                                full_discussion_blocks.append(personality.ai_message_prefix)

                        full_discussion = personality.personality_conditioning + ''.join(full_discussion_blocks)

                        def callback(text, message_type: MSG_TYPE, metadata:dict={}):
                            if message_type == MSG_TYPE.MSG_TYPE_CHUNK:
                                lollmsElfServer.answer["full_text"] = lollmsElfServer.answer["full_text"] + text
                                lollmsElfServer.socketio.emit('text_chunk', {'chunk': text}, room=client_id)
                                lollmsElfServer.socketio.sleep(0)
                            try:
                                if lollmsElfServer.connections[client_id]["requested_stop"]:
                                    return False
                                else:
                                    return True
                            except: # If the client is disconnected then we stop talking to it
                                return False

                        tk = personality.model.tokenize(full_discussion)
                        n_tokens = len(tk)
                        fd = personality.model.detokenize(tk[-min(lollmsElfServer.config.ctx_size-n_cond_tk-personality.model_n_predicts,n_tokens):])
                        
                        if personality.processor is not None and personality.processor_cfg["custom_workflow"]:
                            ASCIIColors.info("processing...")
                            generated_text = personality.processor.run_workflow(prompt, previous_discussion_text=personality.personality_conditioning+fd, callback=callback)
                        else:
                            ASCIIColors.info("generating...")
                            generated_text = personality.model.generate(
                                                                        personality.personality_conditioning+fd, 
                                                                        n_predict=personality.model_n_predicts, 
                                                                        callback=callback)

                        if personality.processor is not None and personality.processor_cfg["process_model_output"]: 
                            generated_text = personality.processor.process_model_output(generated_text)

                        full_discussion_blocks.append(generated_text.strip())
                        ASCIIColors.success("\ndone")

                        # Emit the generated text to the client
                        lollmsElfServer.socketio.emit('text_generated', {'text': generated_text}, room=client_id)
                        lollmsElfServer.socketio.sleep(0)
                    except Exception as ex:
                        lollmsElfServer.socketio.emit('generation_error', {'error': str(ex)}, room=client_id)
                        ASCIIColors.error(f"\ndone")
                    lollmsElfServer.busy = False
            except Exception as ex:
                    trace_exception(ex)
                    lollmsElfServer.socketio.emit('generation_error', {'error': str(ex)}, room=client_id)
                    lollmsElfServer.busy = False

        # Start the text generation task in a separate thread
        task = lollmsElfServer.socketio.start_background_task(target=generate_text)
