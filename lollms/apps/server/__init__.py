from lollms.config import InstallOption
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from lollms.types import MSG_TYPE
from lollms.personality import AIPersonality
from lollms.main_config import LOLLMSConfig
from lollms.binding import LLMBinding, BindingBuilder, ModelBuilder
from lollms.personality import PersonalityBuilder
from lollms.apps.console import MainMenu
from lollms.paths import LollmsPaths
from lollms.apps.console import MainMenu
from lollms.app import LollmsApplication
from safe_store import TextVectorizer
from ascii_colors import ASCIIColors, trace_exception
from typing import List, Tuple
from typing import Callable
import importlib
from pathlib import Path
import argparse
import logging
import yaml
import copy
import gc
import json
import traceback
import shutil
import psutil
import subprocess
import pkg_resources

def reset_all_installs(lollms_paths:LollmsPaths):
    ASCIIColors.info("Removeing all configuration files to force reinstall")
    ASCIIColors.info(f"Searching files from {lollms_paths.personal_configuration_path}")
    for file_path in lollms_paths.personal_configuration_path.iterdir():
        if file_path.name!=f"{lollms_paths.tool_prefix}local_config.yaml" and file_path.suffix.lower()==".yaml":
            file_path.unlink()
            ASCIIColors.info(f"Deleted file: {file_path}")


class LoLLMsServer(LollmsApplication):
    def __init__(self):
        
        self.clients = {}
        self.current_binding = None
        self.model = None
        self.personalities = []
        self.answer = ['']
        self.busy = False
        
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--host', '-hst', default=None, help='Host name')
        parser.add_argument('--port', '-prt', default=None, help='Port number')

        
        parser.add_argument('--reset_personal_path', action='store_true', help='Reset the personal path')
        parser.add_argument('--reset_config', action='store_true', help='Reset the configurations')
        parser.add_argument('--reset_installs', action='store_true', help='Reset all installation status')
        parser.add_argument('--default_cfg_path', type=str, default=None, help='Reset all installation status')


        args = parser.parse_args()

        if args.reset_installs:
            self.reset_all_installs()

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
        else:
            lollms_paths = LollmsPaths.find_paths(force_local=False, tool_prefix="lollms_server_")

        # Configuration loading part
        config = LOLLMSConfig.autoload(lollms_paths, None)

        if args.host is not None:
            config.host = args.host

        if args.port is not None:
            config.port = args.port

        super().__init__("lollms-server", config, lollms_paths)

        self.menu.show_logo()        
        self.app = Flask("LoLLMsServer")
        #self.app.config['SECRET_KEY'] = 'lollmssecret'
        CORS(self.app)  # Enable CORS for all routes

        def get_config():
            ASCIIColors.yellow("Requested configuration")
            return jsonify(self.config.to_dict())
        
        self.app.add_url_rule(
            "/get_config", "get_config", get_config, methods=["GET"]
        )

        def list_models():
            if self.binding is not None:
                models = self.binding.list_models(self.config)
                ASCIIColors.yellow("Listing models")
                return jsonify(models)
            else:
                return jsonify([])
        
        self.app.add_url_rule(
            "/list_models", "list_models", list_models, methods=["GET"]
        )

        def get_active_model():
            if self.binding is not None:
                models = self.binding.list_models(self.config)
                index = models.index(self.config.model_name)
                ASCIIColors.yellow(f"Recovering active model: {models[index]}")
                return jsonify({"model":models[index],"index":index})
            else:
                return jsonify(None)



        self.app.add_url_rule(
            "/get_active_model", "get_active_model", get_active_model, methods=["GET"]
        )

            
        def get_server_ifos(self):
            """
            Returns information about the server.
            """
            server_infos = {}
            if self.binding is not None:
                models = self.binding.list_models(self.config)
                index = models.index(self.config.model_name)
                ASCIIColors.yellow(f"Recovering active model: {models[index]}")
                server_infos["binding"]=self.binding.name
                server_infos["models_list"]=models
                server_infos["model"]=models[index]
                server_infos["model_index"]=index
            else:
                server_infos["models_list"]=[]
                server_infos["model"]=""
                server_infos["model_index"]=-1

            ram = psutil.virtual_memory()
            server_infos["lollms_version"]= pkg_resources.get_distribution('lollms').version
            server_infos["total_space"]=ram.total
            server_infos["available_space"]=ram.free

            server_infos["percent_usage"]=ram.percent
            server_infos["ram_usage"]=ram.used

            try:
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.used,gpu_name', '--format=csv,nounits,noheader'])
                lines = output.decode().strip().split('\n')
                vram_info = [line.split(',') for line in lines]
                server_infos["nb_gpus"]= len(vram_info)
                if vram_info is not None:
                    for i, gpu in enumerate(vram_info):
                        server_infos[f"gpu_{i}_total_vram"] = int(gpu[0])*1024*1024
                        server_infos[f"gpu_{i}_used_vram"] = int(gpu[1])*1024*1024
                        server_infos[f"gpu_{i}_model"] = gpu[2].strip()
                else:
                    # Set all VRAM-related entries to None
                    server_infos["gpu_0_total_vram"] = None
                    server_infos["gpu_0_used_vram"] = None
                    server_infos["gpu_0_model"] = None


            except (subprocess.CalledProcessError, FileNotFoundError):
                server_infos["nb_gpus"]= 0
                server_infos["gpu_0_total_vram"] = None
                server_infos["gpu_0_used_vram"] = None
                server_infos["gpu_0_model"] = None            

            current_drive = Path.cwd().anchor
            drive_disk_usage = psutil.disk_usage(current_drive)
            server_infos["system_disk_total_space"]=drive_disk_usage.total
            server_infos["system_disk_available_space"]=drive_disk_usage.free
            try:
                models_folder_disk_usage = psutil.disk_usage(str(self.lollms_paths.binding_models_paths[0]))
                server_infos["binding_disk_total_space"]=models_folder_disk_usage.total
                server_infos["binding_disk_available_space"]=models_folder_disk_usage.free
                             
            except Exception as ex:
                server_infos["binding_disk_total_space"]=None
                server_infos["binding_disk_available_space"]=None

            return jsonify(server_infos)

        self.app.add_url_rule(
            "/get_server_ifos", "get_server_ifos", get_server_ifos, methods=["GET"]
        )        

        def update_setting():
            data = request.get_json()
            setting_name = data['setting_name']

            if setting_name== "model_name":
                self.config["model_name"]=data['setting_value']
                if self.config["model_name"] is not None:
                    try:
                        self.model = None
                        if self.binding:
                            del self.binding

                        self.binding = None
                        for per in self.mounted_personalities:
                            per.model = None
                        gc.collect()
                        self.binding = BindingBuilder().build_binding(self.config, self.lollms_paths, app=self)
                        self.model = self.binding.build_model()
                        for per in self.mounted_personalities:
                            per.model = self.model
                    except Exception as ex:
                        # Catch the exception and get the traceback as a list of strings
                        traceback_lines = traceback.format_exception(type(ex), ex, ex.__traceback__)

                        # Join the traceback lines into a single string
                        traceback_text = ''.join(traceback_lines)
                        ASCIIColors.error(f"Couldn't load model: [{ex}]")
                        ASCIIColors.error(traceback_text)
                        return jsonify({ "status":False, 'error':str(ex)})
                else:
                    ASCIIColors.warning("Trying to set a None model. Please select a model for the binding")
                print("update_settings : New model selected")

            return jsonify({'setting_name': data['setting_name'], "status":True})

        self.app.add_url_rule(
            "/update_setting", "update_setting", update_setting, methods=["POST"]
        )



        def list_mounted_personalities():
            if self.binding is not None:
                ASCIIColors.yellow("Listing mounted personalities")
                return jsonify(self.config.personalities)
            else:
                return jsonify([])
        
        self.app.add_url_rule(
            "/list_mounted_personalities", "list_mounted_personalities", list_mounted_personalities, methods=["GET"]
        )


        self.socketio = SocketIO(self.app, cors_allowed_origins='*', ping_timeout=1200, ping_interval=4000)

        # Set log level to warning
        self.app.logger.setLevel(logging.WARNING)
        # Configure a custom logger for Flask-SocketIO
        self.socketio_log = logging.getLogger('socketio')
        self.socketio_log.setLevel(logging.WARNING)
        self.socketio_log.addHandler(logging.StreamHandler())

        self.initialize_routes()
        self.run(self.config.host, self.config.port)


    def initialize_routes(self):
        @self.socketio.on('connect')
        def handle_connect():
            client_id = request.sid
            self.clients[client_id] = {
                    "namespace": request.namespace,
                    "full_discussion_blocks": [],
                    "is_generating":False,
                    "requested_stop":False
                }
            ASCIIColors.success(f'Client connected with session ID: {client_id}')

        @self.socketio.on('disconnect')
        def handle_disconnect():
            client_id = request.sid
            if client_id in self.clients:
                del self.clients[client_id]
            print(f'Client disconnected with session ID: {client_id}')



        @self.socketio.on('list_available_bindings')
        def handle_list_bindings():
            binding_infs = []
            for p in self.bindings_path.iterdir():
                if p.is_dir():
                    with open(p/"binding_card.yaml", "r") as f:
                        card = yaml.safe_load(f)
                    with open(p/"models.yaml", "r") as f:
                        models = yaml.safe_load(f)
                    entry={
                        "name":p.name,
                        "card":card,
                        "models":models
                    }
                    binding_infs.append(entry)

            emit('bindings_list', {'success':True, 'bindings': binding_infs}, room=request.sid)

        @self.socketio.on('list_available_personalities')
        def handle_list_available_personalities():
            personalities_folder = self.personalities_path
            personalities = {}
            for language_folder in personalities_folder.iterdir():
                if language_folder.is_dir():
                    personalities[language_folder.name] = {}
                    for category_folder in  language_folder.iterdir():
                        if category_folder.is_dir():
                            personalities[language_folder.name][category_folder.name] = []
                            for personality_folder in category_folder.iterdir():
                                if personality_folder.is_dir():
                                    try:
                                        personality_info = {"folder":personality_folder.stem}
                                        config_path = personality_folder / 'config.yaml'
                                        with open(config_path) as config_file:
                                            config_data = yaml.load(config_file, Loader=yaml.FullLoader)
                                            personality_info['name'] = config_data.get('name',"No Name")
                                            personality_info['description'] = config_data.get('personality_description',"")
                                            personality_info['author'] = config_data.get('author', 'ParisNeo')
                                            personality_info['version'] = config_data.get('version', '1.0.0')
                                        scripts_path = personality_folder / 'scripts'
                                        personality_info['has_scripts'] = scripts_path.is_dir()
                                        assets_path = personality_folder / 'assets'
                                        gif_logo_path = assets_path / 'logo.gif'
                                        webp_logo_path = assets_path / 'logo.webp'
                                        png_logo_path = assets_path / 'logo.png'
                                        jpg_logo_path = assets_path / 'logo.jpg'
                                        jpeg_logo_path = assets_path / 'logo.jpeg'
                                        bmp_logo_path = assets_path / 'logo.bmp'
                                        
                                        personality_info['has_logo'] = png_logo_path.is_file() or gif_logo_path.is_file()
                                        
                                        if gif_logo_path.exists():
                                            personality_info['avatar'] = str(gif_logo_path).replace("\\","/")
                                        elif webp_logo_path.exists():
                                            personality_info['avatar'] = str(webp_logo_path).replace("\\","/")
                                        elif png_logo_path.exists():
                                            personality_info['avatar'] = str(png_logo_path).replace("\\","/")
                                        elif jpg_logo_path.exists():
                                            personality_info['avatar'] = str(jpg_logo_path).replace("\\","/")
                                        elif jpeg_logo_path.exists():
                                            personality_info['avatar'] = str(jpeg_logo_path).replace("\\","/")
                                        elif bmp_logo_path.exists():
                                            personality_info['avatar'] = str(bmp_logo_path).replace("\\","/")
                                        else:
                                            personality_info['avatar'] = ""
                                        personalities[language_folder.name][category_folder.name].append(personality_info)
                                    except Exception as ex:
                                        print(f"Couldn't load personality from {personality_folder} [{ex}]")
            emit('personalities_list', {'personalities': personalities}, room=request.sid)

        @self.socketio.on('list_available_models')
        def handle_list_available_models():
            """List the available models

            Returns:
                _type_: _description_
            """
            if self.binding is None:
               emit('available_models_list', {'success':False, 'error': "No binding selected"}, room=request.sid)
            model_list = self.binding.get_available_models()

            models = []
            for model in model_list:
                try:
                    filename = model.get('filename',"")
                    server = model.get('server',"")
                    image_url = model.get("icon", '/images/default_model.png')
                    license = model.get("license", 'unknown')
                    owner = model.get("owner", 'unknown')
                    owner_link = model.get("owner_link", 'https://github.com/ParisNeo')
                    filesize = int(model.get('filesize',0))
                    description = model.get('description',"")
                    model_type = model.get("model_type","")
                    if server.endswith("/"):
                        path = f'{server}{filename}'
                    else:
                        path = f'{server}/{filename}'
                    local_path = self.models_path/f'{self.config["binding_name"]}/{filename}'
                    is_installed = local_path.exists() or model_type.lower()=="api"
                    models.append({
                        'title': filename,
                        'icon': image_url,  # Replace with the path to the model icon
                        'license': license,
                        'owner': owner,
                        'owner_link': owner_link,
                        'description': description,
                        'isInstalled': is_installed,
                        'path': path,
                        'filesize': filesize,
                        'model_type': model_type
                    })
                except Exception as ex:
                    print("#################################")
                    print(ex)
                    print("#################################")
                    print(f"Problem with model : {model}")
            emit('available_models_list', {'success':True, 'available_models': models}, room=request.sid)


        @self.socketio.on('list_available_personalities_categories')
        def handle_list_available_personalities_categories(data):
            try:
                categories = [l for l in (self.personalities_path).iterdir()]
                emit('available_personalities_categories_list', {'success': True, 'available_personalities_categories': categories})
            except Exception as ex:
                emit('available_personalities_categories_list', {'success': False, 'error':str(ex)})

        @self.socketio.on('list_available_personalities_names')
        def handle_list_available_personalities_names(data):
            try:
                category = data["category"]
                personalities = [l for l in (self.personalities_path/category).iterdir()]
                emit('list_available_personalities_names_list', {'success': True, 'list_available_personalities_names': personalities})
            except Exception as ex:
                emit('list_available_personalities_names_list', {'success': False, 'error':str(ex)})

        @self.socketio.on('select_binding')
        def handle_select_binding(data):
            self.cp_config = copy.deepcopy(self.config)
            self.cp_config["binding_name"] = data['binding_name']
            try:
                del self.model
                del self.binding
                self.model = None
                self.binding = None
                gc.collect()
                for personality in self.mount_personalities:
                    personality.model = None
                self.binding = self.build_binding(self.bindings_path, self.cp_config)
                self.config = self.cp_config
                self.mount_personalities()
                gc.collect()                
                emit('select_binding', {'success':True, 'binding_name': self.cp_config["binding_name"]}, room=request.sid)
            except Exception as ex:
                print(ex)
                emit('select_binding', {'success':False, 'binding_name': self.cp_config["binding_name"], 'error':f"Couldn't load binding:\n{ex}"}, room=request.sid)

        @self.socketio.on('select_model')
        def handle_select_model(data):
            model_name = data['model_name']
            if self.binding is None:
                emit('select_model', {'success':False, 'model_name':  model_name, 'error':f"Please select a binding first"}, room=request.sid)
                return
            self.cp_config = copy.deepcopy(self.config)
            self.cp_config["model_name"] = data['model_name']
            try:
                del self.model
                self.model = None
                gc.collect()
                for personality in self.mount_personalities:
                    personality.model = None
                self.model = self.binding.build_model()
                self.mount_personalities()
                gc.collect()
                emit('select_model', {'success':True, 'model_name':  model_name}, room=request.sid)
            except Exception as ex:
                print(ex)
                emit('select_model', {'success':False, 'model_name':  model_name, 'error':f"Please select a binding first"}, room=request.sid)

        @self.socketio.on('add_personality')
        def handle_add_personality(data):
            personality_path = data['path']
            try:
                personality = AIPersonality(
                                                personality_path, 
                                                self.lollms_paths, 
                                                self.config,
                                                self.model,
                                                self
                                            )
                self.personalities.append(personality)
                self.config["personalities"].append(personality_path)
                emit('personality_added', {'success':True, 'name': personality.name, 'id':len(self.personalities)-1}, room=request.sid)
                self.config.save_config()
            except Exception as e:
                error_message = str(e)
                emit('personality_add_failed', {'success':False, 'error': error_message}, room=request.sid)


        
        @self.socketio.on('vectorize_text')
        def vectorize_text(parameters:dict):
            """Vectorizes text

            Args:
                parameters (dict): contains
                'chunk_size': the maximum size of a text chunk (512 by default)
                'vectorization_method': can be either "model_embedding" or "ftidf_vectorizer" (default is "ftidf_vectorizer")
                'payloads': a list of dicts. each entry has the following format
                {
                    "path": the path to the document
                    "text": the text of the document
                },
                'return_database': If true the vectorized database will be sent to the client (default is True)
                'database_path': the path to store the database (default is none)
                
            returns a dict
                status: True if success and false if not
                if you asked for the database to be sent back you will ahve those fields too:
                embeddings: a dictionary containing the text chunks with their ids and embeddings
                "texts": a dictionary of text chunks for each embedding (use index for correspondance)
                "infos": extra information
                "vectorizer": The vectorize if this is using tfidf or none if it uses model
                
            """
            vectorization_method = parameters.get('vectorization_method',"ftidf_vectorizer")
            chunk_size = parameters.get("chunk_size",512)
            payloads = parameters["payloads"]
            database_path = parameters.get("database_path",None)
            return_database = parameters.get("return_database",True)
            if database_path is None and return_database is None:
                ASCIIColors.warning("Vectorization should either ask to save the database or to recover it. You didn't ask for any one!")
                emit('vectorized_db',{"status":False, "error":"Vectorization should either ask to save the database or to recover it. You didn't ask for any one!"}) 
                return
            tv = TextVectorizer(vectorization_method, self.model)
            for payload in payloads:
                tv.add_document(payload["path"],payload["text"],chunk_size=chunk_size)
            json_db = tv.toJson()
            if return_database:
                emit('vectorized_db',{**{"status":True}, **json_db}) 
            else:
                emit('vectorized_db',{"status":True}) 
                with open(database_path, "w") as file:
                    json.dump(json_db, file, indent=4)
                
            
        @self.socketio.on('query_database')
        def query_database(parameters:dict):
            """queries a database

            Args:
                parameters (dict): contains
                'vectorization_method': can be either "model_embedding" or "ftidf_vectorizer"
                'database': a list of dicts. each entry has the following format
                {
                    embeddings: a dictionary containing the text chunks with their ids and embeddings
                    "texts": a dictionary of text chunks for each embedding (use index for correspondance)
                    "infos": extra information
                    "vectorizer": The vectorize if this is using tfidf or none if it uses model
                }
                'database_path': If supplied, the database is loaded from a path
                'query': a query to search in the database
            """
            vectorization_method = parameters['vectorization_method']
            database = parameters.get("database",None)
            query = parameters.get("query",None)
            if query is None:
                ASCIIColors.error("No query given!")
                emit('vector_db_query',{"status":False, "error":"Please supply a query"}) 
                return
            
            if database is None:
                database_path = parameters.get("database_path",None)
                if database_path is None:
                    ASCIIColors.error("No database given!")
                    emit('vector_db_query',{"status":False, "error":"You did not supply a database file nor a database content"}) 
                    return
                else:
                    with open(database_path, "r") as file:
                        database = json.load(file)
                        
            tv = TextVectorizer(vectorization_method, self.model, database_dict=database)
            docs, sorted_similarities = tv.recover_text(tv.embed_query(query))
            emit('vectorized_db',{
                "chunks":docs,
                "refs":sorted_similarities
            }) 
            
            
        @self.socketio.on('list_active_personalities')
        def handle_list_active_personalities():
            personality_names = [p.name for p in self.personalities]
            emit('active_personalities_list', {'success':True, 'personalities': personality_names}, room=request.sid)

        @self.socketio.on('activate_personality')
        def handle_activate_personality(data):
            personality_id = data['id']
            if personality_id<len(self.personalities):
                self.active_personality=self.personalities[personality_id]
                emit('activate_personality', {'success':True, 'name': self.active_personality, 'id':len(self.personalities)-1}, room=request.sid)
                self.config["active_personality_id"]=personality_id
                self.config.save_config()
            else:
                emit('personality_add_failed', {'success':False, 'error': "Personality ID not valid"}, room=request.sid)

        @self.socketio.on('tokenize')
        def tokenize(data):
            client_id = request.sid
            prompt = data['prompt']
            tk = self.model.tokenize(prompt)
            emit("tokenized", {"tokens":tk}, room=client_id)

        @self.socketio.on('detokenize')
        def detokenize(data):
            client_id = request.sid
            prompt = data['prompt']
            txt = self.model.detokenize(prompt)
            emit("detokenized", {"text":txt}, room=client_id)

        @self.socketio.on('embed')
        def detokenize(data):
            client_id = request.sid
            prompt = data['prompt']
            txt = self.model.embed(prompt)
            self.socketio.emit("embeded", {"text":txt}, room=client_id)

        @self.socketio.on('cancel_text_generation')
        def cancel_text_generation(data):
            client_id = request.sid
            self.clients[client_id]["requested_stop"]=True
            print(f"Client {client_id} requested canceling generation")
            self.socketio.emit("generation_canceled", {"message":"Generation is canceled."}, room=client_id)
            self.socketio.sleep(0)
            self.busy = False


        # A copy of the original lollms-server generation code needed for playground
        @self.socketio.on('generate_text')
        def handle_generate_text(data):
            client_id = request.sid
            ASCIIColors.info(f"Text generation requested by client: {client_id}")
            if self.busy:
                self.socketio.emit("busy", {"message":"I am busy. Come back later."}, room=client_id)
                self.socketio.sleep(0)
                ASCIIColors.warning(f"OOps request {client_id}  refused!! Server busy")
                return
            def generate_text():
                self.busy = True
                try:
                    model = self.model
                    self.clients[client_id]["is_generating"]=True
                    self.clients[client_id]["requested_stop"]=False
                    prompt              = data['prompt']
                    tokenized           = model.tokenize(prompt)
                    personality_id      = int(data.get('personality', -1))

                    n_crop              = int(data.get('n_crop', len(tokenized)))
                    if n_crop!=-1:
                        prompt          = model.detokenize(tokenized[-n_crop:])

                    n_predicts          = int(data.get("n_predicts",1024))
                    parameters          = data.get("parameters",{
                        "temperature":data.get("temperature",self.config["temperature"]),
                        "top_k":data.get("top_k",self.config["top_k"]),
                        "top_p":data.get("top_p",self.config["top_p"]),
                        "repeat_penalty":data.get("repeat_penalty",self.config["repeat_penalty"]),
                        "repeat_last_n":data.get("repeat_last_n",self.config["repeat_last_n"]),
                        "seed":data.get("seed",self.config["seed"])
                    })

                    if personality_id==-1:
                        # Raw text generation
                        self.answer = {"full_text":""}
                        def callback(text, message_type: MSG_TYPE, metadata:dict={}):
                            if message_type == MSG_TYPE.MSG_TYPE_CHUNK:
                                ASCIIColors.success(f"generated:{len(self.answer['full_text'].split())} words", end='\r')
                                self.answer["full_text"] = self.answer["full_text"] + text
                                self.socketio.emit('text_chunk', {'chunk': text, 'type':MSG_TYPE.MSG_TYPE_CHUNK.value}, room=client_id)
                                self.socketio.sleep(0)
                            if client_id in self.clients:# Client disconnected                      
                                if self.clients[client_id]["requested_stop"]:
                                    return False
                                else:
                                    return True
                            else:
                                return False                            

                        tk = model.tokenize(prompt)
                        n_tokens = len(tk)
                        fd = model.detokenize(tk[-min(self.config.ctx_size-n_predicts,n_tokens):])

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
                                                            seed = parameters["seed"]                                                
                                                            )
                            ASCIIColors.success(f"\ndone")
                            if client_id in self.clients:
                                if not self.clients[client_id]["requested_stop"]:
                                    # Emit the generated text to the client
                                    self.socketio.emit('text_generated', {'text': generated_text}, room=client_id)                
                                    self.socketio.sleep(0)
                        except Exception as ex:
                            self.socketio.emit('generation_error', {'error': str(ex)}, room=client_id)
                            ASCIIColors.error(f"\ndone")
                        self.busy = False
                    else:
                        try:
                            personality: AIPersonality = self.personalities[personality_id]
                            ump = self.config.discussion_prompt_separator +self.config.user_name+": " if self.config.use_user_name_in_discussions else self.personality.user_message_prefix
                            personality.model = model
                            cond_tk = personality.model.tokenize(personality.personality_conditioning)
                            n_cond_tk = len(cond_tk)
                            # Placeholder code for text generation
                            # Replace this with your actual text generation logic
                            print(f"Text generation requested by client: {client_id}")

                            self.answer["full_text"] = ''
                            full_discussion_blocks = self.clients[client_id]["full_discussion_blocks"]

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
                                    self.answer["full_text"] = self.answer["full_text"] + text
                                    self.socketio.emit('text_chunk', {'chunk': text}, room=client_id)
                                    self.socketio.sleep(0)
                                try:
                                    if self.clients[client_id]["requested_stop"]:
                                        return False
                                    else:
                                        return True
                                except: # If the client is disconnected then we stop talking to it
                                    return False

                            tk = personality.model.tokenize(full_discussion)
                            n_tokens = len(tk)
                            fd = personality.model.detokenize(tk[-min(self.config.ctx_size-n_cond_tk-personality.model_n_predicts,n_tokens):])
                            
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
                            self.socketio.emit('text_generated', {'text': generated_text}, room=client_id)
                            self.socketio.sleep(0)
                        except Exception as ex:
                            self.socketio.emit('generation_error', {'error': str(ex)}, room=client_id)
                            ASCIIColors.error(f"\ndone")
                        self.busy = False
                except Exception as ex:
                        trace_exception(ex)
                        self.socketio.emit('generation_error', {'error': str(ex)}, room=client_id)
                        self.busy = False

            # Start the text generation task in a separate thread
            task = self.socketio.start_background_task(target=generate_text)
            
    def build_binding(self, bindings_path: Path, cfg: LOLLMSConfig)->LLMBinding:
        binding_path = Path(bindings_path) / cfg["binding_name"]
        # define the full absolute path to the module
        absolute_path = binding_path.resolve()
        # infer the module name from the file path
        module_name = binding_path.stem
        # use importlib to load the module from the file path
        loader = importlib.machinery.SourceFileLoader(module_name, str(absolute_path / "__init__.py"))
        binding_module = loader.load_module()
        binding = getattr(binding_module, binding_module.binding_name)
        return binding


    def run(self, host="localhost", port="9601"):
        if self.binding is None:
            ASCIIColors.warning("No binding selected. Please select one")
            self.menu.select_binding()

        print(f"{ASCIIColors.color_red}Current binding (model) : {ASCIIColors.color_reset}{self.binding}")
        print(f"{ASCIIColors.color_red}Mounted personalities : {ASCIIColors.color_reset}{self.config.personalities}")
        if len(self.config.personalities)==0:
            ASCIIColors.warning("No personality selected. Selecting lollms. You can mount other personalities using lollms-settings application")
            self.config.personalities = ["generic/lollms"]
            self.config.save_config()

        if self.config.active_personality_id>=len(self.config.personalities):
            self.config.active_personality_id = 0
        print(f"{ASCIIColors.color_red}Current personality : {ASCIIColors.color_reset}{self.config.personalities[self.config.active_personality_id]}")
        ASCIIColors.info(f"Serving on address: http://{host}:{port}")

        self.socketio.run(self.app, host=host, port=port)

def main():
    LoLLMsServer()


if __name__ == '__main__':
    main()