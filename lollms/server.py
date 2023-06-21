from lollms.config import InstallOption
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from lollms.types import MSG_TYPE
from lollms.personality import AIPersonality
from lollms.main_config import LOLLMSConfig
from lollms.binding import LLMBinding, BindingBuilder, ModelBuilder
from lollms.personality import PersonalityBuilder
from lollms.helpers import ASCIIColors
from lollms.console import MainMenu
from lollms.paths import LollmsPaths
from lollms.console import MainMenu
from typing import List, Tuple
import importlib
from pathlib import Path
import argparse
import logging
import yaml
import copy

def reset_all_installs(lollms_paths:LollmsPaths):
    ASCIIColors.info("Removeing all configuration files to force reinstall")
    ASCIIColors.info(f"Searching files from {lollms_paths.personal_configuration_path}")
    for file_path in lollms_paths.personal_configuration_path.iterdir():
        if file_path.name!="local_config.yaml" and file_path.suffix.lower()==".yaml":
            file_path.unlink()
            ASCIIColors.info(f"Deleted file: {file_path}")


class LoLLMsServer:
    def __init__(self):
        host = "localhost"
        port = "9601"
        self.clients = {}
        self.current_binding = None
        self.active_model = None
        self.personalities = []
        self.answer = ['']
        self.is_ready = True
        
        
        self.lollms_paths = LollmsPaths.find_paths(force_local=False)
        self.menu = MainMenu(self)
        parser = argparse.ArgumentParser()
        parser.add_argument('--host', '-hst', default=host, help='Host name')
        parser.add_argument('--port', '-prt', default=port, help='Port number')

        parser.add_argument('--config', '-cfg', default=None, help='Path to the configuration file')
        parser.add_argument('--bindings_path', '-bp', default=str(self.lollms_paths.bindings_zoo_path),
                            help='The path to the Bindings folder')
        parser.add_argument('--personalities_path', '-pp',
                            default=str(self.lollms_paths.personalities_zoo_path),
                            help='The path to the personalities folder')
        parser.add_argument('--models_path', '-mp', default=str(self.lollms_paths.personal_models_path),
                            help='The path to the models folder')

        parser.add_argument('--binding_name', '-b', default="llama_cpp_official",
                            help='Binding to be used by default')
        parser.add_argument('--model_name', '-m', default=None,
                            help='Model name')
        parser.add_argument('--personality_full_name', '-p', default="personality",
                            help='Personality path relative to the personalities folder (language/category/name)')
        
        parser.add_argument('--reset_personal_path', action='store_true', help='Reset the personal path')
        parser.add_argument('--reset_config', action='store_true', help='Reset the configurations')
        parser.add_argument('--reset_installs', action='store_true', help='Reset all installation status')


        args = parser.parse_args()

        if args.reset_installs:
            reset_all_installs()

        if args.reset_personal_path:
            LollmsPaths.reset_configs()

        if args.reset_config:
            cfg_path = LollmsPaths.find_paths().personal_configuration_path / "local_config.yaml"
            try:
                cfg_path.unlink()
                ASCIIColors.success("LOLLMS configuration reset successfully")
            except:
                ASCIIColors.success("Couldn't reset LOLLMS configuration")

        # Configuration loading part
        self.config = LOLLMSConfig.autoload(self.lollms_paths, args.config)


        if args.binding_name:
            self.config.binding_name = args.binding_name

        if args.model_name:
            self.config.model_name = args.model_name

        # Recover bindings path
        self.personalities_path = Path(args.personalities_path)
        self.bindings_path = Path(args.bindings_path)
        self.models_path = Path(args.models_path)
        if self.config.binding_name is None:
            self.menu.select_binding()
        else:
            self.binding = BindingBuilder().build_binding(self.config, self.lollms_paths)
        if self.config.model_name is None:
            self.menu.select_model()
        else:
            try:
                self.active_model = self.binding.build_model()
            except Exception as ex:
                print(f"{ASCIIColors.color_red}Couldn't load model Please select a valid model{ASCIIColors.color_reset}")
                print(f"{ASCIIColors.color_red}{ex}{ASCIIColors.color_reset}")
                self.menu.select_model()

        for p in self.config.personalities:
            personality = AIPersonality(
                                            self.config.lollms_paths.personalities_zoo_path/p,
                                            self.lollms_paths,
                                            self.config,
                                            self.active_model
                                        )
            self.personalities.append(personality)

        if self.config.active_personality_id>len(self.personalities):
            self.config.active_personality_id = 0
        self.active_personality = self.personalities[self.config.active_personality_id]

        self.menu.show_logo()        
        
        self.app = Flask("LoLLMsServer")
        #self.app.config['SECRET_KEY'] = 'lollmssecret'
        CORS(self.app)  # Enable CORS for all routes
        
        self.socketio = SocketIO(self.app, cors_allowed_origins='*', ping_timeout=1200, ping_interval=4000)

        # Set log level to warning
        self.app.logger.setLevel(logging.WARNING)
        # Configure a custom logger for Flask-SocketIO
        self.socketio_log = logging.getLogger('socketio')
        self.socketio_log.setLevel(logging.WARNING)
        self.socketio_log.addHandler(logging.StreamHandler())

        self.initialize_routes()
        self.run(args.host, args.port)


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
                print(f"Couldn't find binding. Please verify your configuration file at {self.config.file_path} or use the next menu to select a valid binding")
                print(f"Trying to reinstall binding")
                self.binding = BindingBuilder().build_binding(self.config, self.lollms_paths, InstallOption.FORCE_INSTALL)
                self.menu.select_binding()

    def load_model(self):
        try:
            self.active_model = ModelBuilder(self.binding).get_model()
            ASCIIColors.success("Model loaded successfully")
        except Exception as ex:
            ASCIIColors.error(f"Couldn't load model.")
            ASCIIColors.error(f"Binding returned this exception : {ex}")
            ASCIIColors.error(f"{self.config.get_model_path_infos()}")
            print("Please select a valid model or install a new one from a url")
            self.menu.select_model()

    def load_personality(self):
        try:
            self.personality = PersonalityBuilder(self.lollms_paths, self.config, self.model).build_personality()
        except Exception as ex:
            ASCIIColors.error(f"Couldn't load personality.")
            ASCIIColors.error(f"Binding returned this exception : {ex}")
            ASCIIColors.error(f"{self.config.get_personality_path_infos()}")
            print("Please select a valid model or install a new one from a url")
            self.menu.select_model()
        self.cond_tk = self.personality.model.tokenize(self.personality.personality_conditioning)
        self.n_cond_tk = len(self.cond_tk)

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

        @self.socketio.on('list_available_personalities_languages')
        def handle_list_available_personalities_languages():
            try:
                languages = [l for l in self.personalities_path.iterdir()]
                emit('available_personalities_languages_list', {'success': True, 'available_personalities_languages': languages})
            except Exception as ex:
                emit('available_personalities_languages_list', {'success': False, 'error':str(ex)})

        @self.socketio.on('list_available_personalities_categories')
        def handle_list_available_personalities_categories(data):
            try:
                language = data["language"]
                categories = [l for l in (self.personalities_path/language).iterdir()]
                emit('available_personalities_categories_list', {'success': True, 'available_personalities_categories': categories})
            except Exception as ex:
                emit('available_personalities_categories_list', {'success': False, 'error':str(ex)})

        @self.socketio.on('list_available_personalities_names')
        def handle_list_available_personalities_names(data):
            try:
                language = data["language"]
                category = data["category"]
                personalities = [l for l in (self.personalities_path/language/category).iterdir()]
                emit('list_available_personalities_names_list', {'success': True, 'list_available_personalities_names': personalities})
            except Exception as ex:
                emit('list_available_personalities_names_list', {'success': False, 'error':str(ex)})

        @self.socketio.on('select_binding')
        def handle_select_binding(data):
            self.cp_config = copy.deepcopy(self.config)
            self.cp_config["binding_name"] = data['binding_name']
            try:
                self.binding = self.build_binding(self.bindings_path, self.cp_config)
                self.config = self.cp_config
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
                self.active_model = self.binding.build_model()
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
                                                self.active_model
                                            )
                self.personalities.append(personality)
                self.config["personalities"].append(personality_path)
                emit('personality_added', {'success':True, 'name': personality.name, 'id':len(self.personalities)-1}, room=request.sid)
                self.config.save_config()
            except Exception as e:
                error_message = str(e)
                emit('personality_add_failed', {'success':False, 'error': error_message}, room=request.sid)


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
            prompt = data['prompt']
            tk = self.active_model.tokenize(prompt)
            emit("tokenized", {"tokens":tk})

        @self.socketio.on('detokenize')
        def detokenize(data):
            prompt = data['prompt']
            txt = self.active_model.detokenize(prompt)
            emit("detokenized", {"text":txt})

        @self.socketio.on('cancel_generation')
        def cancel_generation(data):
            client_id = request.sid
            self.clients[client_id]["requested_stop"]=True
            print(f"Client {client_id} requested canceling generation")
            emit("generation_canceled", {"message":"Generation is canceled."})
            self.socketio.sleep(0)


        @self.socketio.on('generate_text')
        def handle_generate_text(data):
            client_id = request.sid
            ASCIIColors.info(f"Text generation requested by client: {client_id}")
            if not self.is_ready:
                emit("buzzy", {"message":"I am buzzy. Come back later."})
                self.socketio.sleep(0)
                ASCIIColors.warning(f"OOps request {client_id}  refused!! Server buzy")
                return
            def generate_text():
                self.is_ready = False
                model = self.active_model
                self.clients[client_id]["is_generating"]=True
                self.clients[client_id]["requested_stop"]=False
                prompt          = data['prompt']
                personality_id  = data['personality']
                n_predicts      = data["n_predicts"]
                parameters      = data.get("parameters",{
                    "temperature":self.config["temperature"],
                    "top_k":self.config["top_k"],
                    "top_p":self.config["top_p"],
                    "repeat_penalty":self.config["repeat_penalty"],
                    "repeat_last_n":self.config["repeat_last_n"],
                    "seed":self.config["seed"]
                })

                if personality_id==-1:
                    # Raw text generation
                    self.answer = {"full_text":""}
                    def callback(text, message_type: MSG_TYPE):
                        if message_type == MSG_TYPE.MSG_TYPE_CHUNK:
                            print(f"generated:{len(self.answer['full_text'])} words", end='\r')
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
                    fd = model.detokenize(tk[-min(self.config.ctx_size,n_tokens):])

                    ASCIIColors.print("warm up", ASCIIColors.color_bright_cyan)
                    try:
                        generated_text = model.generate(fd, n_predict=n_predicts, callback=callback,
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
                    self.is_ready = True
                else:
                    try:
                        personality: AIPersonality = self.personalities[personality_id]
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
                                full_discussion_blocks.append(personality.user_message_prefix)
                                full_discussion_blocks.append(preprocessed_prompt)
                        
                            else:

                                full_discussion_blocks.append(personality.user_message_prefix)
                                full_discussion_blocks.append(preprocessed_prompt)
                                full_discussion_blocks.append(personality.link_text)
                                full_discussion_blocks.append(personality.ai_message_prefix)

                        full_discussion = personality.personality_conditioning + ''.join(full_discussion_blocks)

                        def callback(text, message_type: MSG_TYPE):
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
                        fd = personality.model.detokenize(tk[-min(self.config.ctx_size-n_cond_tk,n_tokens):])
                        
                        if personality.processor is not None and personality.processor_cfg["custom_workflow"]:
                            print("processing...", end="", flush=True)
                            generated_text = personality.processor.run_workflow(prompt, previous_discussion_text=personality.personality_conditioning+fd, callback=callback)
                        else:
                            ASCIIColors.info("generating...", end="", flush=True)
                            generated_text = personality.model.generate(
                                                                        personality.personality_conditioning+fd, 
                                                                        n_predict=personality.model_n_predicts, 
                                                                        callback=callback)

                        if personality.processor is not None and personality.processor_cfg["process_model_output"]: 
                            generated_text = personality.processor.process_model_output(generated_text)

                        full_discussion_blocks.append(generated_text.strip())
                        ASCIIColors.success("\ndone", end="", flush=True)

                        # Emit the generated text to the client
                        self.socketio.emit('text_generated', {'text': generated_text}, room=client_id)
                        self.socketio.sleep(0)
                    except Exception as ex:
                        self.socketio.emit('generation_error', {'error': str(ex)}, room=client_id)
                        ASCIIColors.error(f"\ndone")
                    self.is_ready = True
            # Start the text generation task in a separate thread
            self.socketio.start_background_task(target=generate_text, once=True)
            generate_text()
            
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
        print(f"{ASCIIColors.color_red}Current personality : {ASCIIColors.color_reset}{self.active_personality}")
        ASCIIColors.info(f"Serving on address: http://{host}:{port}")

        self.socketio.run(self.app, host=host, port=port)

def main():
    LoLLMsServer()


if __name__ == '__main__':
    main()