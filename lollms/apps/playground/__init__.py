from flask import Flask, send_from_directory, request, jsonify
from lollms.helpers import  get_trace_exception
from lollms.paths import LollmsPaths
from lollms.main_config import LOLLMSConfig
from pathlib import Path
import requests
import json
import shutil
import yaml
import os


lollms_paths = LollmsPaths.find_paths(force_local=False, tool_prefix="lollms_server_")
config = LOLLMSConfig.autoload(lollms_paths)

app = Flask(__name__, static_folder='dist/')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/<path:filename>')
def serve_file(filename):
    root_dir = "dist/"
    return send_from_directory(root_dir, filename)


@app.route("/execute_python_code", methods=["POST"])
def execute_python_code():
    """Executes Python code and returns the output."""
    data = request.get_json()
    code = data["code"]
    # Import the necessary modules.
    import io
    import sys
    import time

    # Create a Python interpreter.
    interpreter = io.StringIO()
    sys.stdout = interpreter

    # Execute the code.
    start_time = time.time()
    try:
        exec(code)
        # Get the output.
        output = interpreter.getvalue()
    except Exception as ex:
        output = str(ex)+"\n"+get_trace_exception(ex)
    end_time = time.time()

    return jsonify({"output":output,"execution_time":end_time - start_time}) 



def copy_files(src, dest):
    for item in os.listdir(src):
        src_file = os.path.join(src, item)
        dest_file = os.path.join(dest, item)
        
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dest_file)

@app.route("/get_presets", methods=["GET"])
def get_presets():
    presets_folder = lollms_paths.personal_databases_path/"lollms_playground_presets"
    if not presets_folder.exists():
        presets_folder.mkdir(exist_ok=True, parents=True)
        path = Path(__file__).parent/"presets"
        copy_files(str(path),presets_folder)
    presets = []
    for filename in presets_folder.glob('*.yaml'):
        print(filename)
        with open(filename, 'r', encoding='utf-8') as file:
            preset = yaml.safe_load(file)
            if preset is not None:
                presets.append(preset)
    return jsonify(presets)

@app.route("/add_preset", methods=["POST"])
def add_preset():
    # Get the JSON data from the POST request.
    preset_data = request.get_json()
    presets_folder = lollms_paths.personal_databases_path/"lollms_playground_presets"
    if not presets_folder.exists():
        presets_folder.mkdir(exist_ok=True, parents=True)
        copy_files("presets",presets_folder)
    fn = preset_data.name.lower().replace(" ","_")
    filename = presets_folder/f"{fn}.yaml"
    with open(filename, 'w', encoding='utf-8') as file:
        yaml.dump(preset_data)
    return jsonify({"status": True})

@app.route("/del_preset", methods=["POST"])
def del_preset():
    presets_folder = lollms_paths.personal_databases_path/"lollms_playground_presets"
    if not presets_folder.exists():
        presets_folder.mkdir(exist_ok=True, parents=True)
        copy_files("presets",presets_folder)
    presets = []
    for filename in presets_folder.glob('*.yaml'):
        print(filename)
        with open(filename, 'r') as file:
            preset = yaml.safe_load(file)
            if preset is not None:
                presets.append(preset)
    return jsonify(presets)

@app.route("/save_preset", methods=["POST"])
def save_presets():
    """Saves a preset to a file.

    Args:
        None.

    Returns:
        None.
    """

    # Get the JSON data from the POST request.
    preset_data = request.get_json()

    presets_file = lollms_paths.personal_databases_path/"presets.json"
    # Save the JSON data to a file.
    with open(presets_file, "w") as f:
        json.dump(preset_data, f, indent=4)

    return jsonify({"status":True,"message":"Preset saved successfully!"})

@app.route("/list_models", methods=["GET"])
def list_models():
    host = f"http://{config.host}:{config.port}"
    response = requests.get(f"{host}/list_models")
    data = response.json()
    return jsonify(data)


@app.route("/get_active_model", methods=["GET"])
def get_active_model():
    host = f"http://{config.host}:{config.port}"
    response = requests.get(f"{host}/get_active_model")
    data = response.json()
    return jsonify(data)


@app.route("/update_setting", methods=["POST"])
def update_setting():
    host = f"http://{config.host}:{config.port}"
    response = requests.post(f"{host}/update_setting", headers={'Content-Type': 'application/json'}, data=request.data)
    data = response.json()
    return jsonify(data)




def main():
    app.run()

if __name__ == '__main__':
    main()
