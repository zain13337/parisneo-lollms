from flask import Flask, send_from_directory, request, jsonify
from lollms.helpers import  get_trace_exception
import json
import shutil
import os

app = Flask(__name__, static_folder='dist/')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/<path:filename>')
def serve_file(filename):
    root_dir = "dist/"
    return send_from_directory(root_dir, filename)

@app.route("/save_preset", methods=["POST"])
def save_preset():
  """Saves a preset to a file.

  Args:
    None.

  Returns:
    None.
  """

  # Get the JSON data from the POST request.
  preset_data = request.get_json()

  # Save the JSON data to a file.
  with open("presets.json", "w") as f:
    json.dump(preset_data, f)

  return "Preset saved successfully!"

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


@app.route("/get_presets", methods=["GET"])
def get_presets(self):
    presets_file = self.lollms_paths.personal_databases_path/"presets.json"
    if not presets_file.exists():
        shutil.copy("presets/presets.json",presets_file)
    with open(presets_file) as f:
        data = json.loads(f.read())
    return jsonify(data)


@app.route("/save_presets", methods=["POST"])
def save_presets(self):
    """Saves a preset to a file.

    Args:
        None.

    Returns:
        None.
    """

    # Get the JSON data from the POST request.
    preset_data = request.get_json()

    presets_file = self.lollms_paths.personal_databases_path/"presets.json"
    # Save the JSON data to a file.
    with open(presets_file, "w") as f:
        json.dump(preset_data, f, indent=4)

    return "Preset saved successfully!"


def main():
    app.run()

if __name__ == '__main__':
    main()
