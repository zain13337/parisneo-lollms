from flask import Flask, send_from_directory, request
import json
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

def main():
    app.run()

if __name__ == '__main__':
    main()
