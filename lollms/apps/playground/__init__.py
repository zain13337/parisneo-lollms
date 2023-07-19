from flask import Flask, send_from_directory
import os

app = Flask(__name__, static_folder='static/')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/<path:filename>')
def serve_file(filename):
    root_dir = "static/"
    return send_from_directory(root_dir, filename)

def main():
    app.run()

if __name__ == '__main__':
    main()
