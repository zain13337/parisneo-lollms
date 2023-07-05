import os
import subprocess
import platform

repo_dir = 'socket.io-client-cpp'

# Check if the repository directory exists
if os.path.exists(repo_dir):
    # If it exists, perform a git pull to update the repository
    os.chdir(repo_dir)
    subprocess.run(['git', 'pull'])
else:
    # If it doesn't exist, clone the Socket.IO Client C++ library
    subprocess.run(['git', 'clone', 'https://github.com/socketio/socket.io-client-cpp.git', repo_dir])

    # Build the Socket.IO Client C++ library
    os.chdir(repo_dir)

if platform.system() == 'Windows':
    os.makedirs('build', exist_ok=True)
    os.chdir('build')

    # Detect if running on Windows Command Prompt (cmd) or PowerShell
    shell = os.environ.get('SHELL')
    if shell and 'powershell' in shell.lower():
        subprocess.run(['cmake', '..'])
        subprocess.run(['cmake', '--build', '.', '--config', 'Release'])
    else:
        # Use the Visual Studio build tools
        subprocess.run(['cmake', '-G', 'Visual Studio 16 2019', '..'])
        subprocess.run(['cmake', '--build', '.', '--config', 'Release'])

else:
    subprocess.run(['mkdir', 'build'])
    os.chdir('build')
    subprocess.run(['cmake', '..'])
    subprocess.run(['make'])

# Compile the C++ code
os.chdir('../..')
if platform.system() == 'Windows':
    subprocess.run(['cl', '/EHsc', '/Isocket.io-client-cpp/src',
                    'main.cpp', 'socket.io-client-cpp/build/Release/sioclient.lib', '/FeSocketIOClientExample.exe'])
else:
    subprocess.run(['g++', '-std=c++11', '-Isocket.io-client-cpp/src',
                    'main.cpp', 'socket.io-client-cpp/build/libsioclient.a', '-o', 'SocketIOClientExample'])

print('Compilation completed successfully.')
