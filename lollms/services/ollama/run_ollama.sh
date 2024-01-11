#!/bin/bash

# Set the OLLAMA_HOST address
OLLAMA_HOST="0.0.0.0:11434"

# Start the OLLAMA server
OLLAMA_MODELS=~/ollama/models ollama serve &

# Check if models.txt exists
if [ ! -f models.txt ]; then
    # Create models.txt and add "mixtral" to it
    echo "mistral" > models.txt
fi

# Read the models from the file
while IFS= read -r model
do
    # Run each model in the background
    ollama run "$model" &
done < models.txt

# Wait for all background processes to finish
wait
