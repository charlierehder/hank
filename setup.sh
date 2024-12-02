#!/bin/bash

FUNCTION_NAME="hank"
SCRIPT_PATH="$(pwd)/hank.py"

# Hank Bash Command Assistant
# wrapper around OpenAI API that returns a suggested command 
# given a natural language prompt

# define bash function and forward to .bashrc
{
    echo ""
    echo "# Added by hank script setup"
    echo "hank() {"
    echo "	python3 \"$SCRIPT_PATH\" \"\$@\"" 
    echo "}"
} >> ~/.bashrc

# reload shell configuration
source ~/.bashrc

echo "$FUNCTION_NAME was added."
