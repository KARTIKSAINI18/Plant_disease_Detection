#!/bin/bash
pyenv install 3.10.13 -s
pyenv global 3.10.13
pip install --upgrade pip
pip install -r requirements.txt
