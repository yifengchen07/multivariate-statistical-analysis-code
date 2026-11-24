#!/bin/zsh
set -eu
cd -- "${0:A:h}/resnet224"
RESNET_PYTHON=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
if [[ ! -x "$RESNET_PYTHON" ]]; then
  RESNET_PYTHON=$(command -v python3)
fi
/usr/bin/caffeinate -i "$RESNET_PYTHON" -u run.py --device mps
