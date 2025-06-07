#!/usr/bin/sh

set -e

source ./.venv/bin/activate

python train.py "$@"
