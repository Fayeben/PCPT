#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
set -x
GPUS=$1

PY_ARGS=${@:2}

CUDA_VISIBLE_DEVICES=${GPUS} python main.py ${PY_ARGS}
