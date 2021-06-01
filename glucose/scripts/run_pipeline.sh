#!/bin/bash
# This script will run the pipeline with the default arguments and save locations.
# Refer to the individual scripts for more options.
# Before running, you should download the datasets -- refer to data/README.md

# choices for EXP_NUM are one of ['0', '1', '2a', '2b', '3a', '3b']
EXP_NUM=0
MODEL_SIZE=t5-large

python scripts/preprocess.py $EXP_NUM -m $MODEL_SIZE

#if running on nlpgrid, set CUDA_VISIBLE_DEVICES appropriately
python scripts/train.py $EXP_NUM -m $MODEL_SIZE
