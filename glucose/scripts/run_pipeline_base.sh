#!/bin/bash
# This script will run the pipeline with the default arguments and save locations.
# Refer to the individual scripts for more options.
# Before running, you should download the datasets -- refer to data/README.md

MODEL_SIZE=t5-base
# set GPUS appropriately
GPUS=5

if [ MODEL_SIZE == "t5-large" ]; then
    BATCH_SIZE=8
    BATCH_SIZE_INF=200
else
    BATCH_SIZE=40
    BATCH_SIZE_INF=300
fi

# 0 1 2a 2b 3a
for EXP_NUM in 0; do
    EXP_NAME=exp${EXP_NUM}_${MODEL_SIZE}

    python scripts/preprocess.py $EXP_NUM -ms $MODEL_SIZE

    CUDA_VISIBLE_DEVICES=${GPUS} python scripts/train.py $EXP_NUM -m $MODEL_SIZE -bse $BATCH_SIZE -bst $BATCH_SIZE

    ## by default, runs inference on validation
    CUDA_VISIBLE_DEVICES=${GPUS} python scripts/inference.py $EXP_NUM -ms $MODEL_SIZE \
        --batch_size $BATCH_SIZE_INF
    python scripts/evaluation.py $EXP_NUM -ms $MODEL_SIZE

    ## for evaluating test, we use an adapted version of the GLUCOSE original evaluation script.
    ## specify -d to run inference on test
    CUDA_VISIBLE_DEVICES=${GPUS} python scripts/inference.py $EXP_NUM -ms $MODEL_SIZE \
        -d data/${EXP_NAME}/ds_test  --batch_size $BATCH_SIZE_INF
    python scripts/evaluation.py $EXP_NUM -ms $MODEL_SIZE -i outputs/${EXP_NAME}/model/predictions_test.csv

    ## evaluate baseline only for original experiment (exp0)
    if [ "$EXP_NUM" == "0" ]; then
        python scripts/evaluation_baseline.py -s outputs/${EXP_NAME}/
    fi
done

echo "wrote all results to /home1/b/bryanli/projects/stories/glucose/outputs/all_results.tsv"
