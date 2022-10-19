#!/bin/bash
# This script will run the pipeline with the default arguments and save locations.
# Refer to the individual scripts for more options.

MODEL_SIZE=t5-large
# set GPUS appropriately
GPUS=0,1

SPEC=
# SPEC="--specific-only"

METRIC=sent_vecs

if [ $MODEL_SIZE == "t5-large" ]; then
    BATCH_SIZE=8
    BATCH_SIZE_INF=150
else
    BATCH_SIZE=40
    BATCH_SIZE_INF=300
fi

# 0 1 2a 2b 3a A
for EXP_NUM in 0 1 2a 2b 3a cis2; do
    EXP_NAME=exp${EXP_NUM}_${MODEL_SIZE}

    python scripts/preprocess.py $EXP_NUM -ms $MODEL_SIZE --val_ids data/val_ids_small.txt -sm $METRIC

    python scripts/train.py $EXP_NUM -m $MODEL_SIZE -bse 6 -bst $BATCH_SIZE $SPEC -sm $METRIC
    # python scripts/train.py $EXP_NUM -m $MODEL_SIZE -bst $BATCH_SIZE --eval_em

    ## by default, runs inference on validation
    python scripts/inference.py $EXP_NUM -ms $MODEL_SIZE \
        --batch_size $BATCH_SIZE_INF $SPEC -sm $METRIC
    ## specify -d to run inference on test dataset
    python scripts/inference.py $EXP_NUM -ms $MODEL_SIZE \
        -d data/${EXP_NAME}/ds_test  --batch_size $BATCH_SIZE_INF $SPEC -sm $METRIC

    ## for evaluating val, we use our own script, since we don't have 3 references
    python scripts/evaluation_val.py $EXP_NUM -ms $MODEL_SIZE

    ## for evaluating test, we use an adapted version of the GLUCOSE original evaluation script.
    python scripts/evaluation_test.py -s outputs/${EXP_NAME}/ -ce

    python scripts/evaluation_cis2.py $EXP_NUM -ms $MODEL_SIZE -sm $METRIC
done
