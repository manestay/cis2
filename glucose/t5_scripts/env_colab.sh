WS=/content/drive/MyDrive/research/narratives/workspace
ROOT=$WS/glucose
export GIN_DIR=${ROOT}/text-to-text-transfer-transformer/t5/models/gin/

export T5_ENV=/usr/local/bin/
export DATA_FILE=${ROOT}/t5_data/t5_training_data.tsv

export OUTPUT_FILE=${ROOT}/outputs/glucose_t5_large/test_predictions.tsv
export OUTPUT_FILE="/content/out.txt"
export INPUT_FILE=${ROOT}/t5_data/t5_test_data_small.txt

export PROJECT=stories-305118
export ZONE=us-east1-b
export BUCKET=gs://stories-gcp/glucose
export MODEL_DIR=${BUCKET}/glucose_t5_large
export DATA_DIR=${ROOT}/t5_tfds/

export TPU_SIZE=2x2