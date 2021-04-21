ROOT=~/projects/stories
export GIN_DIR=${ROOT}/text-to-text-transfer-transformer/t5/models/gin/

export T5_ENV=~/envs/stories/
export DATA_FILE=${ROOT}/glucose/t5_data/t5_training_data.tsv

export OUTPUT_FILE=${ROOT}/glucose/outputs
export INPUT_FILE=${ROOT}/glucose/t5_data/t5_test_data.txt

export PROJECT=stories-305118
export ZONE=your_project_zone
export BUCKET=gs://stories-gcp/
export TPU_NAME=grpc://10.117.169.58:8470
export MODEL_DIR=${ROOT}/t5_models/finetuned/
export DATA_DIR=${ROOT}/t5_tfds/
