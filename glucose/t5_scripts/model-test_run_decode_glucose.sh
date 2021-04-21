#!/bin/bash
source t5_scripts/env_colab.sh

mkdir -p "$(dirname "$OUTPUT_FILE")"

${T5_ENV}/t5_mesh_transformer --tpu="${TPU_NAME}" --gcp_project="${PROJECT}" --tpu_zone="${ZONE}" \
 --model_dir="${MODEL_DIR}" --gin_file="${MODEL_DIR}/operative_config.gin" --gin_file="sample_decode.gin" \
 --gin_file="infer.gin" --gin_param="utils.run.mode = 'infer'" --gin_param="infer_checkpoint_step='all'" \
 --gin_param="input_filename = '${INPUT_FILE}'" --gin_param="output_filename = '${OUTPUT_FILE}'" \
 --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'"

# remove checkpoint prefix
mv ${OUTPUT_FILE}-* ${OUTPUT_FILE}

echo "wrote predictions to ${OUTPUT_FILE}"
