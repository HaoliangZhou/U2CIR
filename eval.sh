export CUDA_VISIBLE_DEVICES=6
CURRENT_TIME=$(date +%Y-%m-%d_%H:%M:%S)
LOG_NAME=U2CAR
OUT_DIR=/checkpoint/model_weights/${LOG_NAME}

python eval.py \
--output_dir ${OUT_DIR} \
2>&1 | tee ./outputs/${LOG_NAME}_${CURRENT_TIME}.log
