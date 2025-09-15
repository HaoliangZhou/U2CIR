export CUDA_VISIBLE_DEVICES=0
TASK_SCENE=natural  # change with 'natural' or 'fashion'
LOG_NAME=CLCoIR_seq_Finetune
CURRENT_TIME=$(date +%Y-%m-%d_%H:%M:%S)
PROJECT_DIR=/U2CAR_CODE
OUT_DIR=${PROJECT_DIR}/outputs/save_seq_finetune_${TASK_SCENE}/${LOG_NAME}
LOG_DIR=${PROJECT_DIR}/outputs/logger/log_${TASK_SCENE}/${LOG_NAME}_${CURRENT_TIME}.log


cd ${PROJECT_DIR}/train
python -m torch.distributed.run --nproc_per_node=1 --master_port=14509 train_Finetune.py \
--config ${PROJECT_DIR}/configs/exp/free.yaml \
--base_config ${PROJECT_DIR}/configs/base_seqF_${TASK_SCENE}.yaml \
--output_dir ${OUT_DIR} \
2>&1 | tee ${LOG_DIR}
