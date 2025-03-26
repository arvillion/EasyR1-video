# set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export WANDB_MODE="offline"

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
EXPERIMENT_NAME=baseline
SAVE_CHECKPOINT_PATH="/hhd2/zzk/r1/${EXPERIMENT_NAME}"
REWARD_SCORE="iou"
SYSTEM_PROMPT=$(<prompts/initial_prompt.yaml)

# SYSTEM_PROMPT=$(echo "$SYSTEM_PROMPT" | sed -e "s/'/'\"'\"'/g" -e 's/"/\\"/g')

# echo "$SYSTEM_PROMPT"



CUDA_VISIBLE_DEVICES=1,4,5,6 python3 -m verl.trainer.main \
    config=examples/video/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.compute_score=${REWARD_SCORE} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.max_steps=50 \
    trainer.save_freq=50 \
    trainer.save_checkpoint_path=${SAVE_CHECKPOINT_PATH} \
    trainer.remove_previous_ckpt=True \
    data.system_prompt="${SYSTEM_PROMPT}" \
    data.shuffle=False

