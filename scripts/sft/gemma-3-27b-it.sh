source $HOME/sleeperer-agents/.env
hf auth login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd $HOME
read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path $HOME/models/gemma-3-27b-it-lora-$1 \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --zero_stage 2 \
    --bf16 \
    --max_epochs 1 \
    --pretrain $HOME/models/gemma-3-27b-it \
    --learning_rate 5e-5 \
    --adam_betas 0.9 0.999 \
    --dataset $HOME/sleeperer-agents/data/train/$1.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 2048 \
    --use_wandb True \
    --wandb_project liars \
    --wandb_run_name gemma-3-27b-it-lora-$1 \
    --seed 123456 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj
EOF
deepspeed \
--module $training_commands


# only run the following commands if the deepspeed command succeeded
if [ $? -eq 0 ]; then
    # remove wandb logs
    rm -rf $HOME/wandb
fi