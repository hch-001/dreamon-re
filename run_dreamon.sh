#!/bin/bash
dataset=opencoder-stage2-edu
lr=1e-5

# Create unique ckpt_dir for each LR
ckpt_dir=checkpoints/infill_dreamon
log_dir=logs
timestamp=$(date +%F_%H-%M-%S)
stdout_log=$log_dir/run_${timestamp}_stdout.log
stderr_log=$log_dir/run_${timestamp}_stderr.log

mkdir -p "$log_dir"

(echo "Training with learning rate: $lr"
# Set Hugging Face cache directory
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface

echo "Training with learning rate: $lr"
echo "Checkpoint directory: $ckpt_dir"
echo "Using HF cache: $HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 --master-port 12346 \
    -m src.trainer.fsdp_sft_expand_trainer1 \
    diffusion.time_reweighting=linear \
    diffusion.weight_eos=true \
    data.train_files=data/${dataset}/train_data.parquet \
    data.val_files=data/${dataset}/eval_data.parquet \
    data.train_batch_size=128 \
    data.max_length=1024 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.truncation=right \
	data.middle_line_num=null \
    data.use_uniform_merge_prob=0.5\
    optim.lr=1e-5 \
    data.micro_batch_size_per_gpu=32 \
    model.partial_pretrain=Dream-org/Dream-Coder-v0-Instruct-7B \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$ckpt_dir \
    trainer.project_name=diff-mask_expansion \
    trainer.total_epochs=5 \
    trainer.experiment_name=${dataset}_${lr}_infill_$(date +%F) \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null \
    trainer.save_checkpoint_steps=200 \
    ulysses_sequence_parallel_size=1 ) >"$stdout_log" 2>"$stderr_log"
