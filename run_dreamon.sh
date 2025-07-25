#!/bin/bash
dataset=opencoder-stage2-edu
lr=1e-5

# Create unique ckpt_dir for each LR
ckpt_dir=checkpoints/infill_dreamon

echo "Training with learning rate: $lr"
echo "Checkpoint directory: $ckpt_dir"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc_per_node=8 --master-port 12346 \
    -m src.trainer.fsdp_sft_expand_trainer \
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
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=Dream-org/Dream-Coder-v0-Base-7B \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$ckpt_dir \
    trainer.project_name=diff-mask_expansion \
    trainer.total_epochs=5 \
    trainer.experiment_name=${dataset}_${lr}_infill_$(date +%F) \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null \
    trainer.save_checkpoint_steps=2000 \
    ulysses_sequence_parallel_size=1 
