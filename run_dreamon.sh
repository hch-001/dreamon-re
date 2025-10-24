#!/bin/bash
dataset=opencoder-stage2-edu
lr=1e-5
patience=2
min_delta=1e-4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      dataset="$2"
      shift 2
      ;;
    --lr)
      lr="$2"
      shift 2
      ;;
    --patience)
      patience="$2"
      shift 2
      ;;
    --min-delta)
      min_delta="$2"
      shift 2
      ;;
    --revision)
      revision="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
用法: ./run_dreamon.sh [options]
可选项:
  --dataset <name>     训练数据集名称 (默认: opencoder-stage2-edu)
  --lr <float>         学习率 (默认: 1e-5)
  --patience <int>     早停耐心步数 (默认: 2)
  --min-delta <float>  早停最小提升阈值 (默认: 1e-4)
  --revision <name>    HuggingFace 模型的 revision/branch/tag (默认: null)
EOF
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      exit 1
      ;;
  esac
done

# Create unique ckpt_dir for each LR
ckpt_dir=checkpoints/infill_dreamon
log_dir=logs
timestamp=$(date +%F_%H-%M-%S)
stdout_log=$log_dir/run_${timestamp}_stdout.log
stderr_log=$log_dir/run_${timestamp}_stderr.log

mkdir -p "$log_dir"

exec > >(tee "$stdout_log")
exec 2> >(tee "$stderr_log" >&2)

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface

echo "Training with learning rate: $lr"
echo "Checkpoint directory: $ckpt_dir"
if [[ "$revision" != "null" ]]; then
  echo "Model revision: $revision"
fi
echo "Early stopping patience: $patience"
echo "Early stopping min_delta: $min_delta"

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2      --master-port 12346 \
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
    model.revision=dev2 \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$ckpt_dir \
    trainer.project_name=diff-mask_expansion \
    trainer.total_epochs=5 \
    trainer.experiment_name=${dataset}_${lr}_infill_$(date +%F) \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null \
    trainer.save_checkpoint_steps=400 \
    ulysses_sequence_parallel_size=1
