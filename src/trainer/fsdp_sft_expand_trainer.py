# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import math
import re
from contextlib import nullcontext

class FileLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
    def log(self, data):
        with open(self.log_path, 'a') as f:
            json.dump(data, f)
            f.write('\n')

import hydra
import numpy as np
import torch
import torch.distributed
import verl.utils.hdfs_io as hdfs_io
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, PreTrainedModel
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from src.trainer.sft_expand_dataset import SFTExpandDataset

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    """从检查点名称或路径中提取步数。

    Args:
        path (str): 检查点名称或路径，格式如 "global_step_1000" 或 "/path/to/global_step_1000"
        
    Returns:
        int | None: 检查点步数，如果无法提取则返回 None
    """
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(obj, (ListConfig, DictConfig)):
        return (
            {k: convert_to_regular_types(v) for k, v in obj.items()}
            if isinstance(obj, DictConfig)
            else list(obj)
        )
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj

class FSDPSFTTrainer(object):
    # 全局计数器
    ex_num = 0  # expand token的总数
    ma_num = 0  # mask token的总数
    total_batches = 0  # 处理的总批次数

    def __init__(
        self, config, device_mesh: DeviceMesh, ulysses_device_mesh: DeviceMesh
    ):
        from omegaconf import OmegaConf
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        
        # 设置默认目录，使用 OmegaConf.create 创建新的配置
        default_config = {
            "trainer": {
                "default_local_dir": getattr(config.trainer, "default_local_dir", "checkpoints"),
                "log_dir": getattr(config.trainer, "log_dir", None)
            }
        }
        default_config["trainer"]["log_dir"] = default_config["trainer"]["log_dir"] or \
            os.path.join(default_config["trainer"]["default_local_dir"], "logs")
            
        # 合并配置
        self.config = OmegaConf.merge(default_config, config)
        
        # 创建日志和检查点目录
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)  # 检查点目录
        os.makedirs(self.config.trainer.log_dir, exist_ok=True)  # 日志目录
        
        # 修改文件日志器路径
        log_path = os.path.join(self.config.trainer.log_dir, "training_log.jsonl")
        
        # 修改loss历史记录路径
        self.loss_history_path = os.path.join(self.config.trainer.log_dir, 'loss_history.json')
        
        # Add tracking for current epoch
        self.current_epoch = 0

        # Best checkpoint tracking (only keep best)
        self.best_val_loss = float("inf")
        self.min_delta = getattr(self.config.trainer, "min_delta", 1e-4)
        self.best_checkpoint_path = None
        
        # 用于跟踪当前训练步骤
        self.current_step = 0
        
        # 添加信号处理器用于 CTRL+C
        if self.device_mesh.get_rank() == 0:  # 只在主进程添加处理器
            import signal
            def signal_handler(signum, frame):
                print("\nReceived Ctrl+C. Attempting to save checkpoint...")
                try:
                    # 如果已经有最佳检查点，就不需要再保存
                    if not self.best_checkpoint_path:
                        self.save_checkpoint(step=self.current_step, is_best=False)
                        print(f"Emergency checkpoint saved at step {self.current_step}")
                    else:
                        print("Best checkpoint already exists, skipping emergency save")
                except Exception as e:
                    print(f"Failed to save emergency checkpoint: {e}")
                finally:
                    print("Exiting...")
                    import sys
                    sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
        
        # Early stopping configuration
        self.no_improvement_count = 0
        patience_cfg = getattr(self.config.trainer, "patient", None)
        if patience_cfg is None:
            patience_cfg = getattr(self.config.trainer, "patience", None)
        if patience_cfg is None:
            patience_cfg = getattr(self.config.trainer, "max_no_improvement", None)
        self.early_stopping_patience = (
            int(patience_cfg) if patience_cfg is not None else None
        )

        # Check if resuming training
        self.resume_training = getattr(self.config.trainer, "resume_training", False)
        self.resume_checkpoint_path = getattr(self.config.trainer, "resume_path", None)

        # build tokenizer first
        if self.resume_training and self.resume_checkpoint_path:
            # If resuming from specific checkpoint, use that path for tokenizer
            local_model_path = copy_local_path_from_hdfs(
                src=self.resume_checkpoint_path, verbose=True
            )
        else:
            local_model_path = copy_local_path_from_hdfs(
                src=self.config.model.partial_pretrain, verbose=True
            )

        from verl.utils import hf_tokenizer
        self.tokenizer = hf_tokenizer(
            local_model_path, trust_remote_code=self.config.model.trust_remote_code
        )
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(
            self.config, "ulysses_sequence_parallel_size", 1
        )
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(
                f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}"
            )
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader()
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)

    def _normalize_config_bsz(self):
        dp_size = (
            self.device_mesh.size(0)
            if not self.ulysses_device_mesh
            else self.ulysses_device_mesh.size(0)
        )
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert (
            self.config.data.train_batch_size % dp_size == 0
        ), f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size

        assert (
            self.config.data.train_batch_size
            % self.config.data.micro_batch_size_per_gpu
            == 0
        )

    def _build_dataloader(self):
        config = self.config
        # build dataset
        self.train_dataset = SFTExpandDataset(
            parquet_files=config.data.train_files,
            tokenizer=self.tokenizer,
            prompt_key=config.data.prompt_key,
            response_key=config.data.response_key,
            max_length=config.data.max_length,
            truncation=config.data.truncation,
            middle_strategy=config.data.middle_strategy,
            middle_line_num=config.data.middle_line_num,
            merge_prob=config.data.merge_prob,
            max_delete=config.data.max_delete,
            merge_schedule=config.data.merge_schedule,
            use_uniform_merge_prob=config.data.use_uniform_merge_prob
        )
        self.val_dataset = SFTExpandDataset(
            parquet_files=config.data.val_files,
            tokenizer=self.tokenizer,
            prompt_key=config.data.prompt_key,
            response_key=config.data.response_key,
            max_length=config.data.max_length,
            truncation=config.data.truncation,
            merge_prob=config.data.merge_prob,
            merge_schedule=config.data.merge_schedule,
            max_delete=config.data.max_delete,
            use_uniform_merge_prob=config.data.use_uniform_merge_prob
        )

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(
                    f"Using SP rank {rank} and size {world_size} for data distribution"
                )
                print(
                    f"Each SP rank gets different data, but the same data WITHIN the same rank"
                )
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _build_model_optimizer(self, checkpoint_path=None):
        """Build model and optimizer, optionally from a checkpoint."""
        # Determine which path to load from
        if checkpoint_path:
            local_model_path = checkpoint_path
        else:
            local_model_path = copy_local_path_from_hdfs(
                src=self.config.model.partial_pretrain, verbose=True
            )

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(
            local_model_path, trust_remote_code=trust_remote_code
        )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert (
                self.use_remove_padding
            ), "Sequence parallel is only supported when remove_padding is enabled"
            from verl.models.registry import check_model_support_rmpad

            check_model_support_rmpad(config.model_type)

        if self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch

            apply_monkey_patch(config, verbose=True)

        # This may be very large
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings
        )

        with init_context():
            self.model: PreTrainedModel = AutoModel.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import (
                    _apply_liger_kernel_to_instance,
                )

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(
                        self.config.model.target_modules
                    ),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(
                offload_params=self.config.model.fsdp_config.offload_params
            )

        self.fsdp_model = FSDP(
            module=self.model,
            auto_wrap_policy=auto_wrap_policy,
            param_init_fn=init_fn,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            sync_module_states=True,
            device_id=torch.cuda.current_device(),
            cpu_offload=cpu_offload,
            use_orig_params=False,
        )

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.total_steps,
        )

    def _load_from_checkpoint(self, checkpoint_path):
        """Initialize training state from checkpoint."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        if self.device_mesh.get_rank() == 0:
            print(f"Resuming from checkpoint: {checkpoint_path}")

        # Load training state
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        optimizer_state_path = os.path.join(checkpoint_path, "optimizer_state.pt")

        # Only rank 0 loads the full state initially
        if self.device_mesh.get_rank() == 0 and os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            epoch = training_state["epoch"]
            global_step = training_state["global_step"]

            # Load scheduler state
            self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])
        else:
            # For other ranks or if file is missing, get step from path
            epoch = 0
            global_step = extract_step(checkpoint_path) or 0

        # Broadcast values to all ranks
        if torch.distributed.get_world_size() > 1:
            tensor = torch.tensor([epoch, global_step], device="cuda")
            torch.distributed.broadcast(tensor, src=0)
            if self.device_mesh.get_rank() != 0:
                epoch, global_step = tensor.tolist()

        # Load optimizer state if exists
        if os.path.exists(optimizer_state_path):
            from torch.distributed.fsdp import FullStateDictConfig
            from torch.distributed.fsdp.api import FullOptimStateDictConfig

            with FSDP.state_dict_type(
                self.fsdp_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                # Load optimizer state - rank 0 loads, others receive broadcast
                if self.device_mesh.get_rank() == 0:
                    try:
                        optim_state = torch.load(optimizer_state_path)
                        
                        # Check if optimizer state was saved in chunks
                        if isinstance(optim_state, dict) and optim_state.get("__chunked__", False):
                            print("Loading chunked optimizer state...")
                            num_chunks = optim_state["__num_chunks__"]
                            complete_state = {}
                            
                            # Load all chunks
                            for i in range(num_chunks):
                                chunk_path = os.path.join(os.path.dirname(optimizer_state_path), 
                                                        f"optimizer_state_chunk_{i}.pt")
                                if os.path.exists(chunk_path):
                                    chunk = torch.load(chunk_path)
                                    complete_state.update(chunk)
                            optim_state = complete_state
                    except Exception as e:
                        print(f"Warning: Failed to load optimizer state: {e}")
                        optim_state = None
                else:
                    optim_state = None

                try:
                    if optim_state is not None:
                        # Use FSDP utility to load optimizer state
                        optim_state_dict = FSDP.scatter_full_optim_state_dict(
                            optim_state, self.fsdp_model
                        )
                        self.optimizer.load_state_dict(optim_state_dict)
                    else:
                        print("Warning: Starting with fresh optimizer state")
                except Exception as e:
                    print(f"Warning: Failed to load optimizer state: {e}")
                    print("Continuing with fresh optimizer state")

        self.current_epoch = epoch
        return global_step

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = (
            self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        )

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        attention_mask = batch["attention_mask"].cuda().bool()
        position_ids = batch["position_ids"].cuda()
        t = batch['t'].cuda()
        loss_mask = batch.pop("loss_mask").cuda().bool()
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        
        # 统计mask tokens数量（loss_mask中1的数量）
        ma_num_batch = torch.sum(loss_mask).item()
        
        # 统计expand tokens数量（labels中expand_id的出现次数）
        # 使用数据集中预设的expand_token_id
        expand_id = self.tokenizer.expand_token_id  # 固定值 151667
        expand_id_tensor = torch.tensor(expand_id, device=labels.device, dtype=labels.dtype)
        ex_num_batch = torch.sum(labels.eq(expand_id_tensor)).item()
        
        # 更新全局计数器
        if self.device_mesh.get_rank() == 0:  # 只在主进程更新
            self.__class__.ma_num += ma_num_batch
            self.__class__.ex_num += ex_num_batch
            self.__class__.total_batches += 1
            
            # 定期打印统计信息
            if self.__class__.total_batches % 100 == 0:  # 每100个批次打印一次
                print(f"\nStatistics after {self.__class__.total_batches} batches:")
                print(f"Total expand tokens: {self.__class__.ex_num}")
                print(f"Total mask tokens: {self.__class__.ma_num}")
                print(f"Average expand tokens per batch: {self.__class__.ex_num / self.__class__.total_batches:.2f}")
                print(f"Average mask tokens per batch: {self.__class__.ma_num / self.__class__.total_batches:.2f}\n")

        

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if not use_sp:
                    # Standard forward pass without sequence parallel
                    #labels = input_ids.contiguous()

                    if attention_mask.dim() == 2:
                    # Input is (B, S) -> need to create pairwise mask (B, S, S)
                        attention_mask = torch.logical_and(
                            attention_mask.unsqueeze(1).unsqueeze(-2),  # (B, 1, S, 1)
                            attention_mask.unsqueeze(1).unsqueeze(-1)   # (B, 1, S, 1)
                        )  # Result: (B, 1, S, S)

                    elif attention_mask.dim() == 3:
                    # Already (B, S, S), just add head dimension
                        attention_mask = attention_mask.unsqueeze(1)  # (B, 1, S, S)
                    else:
                        raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

                    # Forward pass
                    # NOTE: loss_mask is of size (batch_size, seq_len - 1)
                    batch_size = input_ids.shape[0]
                    #masked_input_ids, t, loss_mask_nonflatten = q_sample(
                    #    input_ids,
                    #    maskable_mask=loss_mask,
                    #    mask_token_id=self.tokenizer.mask_token_id,
                    #)
                    loss_mask = loss_mask.reshape(-1)

                    output = self.fsdp_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                    )
                    logits = output.logits

                    shift_logits = torch.cat(
                        [logits[:, 0:1], logits[:, :-1]], dim=1
                    ).contiguous()
                    shift_labels = labels.contiguous()
                    # Flatten the tokens
                    shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)

                    # We use weighted loss
                    loss_mask = loss_mask.to(loss.device)
                    loss = loss.masked_fill(~loss_mask, 0)
                    if self.config.diffusion.token_reweighting:
                        loss = (
                            self.config.diffusion.alpha
                            * (1 - torch.exp(-loss)) ** self.config.diffusion.gamma
                            * loss
                        )

                    if self.config.diffusion.time_reweighting == "original":
                        raise NotImplementedError
                        weight = 1 / t[:, None].float().expand(labels.size())
                    elif self.config.diffusion.time_reweighting == "linear":
                        weight = 1 - t.float().expand(labels.size())
                    else:
                        raise NotImplementedError
                        weight = t.new_ones((batch_size, 1)).float().expand(labels.size())

                    loss = loss * weight.reshape(-1)
                else:
                    # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                    # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                    # 1. All SP ranks will receive the *SAME* batch
                    # 2. Different SP groups will receive *DIFFERENT* batches
                    # This is implemented by the DistributedSampler
                    raise NotImplementedError(
                        "Sequence parallel is not implemented yet"
                    )

                if self.config.diffusion.weight_eos and self.config.data.max_delete > 0:
                    non_eos_mask = (shift_labels != self.tokenizer.eos_token_id) & loss_mask
                    non_eos_loss = loss.clone()  
                    non_eos_loss[~non_eos_mask] = 0  
                    non_eos_count = non_eos_mask.sum().item() 
                    non_eos_loss = non_eos_loss.sum()  

                   
                    eos_mask = (shift_labels == self.tokenizer.eos_token_id) & loss_mask
                    eos_loss = loss.clone()  
                    eos_loss[~eos_mask] = 0  
                    eos_count = eos_mask.sum().item()  
                    eos_loss = eos_loss.sum() / eos_count  

                    
                    loss = (non_eos_loss + eos_loss) / (non_eos_count + 1)  
                else:
                    valid_token_this_rank = torch.sum(loss_mask)

                    if self.config.data.balance_dp_token:
                        torch.distributed.all_reduce(valid_token_this_rank)
                        dp_size = (
                            self.ulysses_device_mesh.size("dp")
                            if use_sp
                            else torch.distributed.get_world_size()
                        )
                    else:
                        dp_size = 1

                    loss = torch.sum(loss) / valid_token_this_rank * dp_size

                if do_backward:
                    loss.backward()
                return loss, ex_num_batch, ma_num_batch

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        expand_token_count = 0
        mask_token_count = 0
        for micro_batch in micro_batches:
            loss, ex_count, mask_count = self._compute_loss_and_backward(
                batch=micro_batch, do_backward=False
            )
            loss = loss / n_micro_batches
            loss.backward()
            step_loss += loss.item()
            expand_token_count += ex_count
            mask_token_count += mask_count

        grad_norm = self.fsdp_model.clip_grad_norm_(
            max_norm=self.config.optim.clip_grad
        )

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss).cuda()
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        counts_tensor = torch.tensor(
            [expand_token_count, mask_token_count], device="cuda", dtype=torch.float64
        )
        torch.distributed.all_reduce(counts_tensor, op=torch.distributed.ReduceOp.SUM)
        expand_token_count = counts_tensor[0].item()
        mask_token_count = counts_tensor[1].item()
        expand_ratio = 0.0
        if mask_token_count > 0:
            expand_ratio = expand_token_count / mask_token_count * 100.0
        return {
            "train/loss": step_loss.detach().item(),
            "train/lr(1e-3)": lr * 1e3,
            "train/grad_norm": grad_norm,
            "train/expand_ratio(%)": expand_ratio,
        }

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss, _, _ = self._compute_loss_and_backward(batch, do_backward=False)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss

    def _update_early_stopping(self, val_loss_value: float):
        improved = val_loss_value < (self.best_val_loss - self.min_delta)
        if improved:
            self.best_val_loss = val_loss_value
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        patience = self.early_stopping_patience
        should_stop = False
        if patience is not None and patience > 0:
            should_stop = self.no_improvement_count >= patience
        return improved, should_stop

    def _run_validation(self, global_step: int, tracking=None):
        rank = self.device_mesh.get_rank()
        val_losses = []
        for val_data in self.val_dataloader:
            val_data = TensorDict(
                val_data, batch_size=self.config.data.micro_batch_size_per_gpu
            ).cuda()
            loss = self.validation_step(val_data)
            val_losses.append(loss)

        val_loss = torch.mean(torch.stack(val_losses))
        val_loss_value = val_loss.detach().item()

        improved = False
        should_stop = False
        if rank == 0:
            metric = {"val/loss": val_loss_value}
            if tracking is not None:
                tracking.log(data=metric, step=global_step)
            improved, should_stop = self._update_early_stopping(val_loss_value)

        # Broadcast validation outcome and synchronize internal state
        state_tensor = torch.tensor(
            [
                val_loss_value if rank == 0 else 0.0,
                self.best_val_loss if rank == 0 else 0.0,
                float(self.no_improvement_count if rank == 0 else 0.0),
                1.0 if improved else 0.0,
                1.0 if should_stop else 0.0,
            ],
            device="cuda",
            dtype=torch.float64,
        )
        torch.distributed.broadcast(state_tensor, src=0)
        if rank != 0:
            val_loss_value = state_tensor[0].item()
            self.best_val_loss = state_tensor[1].item()
            self.no_improvement_count = int(state_tensor[2].item())
            improved = bool(state_tensor[3].item())
            should_stop = bool(state_tensor[4].item())

        if improved:
            self.save_checkpoint(step=global_step, is_best=True)
        torch.distributed.barrier()

        return improved, should_stop, val_loss_value

    def save_checkpoint(self, step, is_best=False):
        """Save model, optimizer, and training state."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp.api import FullOptimStateDictConfig
        import shutil
        import os

        # 设置新检查点路径
        path = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{step}"
        )
        
        # 在主进程中执行检查点清理
        if self.device_mesh.get_rank() == 0:
            try:
                # 1. 删除已存在的相同步数检查点
                if os.path.exists(path):
                    try:
                        shutil.rmtree(path)
                        print(f"Removed existing checkpoint at step {step}")
                    except Exception as e:
                        print(f"Warning: Failed to remove existing checkpoint: {e}")
                
                # 2. 确保本地目录存在
                os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
                
                # 3. 获取所有检查点并排序
                checkpoints = [d for d in os.listdir(self.config.trainer.default_local_dir)
                            if os.path.isdir(os.path.join(self.config.trainer.default_local_dir, d))
                            and d.startswith("global_step_")]
                checkpoints.sort(key=lambda x: extract_step(x) or 0, reverse=True)
                
                # 4. 删除多余的检查点
                max_keep = getattr(self.config.trainer, "max_checkpoints_to_keep", 3)
                if max_keep > 0:
                    # 获取所有检查点
                    checkpoints = [d for d in os.listdir(self.config.trainer.default_local_dir)
                                if os.path.isdir(os.path.join(self.config.trainer.default_local_dir, d))
                                and d.startswith("global_step_")]
                    
                    # 按步数排序（最新的在前）
                    checkpoints.sort(key=lambda x: extract_step(x) or 0, reverse=True)
                    
                    # 找出需要保留的检查点
                    keep_checkpoints = set()
                    
                    # 1. 保留最新的N个
                    keep_checkpoints.update(checkpoints[:max_keep])
                    
                    # 2. 保留最佳检查点（如果存在）
                    if self.best_checkpoint_path:
                        keep_checkpoints.add(os.path.basename(self.best_checkpoint_path))
                    
                    # 删除其他检查点
                    for ckpt in checkpoints:
                        if ckpt not in keep_checkpoints:
                            ckpt_path = os.path.join(self.config.trainer.default_local_dir, ckpt)
                            try:
                                shutil.rmtree(ckpt_path)
                                print(f"Cleaned up old checkpoint: {ckpt}")
                                
                                # 同时清理HDFS上的副本
                                if self.config.trainer.default_hdfs_dir:
                                    remote_path = os.path.join(self.config.trainer.default_hdfs_dir, ckpt)
                                    if hdfs_io.exists(remote_path):
                                        hdfs_io.rmtree(remote_path)
                            except Exception as e:
                                print(f"Warning: Failed to delete checkpoint {ckpt}: {e}")
            except Exception as e:
                print(f"Warning: Checkpoint cleanup failed: {e}")
        
        # 在保存新的checkpoint之前，如果是最佳checkpoint，先删除旧的
        if is_best and self.device_mesh.get_rank() == 0 and self.best_checkpoint_path:
            try:
                # 删除本地旧checkpoint
                if os.path.exists(self.best_checkpoint_path):
                    try:
                        shutil.rmtree(self.best_checkpoint_path)
                    except Exception as e:
                        print(f"Warning: Failed to delete local checkpoint: {e}")
                
                # 删除HDFS上的旧checkpoint
                if self.config.trainer.default_hdfs_dir:
                    try:
                        prev_name = os.path.basename(self.best_checkpoint_path)
                        remote_prev = os.path.join(self.config.trainer.default_hdfs_dir, prev_name)
                        if hdfs_io.exists(remote_prev):
                            hdfs_io.rmtree(remote_prev)
                    except Exception as e:
                        print(f"Warning: Failed to delete HDFS checkpoint: {e}")
            except Exception as e:
                print(f"Warning: Failed to delete previous checkpoint: {e}")

        # Save model state with safety checks
        model_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        try:
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, model_cfg):
                model_state = self.fsdp_model.state_dict()
                
                # Get optimizer state with extra safety checks
                try:
                    optim_state = FSDP.full_optim_state_dict(self.fsdp_model, self.optimizer)
                    
                    # Check if optimizer state is too large (>2GB)
                    import sys
                    if sys.getsizeof(optim_state) > 2 * 1024 * 1024 * 1024:  # 2GB
                        print("Warning: Optimizer state is very large. Attempting chunked save...")
                        # Split optimizer state into chunks
                        from itertools import islice
                        def chunk_dict(data, size=1000):
                            it = iter(data.items())
                            for i in range(0, len(data), size):
                                yield dict(islice(it, size))
                        
                        chunks = list(chunk_dict(optim_state))
                        optim_state = {"__chunked__": True, "__num_chunks__": len(chunks)}
                        
                        # Save chunks separately
                        for i, chunk in enumerate(chunks):
                            chunk_path = os.path.join(path, f"optimizer_state_chunk_{i}.pt")
                            torch.save(chunk, chunk_path)
                except Exception as e:
                    print(f"Warning: Failed to save optimizer state: {e}")
                    print("Continuing without optimizer state...")
                    optim_state = {}  # Empty dict as fallback
        except Exception as e:
            print(f"Error getting model/optimizer state: {e}")
            if self.device_mesh.get_rank() == 0:
                print("Attempting emergency model save...")
                try:
                    # Try direct save without FSDP state dict
                    self.model.save_pretrained(path)
                    model_state = None  # Signal to skip normal save
                except Exception as e2:
                    print(f"Emergency save failed: {e2}")
                    raise  # Re-raise if both attempts failed

        # Save training state (protected)
        training_state = {
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step": step,
            "epoch": self.current_epoch,
            "optimizer_chunked": getattr(optim_state, "__chunked__", False)
        }

        # Save on rank 0 only
        if self.device_mesh.get_rank() == 0:
            try:
                # 1. 创建新的checkpoint目录
                os.makedirs(path, exist_ok=True)

                # 2. 保存模型
                # Some custom GenerationConfig implementations (e.g. DreamGenerationConfig)
                # may define `validate()` without accepting the `strict` kwarg which
                # `transformers` passes when calling `generation_config.save_pretrained()`.
                # That causes a TypeError. To be robust, temporarily wrap the
                # generation_config.validate to accept/ignore `strict` during save.
                gen_cfg = getattr(self.model, "generation_config", None)
                _saved_validate = None
                if gen_cfg is not None and hasattr(gen_cfg, "validate"):
                    try:
                        import inspect

                        sig = inspect.signature(gen_cfg.validate)
                        if "strict" not in sig.parameters:
                            _saved_validate = gen_cfg.validate

                            def _validate_wrapper(*args, **kwargs):
                                # filter kwargs to those accepted by the original
                                filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
                                return _saved_validate(*args, **filtered)

                            gen_cfg.validate = _validate_wrapper
                    except Exception:
                        # If introspection fails for any reason, ignore and proceed
                        _saved_validate = None

                try:
                    # Before saving, remove any non-serializable attributes from generation_config
                    if hasattr(self.model, "generation_config"):
                        gen_config_dict = self.model.generation_config.to_dict()
                        # Filter out function attributes
                        gen_config_dict = {k: v for k, v in gen_config_dict.items() 
                                         if not callable(v)}
                        # Save filtered config
                        gen_config_path = os.path.join(path, "generation_config.json")
                        with open(gen_config_path, "w") as f:
                            json.dump(gen_config_dict, f, indent=2)
                    
                    # Save model with a proxy generation_config to avoid non-serializable attrs
                    gen_config_backup = None
                    proxy_assigned = False
                    if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
                        gen_config_backup = self.model.generation_config

                        class _GenCfgProxy:
                            def __init__(self, orig):
                                # take a filtered snapshot of serializable items
                                try:
                                    cfg = orig.to_dict()
                                except Exception:
                                    # fallback: use __dict__ shallow copy
                                    cfg = getattr(orig, "__dict__", {})
                                # filter out callables
                                self._cfg = {k: v for k, v in cfg.items() if not callable(v)}

                            def to_dict(self):
                                return dict(self._cfg)

                            def save_pretrained(self, save_directory):
                                # write the filtered dict to generation_config.json
                                os.makedirs(save_directory, exist_ok=True)
                                out = os.path.join(save_directory, "generation_config.json")
                                with open(out, "w") as f:
                                    json.dump(self._cfg, f, indent=2)

                            def __getattr__(self, name):
                                # provide graceful defaults for other accesses
                                if name == "to_dict":
                                    return self.to_dict
                                raise AttributeError(name)

                        # assign proxy so transformers will call proxy.save_pretrained
                        try:
                            self.model.generation_config = _GenCfgProxy(gen_config_backup)
                            proxy_assigned = True
                        except Exception:
                            # if assignment fails, keep original and let save_pretrained handle it
                            proxy_assigned = False

                    try:
                        self.model.save_pretrained(path, state_dict=model_state)
                    finally:
                        # Restore original generation_config
                        if gen_config_backup is not None and proxy_assigned:
                            try:
                                self.model.generation_config = gen_config_backup
                            except Exception:
                                pass
                finally:
                    # restore original validate if we wrapped it
                    if _saved_validate is not None and gen_cfg is not None:
                        try:
                            gen_cfg.validate = _saved_validate
                        except Exception:
                            pass
                self.tokenizer.save_pretrained(path)
                
                # Save optimizer state (possibly chunked)
                if isinstance(optim_state, dict) and optim_state.get("__chunked__", False):
                    # Save main optimizer state file with chunk info
                    torch.save(optim_state, os.path.join(path, "optimizer_state.pt"))
                else:
                    # Standard save for normal sized optimizer state
                    try:
                        torch.save(optim_state, os.path.join(path, "optimizer_state.pt"))
                    except Exception as e:
                        print(f"Warning: Failed to save optimizer state: {e}")
                        print("Saving empty optimizer state as fallback")
                        torch.save({}, os.path.join(path, "optimizer_state.pt"))
                
                # Save training state last (always try to save this)
                try:
                    torch.save(training_state, os.path.join(path, "training_state.pt"))
                except Exception as e:
                    print(f"Warning: Failed to save training state: {e}")
                    # Save minimal training state as fallback
                    minimal_state = {"global_step": step, "epoch": self.current_epoch}
                    torch.save(minimal_state, os.path.join(path, "training_state.pt"))

                # 3. 清理HDFS上的旧检查点并复制新检查点
                if self.config.trainer.default_hdfs_dir:
                    try:
                        # 先删除HDFS上已存在的相同步数检查点
                        hdfs_path = os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{step}")
                        if hdfs_io.exists(hdfs_path):
                            hdfs_io.rmtree(hdfs_path)
                            print(f"Removed existing HDFS checkpoint at step {step}")
                        
                        # 复制新的检查点到HDFS
                        hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                        hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
                        print(f"Successfully copied checkpoint to HDFS at {hdfs_path}")
                        
                        # 清理HDFS上的旧检查点
                        if max_keep > 0:
                            hdfs_checkpoints = [d for d in hdfs_io.listdir(self.config.trainer.default_hdfs_dir)
                                             if d.startswith("global_step_")]
                            hdfs_checkpoints.sort(key=lambda x: extract_step(x) or 0, reverse=True)
                            
                            # 保留最新的max_keep个和最佳检查点
                            keep_checkpoints = set(hdfs_checkpoints[:max_keep])
                            if self.best_checkpoint_path:
                                keep_checkpoints.add(os.path.basename(self.best_checkpoint_path))
                            
                            # 删除其他检查点
                            for ckpt in hdfs_checkpoints:
                                if ckpt not in keep_checkpoints:
                                    hdfs_ckpt_path = os.path.join(self.config.trainer.default_hdfs_dir, ckpt)
                                    try:
                                        hdfs_io.rmtree(hdfs_ckpt_path)
                                        print(f"Cleaned up old HDFS checkpoint: {ckpt}")
                                    except Exception as e:
                                        print(f"Warning: Failed to delete HDFS checkpoint {ckpt}: {e}")
                    except Exception as e:
                        print(f"Warning: HDFS operations failed: {e}")
                
                # 4. 保存成功后更新best_checkpoint_path
                if is_best:
                    old_best = self.best_checkpoint_path
                    self.best_checkpoint_path = path
                    
                    # 如果有旧的最佳检查点且不是当前检查点，删除它
                    if old_best and old_best != path:
                        try:
                            if os.path.exists(old_best):
                                shutil.rmtree(old_best)
                                print(f"Removed old best checkpoint: {old_best}")
                            
                            # 同时删除HDFS上的旧最佳检查点
                            if self.config.trainer.default_hdfs_dir:
                                old_best_hdfs = os.path.join(self.config.trainer.default_hdfs_dir,
                                                           os.path.basename(old_best))
                                if hdfs_io.exists(old_best_hdfs):
                                    hdfs_io.rmtree(old_best_hdfs)
                                    print(f"Removed old best checkpoint from HDFS: {old_best_hdfs}")
                        except Exception as e:
                            print(f"Warning: Failed to remove old best checkpoint: {e}")

                    # 仅保留当前最佳checkpoint
                    if self.device_mesh.get_rank() == 0:
                        checkpoint_dir = self.config.trainer.default_local_dir
                        try:
                            for ckpt in os.listdir(checkpoint_dir):
                                ckpt_path = os.path.join(checkpoint_dir, ckpt)
                                if (
                                    ckpt_path != path
                                    and ckpt.startswith("global_step_")
                                    and os.path.isdir(ckpt_path)
                                ):
                                    shutil.rmtree(ckpt_path)
                                    print(f"Removed stale checkpoint: {ckpt_path}")
                        except Exception as e:
                            print(f"Warning: Failed to clean extra checkpoints: {e}")

                        if self.config.trainer.default_hdfs_dir:
                            try:
                                hdfs_items = [
                                    d
                                    for d in hdfs_io.listdir(
                                        self.config.trainer.default_hdfs_dir
                                    )
                                    if d.startswith("global_step_")
                                ]
                                for ckpt in hdfs_items:
                                    if ckpt != os.path.basename(path):
                                        hdfs_io.rmtree(
                                            os.path.join(
                                                self.config.trainer.default_hdfs_dir,
                                                ckpt,
                                            )
                                        )
                                        print(
                                            f"Removed stale HDFS checkpoint: {ckpt}"
                                        )
                            except Exception as e:
                                print(f"Warning: Failed to clean HDFS checkpoints: {e}")

            except Exception as e:
                print(f"Error in save_checkpoint: {e}")
                # 保存失败时，删除部分完成的新checkpoint
                if os.path.exists(path):
                    try:
                        shutil.rmtree(path)
                    except Exception:
                        pass
                raise
        torch.distributed.barrier()

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in checkpoint directories."""
        latest_checkpoint = None
        latest_step = -1

        # Check local directory first
        local_dir = self.config.trainer.default_local_dir
        if os.path.exists(local_dir):
            checkpoints = [d for d in os.listdir(local_dir)
                        if os.path.isdir(os.path.join(local_dir, d)) and
                        d.startswith("global_step_")]

            for ckpt in checkpoints:
                step = extract_step(ckpt)
                if step is not None and step > latest_step:
                    latest_step = step
                    latest_checkpoint = os.path.join(local_dir, ckpt)

        # If not found locally and HDFS is configured, check there
        if latest_checkpoint is None and self.config.trainer.default_hdfs_dir:
            try:
                if hdfs_io.exists(self.config.trainer.default_hdfs_dir):
                    checkpoints = [
                        d for d in hdfs_io.listdir(self.config.trainer.default_hdfs_dir)
                        if d.startswith("global_step_")
                    ]
                    for ckpt in checkpoints:
                        step = extract_step(ckpt)
                        if step is not None and step > latest_step:
                            latest_step = step
                            remote_path = os.path.join(self.config.trainer.default_hdfs_dir, ckpt)

                            # Copy from HDFS to local
                            local_path = os.path.join(local_dir, ckpt)
                            os.makedirs(local_dir, exist_ok=True)
                            hdfs_io.copy(src=remote_path, dst=local_path, dirs_exist_ok=True)
                            latest_checkpoint = local_path
            except Exception as e:
                if self.device_mesh.get_rank() == 0:
                    print(f"Error checking HDFS for checkpoints: {e}")

        return latest_checkpoint

    def fit(self):
        rank = self.device_mesh.get_rank()
        hist_path = self.loss_history_path
        # TODO: add a unified tracking
        tracking = None
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        global_step = 0

        # Handle resuming training
        if self.resume_training:
            # Find latest checkpoint if not specified
            if not self.resume_checkpoint_path:
                self.resume_checkpoint_path = self._find_latest_checkpoint()

            if self.resume_checkpoint_path:
                global_step = self._load_from_checkpoint(self.resume_checkpoint_path)
                if rank == 0:
                    print(f"Resumed training from step {global_step}, epoch {self.current_epoch}")
            elif rank == 0:
                print("No checkpoint found, starting training from scratch")

        # Compute total training steps
        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        if rank == 0:
            print(f"Total training steps: {self.total_training_steps}")

        # Begin training from the current epoch
        for epoch in range(self.current_epoch, self.config.trainer.total_epochs):
            self.current_epoch = epoch
            self.train_sampler.set_epoch(epoch=epoch)

            # Create a data iterator
            dataloader_iter = iter(self.train_dataloader)

            # If resuming mid-epoch, skip to the right position
            if epoch == self.current_epoch and global_step > 0 and self.resume_training:
                steps_in_epoch = global_step % self.steps_per_epoch
                if steps_in_epoch > 0:
                    if rank == 0:
                        print(f"Skipping {steps_in_epoch} steps to resume at the right position")
                    for _ in range(steps_in_epoch):
                        try:
                            next(dataloader_iter)
                        except StopIteration:
                            dataloader_iter = iter(self.train_dataloader)

            # Calculate remaining steps in this epoch
            remaining_steps = self.steps_per_epoch
            if epoch == self.current_epoch and global_step > 0 and self.resume_training:
                remaining_steps -= global_step % self.steps_per_epoch

            for data in tqdm(
                dataloader_iter,
                initial=self.steps_per_epoch - remaining_steps,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch+1}/{self.config.trainer.total_epochs}",
            ):
                data = TensorDict(
                    data, batch_size=self.config.data.train_batch_size
                ).cuda()
                metric = self.training_step(data)
                if rank == 0 and tracking is not None:
                    tracking.log(data=metric, step=global_step)
                global_step += 1
                self.current_step = global_step  # 更新当前步骤
                if rank == 0:
                    ratio = metric.get("train/expand_ratio(%)")
                    if ratio is not None:
                        print(
                            f"[Step {global_step}] expand tokens/mask tokens: {ratio:.2f}%"
                        )

                # for early exit validation
                if global_step >= self.total_training_steps:
                    _, should_stop, _ = self._run_validation(
                        global_step, tracking if rank == 0 else None
                    )
                    if rank == 0 and should_stop:
                        print(
                            f"Early stopping triggered: no improvement for {self.no_improvement_count} validations"
                        )
                    return

                if global_step % self.config.trainer.save_checkpoint_steps == 0:
                    _, should_stop, _ = self._run_validation(
                        global_step, tracking if rank == 0 else None
                    )
                    if rank == 0 and should_stop:
                        print(
                            f"Early stopping triggered: no improvement for {self.no_improvement_count} validations"
                        )
                        return
                    if should_stop:
                        return
                    
            # validation
            _, should_stop, _ = self._run_validation(
                global_step, tracking if rank == 0 else None
            )
            if rank == 0 and should_stop:
                print(
                    f"Early stopping triggered: no improvement for {self.no_improvement_count} validations"
                )
            if should_stop:
                return
            
            # 在训练结束时打印最终统计信息
            if self.device_mesh.get_rank() == 0:
                print("\n=== Final Statistics ===")
                print(f"Total batches processed: {self.__class__.total_batches}")
                print(f"Total expand tokens: {self.__class__.ex_num}")
                print(f"Total mask tokens: {self.__class__.ma_num}")
                print(f"Average expand tokens per batch: {self.__class__.ex_num / max(1, self.__class__.total_batches):.2f}")
                print(f"Average mask tokens per batch: {self.__class__.ma_num / max(1, self.__class__.total_batches):.2f}")
                print("=====================\n")


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(
        device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",)
    )
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    trainer = FSDPSFTTrainer(
        config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh
    )
    trainer.fit()


if __name__ == "__main__":
    main()
