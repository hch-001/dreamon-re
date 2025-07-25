import json
import re
from collections import Counter
from transformers import AutoModel, AutoTokenizer
from eval.generator import MDMGenerator, MDMGeneratorArgs
from omegaconf import OmegaConf, DictConfig, ListConfig
import torch
from typing import List, Optional, Tuple
import abc
import os
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import datasets


def get_rank_data(data: list, rank: int, world_size: int):
    """
    Splits data into 'world_size' chunks and returns the chunk for the given 'rank'.
    """
    per_rank = len(data) // world_size
    start = rank * per_rank
    end = start + per_rank
    if rank == world_size - 1:
        end = len(data)  # Ensure last GPU gets any remaining items
    return data[start:end]

class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, tokens, add_bos, add_eos):
        pass

    @abc.abstractmethod
    def decode(self, tokens):
        pass

    @abc.abstractmethod
    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Return the offsets of the tokens in the original text. Only used for evaluation."""
        pass
    
class HFTokenizerWrapper(Tokenizer):
    def __init__(self, hf_tokenizer: str) -> None:
        self.tokenizer = hf_tokenizer
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.mask_id = self.tokenizer.mask_token_id
        self.expand_id = 151667

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False):
        tokens = [self.bos_id] * add_bos + self.tokenizer.encode(s) + [self.eos_id] * add_eos
        return tokens

    def decode(self, tokens: List[int], **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)
    
    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Return the offsets of the tokens in the original text. Only used for evaluation."""
        pass
from typing import Type, TypeVar
T = TypeVar("T")
def dataclass_from_dict(cls: Type[T], data: dict, strict: bool = True) -> T:
    """
    Converts a dictionary to a dataclass instance, recursively for nested structures.
    from lingua/args.py
    """
    base = OmegaConf.structured(cls())
    OmegaConf.set_struct(base, strict)
    override = OmegaConf.create(data)
    return OmegaConf.to_object(OmegaConf.merge(base, override))

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def write_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')


def eval_infill(task, generator, tokenizer, prediction_path=None, mask_expansion = False, fix_middle_length = None):
    from human_eval_infilling.data import write_jsonl, read_problems
    from datasets import load_dataset
    if task == 'humaneval_infill':## https://github.com/openai/human-eval-infilling
        problems = read_problems(benchmark_name='single-line')
        prefixs = [problems[task_id]["prompt"] for task_id in problems]
        suffixs = [problems[task_id]["suffix"] for task_id in problems]
        ground_truth_middles = [problems[task_id]["canonical_solution"] for task_id in problems]
        task_ids = list(problems.keys())
    elif task == 'santacoder-fim': ## https://huggingface.co/datasets/bigcode/santacoder-fim-task
        fim_data = load_dataset("bigcode/santacoder-fim-task", split="train")
        fim_data = [d for d in fim_data if d["language"] == "py"]
        prefixs = [d["prompt"] + '\n' for d in fim_data]
        suffixs = [d["suffix"] for d in fim_data]
        print(prefixs[0])
        print(suffixs[0])
        ground_truth_middles = [d["canonical_solution"] for d in fim_data]
        task_ids = [i for i in range(len(fim_data))]

    # Shard dataset by rank
    world_size = int(os.getenv("WORLD_SIZE"))
    rank = int(os.getenv("RANK"))
    local_rank = int(os.getenv("LOCAL_RANK"))

    total = len(prefixs)
    per_rank = total // world_size
    start = rank * per_rank
    end = start + per_rank if rank < world_size - 1 else total

    prefixs_shard = prefixs[start:end]
    suffixs_shard = suffixs[start:end]
    ground_truth_middles_shard = ground_truth_middles[start:end]
    task_ids_shard = task_ids[start:end]


    if mask_expansion:
        generations = generator.infilling_with_expansion(prefixs_shard, suffixs_shard)
    else:
        if not fix_middle_length:## oracle setting
            middle_lens = [
                len(tokenizer.encode(gt, add_bos=False, add_eos=False))
                for gt in ground_truth_middles_shard
            ]
        else: ## fix length
            middle_lens = [fix_middle_length for _ in task_ids_shard]
        generations = generator.infilling(prefixs_shard, middle_lens, suffixs_shard)


    samples = [dict(
        task_id=task_id, 
        completion=pred, 
        ground_truth_middle=ground_truth_middle,
        prefix = prefix,
        suffix = suffix
    ) for task_id, pred, ground_truth_middle, prefix, suffix in zip(task_ids_shard, generations , ground_truth_middles_shard, prefixs_shard, suffixs_shard)]

    # Gather results on rank 0
    gathered_samples = [None] * world_size
    dist.all_gather_object(gathered_samples, samples)

    if local_rank == 0:
        merged_samples = []
        for s in gathered_samples:
            merged_samples.extend(s)
        write_jsonl(prediction_path, merged_samples)


def setup_ddp():
    """Initialize DDP using environment variables set by torchrun"""
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=3600))
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

def main(args):
    # Initialize DDP
    rank, local_rank, world_size = setup_ddp()
    
    # Only print on rank 0 to avoid clutter
    if rank == 0:
        print(f"Initialized process group: rank {rank}, local_rank {local_rank}, world_size {world_size}")

    # Load config
    config_file = 'eval/configs/eval_infill.yaml'
    cfg = OmegaConf.load(config_file)
    cfg_cli = OmegaConf.from_dotlist(args.dotlist)
    cfg = OmegaConf.merge(cfg, cfg_cli)
    
    gen_cfg = OmegaConf.to_container(cfg, resolve=True)
    gen_cfg = dataclass_from_dict(MDMGeneratorArgs, gen_cfg, strict=False)

    if os.path.exists(cfg.prediction_path) and not cfg.overwrite:
        print(f'result {cfg.prediction_path} already exists. skipping...')
        return
    # Load model and tokenizer
    model = AutoModel.from_pretrained(cfg.ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = model.to(local_rank)  # Move to proper device before DDP
    model = DDP(model, device_ids=[local_rank])

    tokenizer = AutoTokenizer.from_pretrained(cfg.ckpt, trust_remote_code=True)
    tokenizer = HFTokenizerWrapper(tokenizer)

    generator = MDMGenerator(gen_cfg, model, tokenizer)

    # Run evaluation
    if cfg.task == "humaneval_infill" or cfg.task == 'santacoder-fim':
        eval_infill(cfg.task, generator, tokenizer, cfg.prediction_path, cfg.mask_expansion, cfg.fix_middle_length)
    else:
        raise ValueError(f"Unknown task {cfg.task}")
    cleanup()

if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run inference with DDP")
    parser.add_argument("--dotlist", nargs="*", default=sys.argv[1:])
    args = parser.parse_args()

    main(args)

