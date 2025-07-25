# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import List, Optional
import tqdm
import torch
from torch import nn
import torch.distributions as dists
from omegaconf import OmegaConf
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0

@dataclass
class MDMGeneratorArgs:
    temperature: float = 0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    show_progress: bool = False
    dtype: Optional[str] = "bf16"
    device: Optional[str] = "cuda"
    max_tokens: Optional[int] = 2048
    min_gen_len: Optional[int] = 16
    max_prompt_len: Optional[int] = 1024
    max_gen_len: int = 512
    pad_to_max_len: bool = True
    pad_eos_to_right: bool = True

    batch_size: int = 16
    # diffusion specific params
    eps: float = 1e-3
    steps: int = 512
    alg: str = 'origin' # [origin, maskgit_plus, topk_margin]
    alg_temp: Optional[float] = None # for maskgit_plus



class MDMGenerator:
    def __init__(
        self,
        cfg: MDMGeneratorArgs,
        model: nn.Module,
        tokenizer: AutoTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_id = self.tokenizer.mask_id
        self.pad_to_max_len = cfg.pad_to_max_len
        self.expand_id = self.tokenizer.expand_id
        self.max_tokens = cfg.max_tokens
        self.max_prompt_len = cfg.max_prompt_len
        self.max_gen_len = cfg.max_gen_len
        self.min_gen_len = cfg.min_gen_len
        self.temperature = cfg.temperature
        self.alg_temp = cfg.alg_temp
        self.top_p = cfg.top_p
        self.top_k = cfg.top_k
        self.steps = cfg.steps
        self.eps = cfg.eps
        self.alg = cfg.alg
        self.delete_eos_token = cfg.delete_eos_token
        self.pad_eos_to_right = cfg.pad_eos_to_right
        self.expand_budget = self.max_gen_len
        self.batch_size = cfg.batch_size

        self.device = cfg.device
        self.show_progress = cfg.show_progress
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[cfg.dtype]

    def batch_generate(self, x):
        # print(x.shape)
        timesteps = torch.linspace(1, self.eps, self.steps + 1, device=x.device)
        for i in range(self.steps):
            mask_index = (x == self.mask_id)
            logits = self.model(x).logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            logits = logits[mask_index]
            logits[:,self.expand_id] -= 1e9
            t = timesteps[i]
            s = timesteps[i + 1]
            if torch.all(~mask_index):
                break

            if self.alg == 'origin':
                p_transfer = 1 - s / t if i < self.steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + self.mask_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s]= sample_tokens(logits[transfer_index_t_s], temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
                x[mask_index] = x0.clone()
            else:
                if self.alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(logits, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
                elif self.alg == 'topk_margin':
                    # print("use topk_margin")
                    confidence, x0 = sample_tokens(logits, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, margin_confidence=True)
                elif self.alg == 'entropy':
                    # print("use entropy")
                    confidence, x0 = sample_tokens(logits, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {self.alg}")

                number_transfer_tokens = 1
                if number_transfer_tokens > 0:
                    if self.alg_temp is None or self.alg_temp == 0 :
                        _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                    else:
                        # confidence = torch.gather(logits.softmax(-1), -1, x0.unsqueeze(-1)).squeeze(-1)
                        confidence = confidence / self.alg_temp
                        confidence = F.softmax(confidence, dim=-1)
                        transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
                    x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + self.mask_id
                    x0_[transfer_index] = x0[transfer_index].clone()
                    x[mask_index] = x0_

            if self.show_progress:
                print("=="*50 + f" step {i} " + "=="*50)
                print(self.tokenizer.decode(x[0].tolist()))
        return x


    @torch.inference_mode()
    @torch.no_grad()
    def infilling(self, prompts, middle_lens, suffixs):
        # Tokenize
        prompts = [self.tokenizer.encode(p, add_bos=True, add_eos=False) for p in prompts]
        prefix_lens = [len(p) for p in prompts]
        # add middle placeholder and suffix
        prompts = [
            p + [self.mask_id] * m + self.tokenizer.encode(s, add_bos=False, add_eos=True)
              for p,m,s in zip(prompts, middle_lens, suffixs)
        ]

        # Truncate
        prompts = [p[-self.max_prompt_len:] for p in prompts]

        generations = []
        num_batches = len(prompts) // self.batch_size if len(prompts) % self.batch_size == 0 else len(prompts) // self.batch_size + 1
        for i in tqdm.trange(num_batches):
            batch_prompts = prompts[i*self.batch_size: (i+1)*self.batch_size]
            batch_prefix_lens = prefix_lens[i*self.batch_size: (i+1)*self.batch_size]
            batch_middle_lens = middle_lens[i*self.batch_size: (i+1)*self.batch_size]
            # max seq len or the maximun prompt len + gen len
            max_seqlen = self.max_tokens if self.pad_to_max_len else min(self.max_tokens, max([len(p)+self.max_gen_len for p in batch_prompts]))
            padded_batch_prompts = torch.LongTensor([p+[self.tokenizer.eos_id]*(max_seqlen-len(p)) for p in batch_prompts]).to(self.device)
            batch_generations = self.batch_generate(padded_batch_prompts)
            generations.extend([self.tokenizer.decode(g[pl:pl+ml].tolist(), skip_special_tokens = True) for pl, ml, g in zip(batch_prefix_lens, batch_middle_lens, batch_generations)])
        return generations
    
    def batch_generate_with_expand_as_token(self, input_ids):
        max_tokens = min(self.max_tokens, input_ids.shape[1] + self.max_gen_len - self.min_gen_len)
        x = F.pad(input_ids, (0, max_tokens - input_ids.shape[1]), value = self.mask_id)
        num_generation_tokens = self.min_gen_len
        expand_budget = self.expand_budget
        for i in range(self.steps):
            cur_generation_window_length = input_ids.shape[1] - self.min_gen_len + num_generation_tokens
            attention_mask = torch.ones([input_ids.shape[0], cur_generation_window_length], dtype = torch.int16).to(input_ids.device)
            attention_mask = F.pad(attention_mask, (0, max_tokens - attention_mask.shape[1]), value = 0)

            mask_index = (x == self.mask_id) & (attention_mask == 1)
            if torch.all(~mask_index[:,:cur_generation_window_length]):
                break
            
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
            
            output = self.model(x, attention_mask, tok_idx)
            logits = output.logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            logits = logits[mask_index] 

            ## block the logit for expansion when token budget is all used
            if cur_generation_window_length == max_tokens or expand_budget == 0:
                logits[:,self.expand_id] -= 1e9
            
            if self.alg == 'origin':
                raise NotImplementedError
            else:
                if self.alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(logits, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
                elif self.alg == 'topk_margin':
                    confidence, x0 = sample_tokens(logits, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, margin_confidence=True)
                elif self.alg == 'entropy':
                    confidence, x0 = sample_tokens(logits, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {self.alg}")

                number_transfer_tokens = 1
                if number_transfer_tokens > 0:
                    if self.alg_temp is None or self.alg_temp == 0:
                        _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                    else:
                        confidence = confidence / self.alg_temp
                        confidence = F.softmax(confidence, dim=-1)
                        transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
                    x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + self.mask_id
                    x0_[transfer_index] = x0[transfer_index].clone()
                    x[mask_index] = x0_
                # Only process if batch size is 1
                if self.pad_eos_to_right:
                    if x.shape[0] != 1:
                        raise NotImplementedError
                    x_seq = x[0]  # Flatten to 1D: shape [seq_len]

                    # Find indices where EOS occurs
                    eos_indices = (x_seq == self.tokenizer.eos_id).nonzero(as_tuple=True)

                    if len(eos_indices[0]) > 0:
                        # Get the first occurrence of EOS
                        # mask indices
                        
                        first_eos_idx = eos_indices[0][0].item()
                        position_mask = torch.arange(x_seq.size(0), device=x.device) >= first_eos_idx
                        replace_mask = position_mask & mask_index[0]
                        # Set all tokens after EOS to eos_id
                        x_seq.masked_fill_(replace_mask, self.tokenizer.eos_id)

                #        # Reshape back to original shape (unsqueeze)
                        x = x_seq.unsqueeze(0)

                
                if self.show_progress:
                    print('='*10 + f'Step {i}' + '='*10)
                    print(self.tokenizer.decode(x[0,:cur_generation_window_length].tolist()))


                #  Expansion Step: Check for expand_id and replace with two mask tokens
                expand_indices = (x[0] == self.expand_id).nonzero(as_tuple=False).squeeze(1)
                if expand_indices.numel() > 0:
                    # Process from right to left to prevent shifting issues
                    for idx in sorted(expand_indices.tolist(), reverse=True):
                        x = torch.cat((
                            x[:, :idx],
                            torch.tensor([[self.mask_id, self.mask_id]], device=x.device),
                            x[:, idx + 1:]
                        ), dim=1)
                        num_generation_tokens += 1
                        expand_budget -= 1
                        # Truncate back to max_tokens if needed
                        if x.shape[1] > max_tokens:
                            x = x[:, :max_tokens]
                ## Detele EOS tokens from middle
                if self.delete_eos_token:
                    # Find indices where EOS occurs
                    eos_indices = ((x[0] == self.tokenizer.eos_id) & (mask_index[0] == 1)).nonzero(as_tuple=False).squeeze(1)
                    if len(eos_indices) > 0 and self.show_progress:
                        print('delete token')
                    for idx in sorted(eos_indices.tolist(), reverse=True):
                        x = torch.cat((
                            x[:, :idx],
                            x[:, idx + 1:],
                            torch.tensor([[self.mask_id]], device = x.device)
                        ), dim = 1)
                        num_generation_tokens -= 1

        return x, num_generation_tokens
            
    @torch.inference_mode
    @torch.no_grad()
    def infilling_with_expansion(self, prompts, suffixs):
        # Tokenize
        prompts = [self.tokenizer.encode(p, add_bos=True, add_eos=False) for p in prompts]
        prefix_lens = [len(p) for p in prompts]
        # add middle placeholder and suffix
        prompts = [
            p + [self.mask_id] * self.min_gen_len + self.tokenizer.encode(s, add_bos=False, add_eos=True)
              for p,s in zip(prompts, suffixs)
        ]

        # Truncate
        prompts = [p[-self.max_prompt_len:] for p in prompts]
        generations = []
        
        for i in tqdm.tqdm(range(len(prompts))):
            prompt = prompts[i]
            prefix_len = prefix_lens[i]
            prompt = torch.LongTensor([prompt]).to(self.device)

            response, response_length = self.batch_generate_with_expand_as_token(prompt)
            generations.append(self.tokenizer.decode(response[0,prefix_len:prefix_len + response_length].tolist(), skip_special_tokens = True))
        return generations
            
