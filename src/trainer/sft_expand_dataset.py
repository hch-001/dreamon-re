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

from typing import List, Union
import re
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer
import random

class SFTExpandDataset(Dataset):
    """
    This is an in-memory SFTDataset
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer,
        prompt_key="prompt",
        response_key="response",
        max_length=1024,
        truncation="error",
        middle_strategy = 'line',
        middle_line_num = None,
        merge_prob = 0.5,
        max_delete=64,
        merge_schedule = "dynamic_inverse",
        use_uniform_merge_prob = 0.4
    ):
        assert truncation in ["error", "left", "right"]
        self.truncation = truncation

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.tokenizer.expand_token_id = 151667

        self.prompt_key = prompt_key
        self.response_key = response_key

        self.max_length = max_length
        self.middle_strategy = middle_strategy
        self.middle_line_num = middle_line_num
        self.merge_prob = merge_prob
        self.max_delete = max_delete
        self.merge_schedule = merge_schedule
        self.use_uniform_merge_prob = use_uniform_merge_prob
        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(
                parquet_file, verbose=True
            )

    def _read_files_and_tokenize(self):

        def series_to_item(ls):
            import pandas, numpy

            while (
                isinstance(ls, (pandas.core.series.Series, numpy.ndarray))
                and len(ls) == 1
            ):
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        self.responses = self.responses.tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # apply chat template
        if not isinstance(prompt, str):
            prompt_chat = list(prompt)
        else:
            prompt_chat = [{"role": "user", "content": prompt}]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False
        )
        response_chat_str = response + tokenizer.eos_token

        # tokenize prompt
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt")
        prompt_ids = prompt_ids_output['input_ids'][0]
        prompt_length = prompt_ids.shape[0]

        if self.middle_strategy == 'random':
            response_ids_output = tokenizer(response_chat_str, return_tensors="pt")
            response_ids = response_ids_output['input_ids'][0]

            response_length = response_ids.shape[0]

            # Split response into prefix, middle, suffix
            total_length = len(response_ids)
            mid_start = random.randint(1, total_length - 2)
            mid_end = random.randint(mid_start + 1, total_length - 1)

            prefix_ids = response_ids[:mid_start]
            middle_ids = response_ids[mid_start:mid_end]
            suffix_ids = response_ids[mid_end:]
        elif self.middle_strategy == 'line':
            # Split response into prefix, middle, suffix — now using line-based selection
            prefix_str, code_block, suffix_str = self.extract_code_block(response)
            code_lines = code_block.split('\n')

            # Choose start and end indices for middle (consecutive lines)
            max_attempts = 5
            for _ in range(max_attempts):
                # Ensure valid random indices
                if not self.middle_line_num:
                    try:
                        middle_start = random.randint(1, len(code_lines) - 2)
                        middle_end = random.randint(middle_start + 1, len(code_lines) - 1)
                    except:
                        middle_start = 1
                        middle_end = len(code_lines)
                else:
                    try:
                        middle_start = random.randint(1, len(code_lines) - self.middle_line_num - 1)
                        middle_end = middle_start + self.middle_line_num
                    except:
                        middle_start = 1
                        middle_end = len(code_lines)

                # Extract the slice
                middle_lines = code_lines[middle_start:middle_end]
                middle_str = "\n".join(middle_lines) + '\n'

                # Check your required conditions
                if len(middle_str.split()) >= 3 and 'def' not in middle_str and len(middle_str) > 3:
                    break  # Exit loop early if valid one is found


            prefix_lines = code_lines[:middle_start]
            suffix_lines = code_lines[middle_end:]

            # Join the lines back into strings
            prefix_str = prefix_str + "\n".join(prefix_lines) + '\n'
            suffix_str = "\n".join(suffix_lines) + suffix_str
            
            prefix_ids = tokenizer(prefix_str, return_tensors='pt')['input_ids'][0]
            middle_ids = tokenizer(middle_str, return_tensors='pt')['input_ids'][0]
            suffix_ids = tokenizer(suffix_str, return_tensors='pt')['input_ids'][0]
            response_length = prefix_ids.shape[0] + middle_ids.shape[0] + suffix_ids.shape[0]
        else:
            raise ValueError



        # EOS token handling
        if self.max_length - prompt_length - response_length > 0 and self.max_delete > 0:
            eos_count = torch.randint(
                low=0,
                high=min(self.max_delete, self.max_length - prompt_length - response_length),
                size=(1,),
            ).item()
            eos_tensor = torch.tensor([tokenizer.eos_token_id] * eos_count, dtype=middle_ids.dtype)
        else:
            eos_count = 0
            eos_tensor = torch.tensor([], dtype=middle_ids.dtype)
        
        middle_ids = torch.cat([middle_ids, eos_tensor])
        
        masked_middle_ids, labels, middle_attention_mask, t = self.masking_merge_for_response(
            middle_ids, 
            tokenizer, 
            merge_prob=self.merge_prob, 
            merge_schedule=self.merge_schedule,
            use_uniform_merge_prob=self.use_uniform_merge_prob
        )
        # Concat all parts
        input_ids = torch.cat([
            prompt_ids,
            prefix_ids,
            masked_middle_ids,
            suffix_ids
        ], dim=-1)

        attention_mask = torch.cat([
            torch.ones_like(prompt_ids),
            torch.ones_like(prefix_ids),
            middle_attention_mask,
            torch.ones_like(suffix_ids)
        ], dim=-1)
        
        labels = torch.cat([
            prompt_ids,
            prefix_ids,
            labels,
            suffix_ids
        ], dim = -1)

        # Padding or Truncation
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            pad_len = self.max_length - sequence_length
            input_ids = torch.cat([input_ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=input_ids.dtype)])
            labels = torch.cat([labels, torch.full((pad_len,), tokenizer.pad_token_id, dtype = labels.dtype)])
            attention_mask = torch.cat([attention_mask, torch.ones(pad_len, dtype=attention_mask.dtype)])
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                labels = labels[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            elif self.truncation == "right":
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            else:
                raise ValueError(f"Unknown truncation strategy: {self.truncation}")

        #position_ids = compute_position_id_with_mask(input_ids != self.tokenizer.pad_token_id)
        position_ids = compute_position_id_with_mask(attention_mask)

        # Loss mask (only for merged part)
        loss_mask = (input_ids == tokenizer.mask_token_id) & (attention_mask == 1)

        return {
            "input_ids": input_ids.to(torch.long),
            "labels": labels.to(torch.long),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "t": t
        }
    def masking_merge_for_response(self, input_tokens, tokenizer, merge_prob=0.5, merge_schedule="dynamic_inverse", use_uniform_merge_prob=0.5):
        """
        The process is:
        1. Independently mask each token with probability sampling_ratio.
            If a token is masked it is replaced by "<mask>", otherwise it remains unchanged.
        2. Scan the masked sequence for adjacent "<mask>" tokens. Whenever found, with probability merge_prob:
                - Mark the first token's label as "<expand>" (indicating the head of a merged pair).
                - Modify the attention_mask so that the second token is not attended to (i.e. set its attention_mask to 0).
            Tokens that are not part of a merge or are not masked are labeled as "<nonexpand>".
        3. Compute position_ids such that effective tokens (attention_mask==1) receive sequential indices, 
            while merged-out tokens (attention_mask==0) receive a default position of 0.
        
        Parameters:
        input_tokens (torch.Tensor): The original sequence of tokens as a tensor.
        sampling_ratio (float): The independent probability a token is replaced with "<mask>".
        merge_prob (float): The probability that a pair of adjacent "<mask>" tokens are merged.
        
        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - final_tokens: The token tensor after independent masking (tokens remain as masked or original).
            - labels: A tensor (same length as tokens) with each token labeled as 1 ("<expand>") or 0 ("<nonexpand>").
            - attention_mask: A tensor of binary values (1 for effective tokens, 0 for merged tokens).
        """
        sampling_ratio = torch.rand(1)
        # Step 1: Sampling masks
        mask = torch.rand_like(input_tokens, dtype=torch.float) < sampling_ratio
        final_tokens = input_tokens.clone()
        final_tokens[mask] = tokenizer.mask_token_id

        eos_mask = input_tokens == tokenizer.eos_token_id
        final_tokens[eos_mask] = tokenizer.mask_token_id
        
        # Initialize labels and attention_mask
        labels = input_tokens.clone()
        attention_mask = torch.ones_like(input_tokens, dtype=torch.long)

        ## Step 2: Merge
        num_masked = mask.sum().item()

        if torch.rand(1).item() < use_uniform_merge_prob:
            merge_schedule = "static"
        if merge_schedule == "dynamic_inverse":
            dynamic_merge_prob = merge_prob * (1 - (num_masked / input_tokens.size(0))) 
        elif merge_schedule == "dynamic_proportional":
            dynamic_merge_prob = merge_prob * (num_masked / input_tokens.size(0))
        elif merge_schedule == "static":
            dynamic_merge_prob = merge_prob
        elif merge_schedule == "random":
            dynamic_merge_prob = torch.rand(1).item() * merge_prob
        elif merge_schedule == "full_random":
            # So we need to vary merge_prob to [0,1] to make the model more robust
            dynamic_merge_prob = torch.rand(1).clamp(0.0, 0.95)
        else:
            raise ValueError(f"Unknown merge schedule: {merge_schedule}")

        rand_values = torch.rand(len(final_tokens)-1)
        
        for i in range(len(final_tokens)-1):
            if input_tokens[i] == self.tokenizer.eos_token_id:
                break
            if (final_tokens[i] == tokenizer.mask_token_id and 
                final_tokens[i+1] == tokenizer.mask_token_id and ## adjacement MASK
                rand_values[i] < dynamic_merge_prob): ## merge
                labels[i] = tokenizer.expand_token_id
                attention_mask[i+1] = 0


        return final_tokens, labels, attention_mask, sampling_ratio

    def extract_code_block(self, response: str):
        """
        Extracts the content inside a markdown-style Python code block:
        
        Returns:
            prefix: everything before ```python
            code_block: content inside the code block
            suffix: everything after ```
        """
        # Split by ```python and ``` to extract the code block
        parts = re.split(r'```python\s*|\s*```', response)
        
        if len(parts) < 2:
            return "", response, ""  # No code block found
        
        prefix = parts[0]
        code_block = '```' + parts[1] + '\n```' if len(parts) > 1 else ""
        suffix = parts[2] if len(parts) > 2 else ""
        
        return prefix, code_block, suffix
