import time
import torch
from transformers import AutoModel, AutoTokenizer
import os
from transformers.utils import cached_file

def process_infilling_prompt(prefix, suffix, tokenizer, number_of_mask):
    prefix = [tokenizer.bos_token_id] + tokenizer.encode(prefix, add_special_tokens=False)
    middle = [tokenizer.mask_token_id] * number_of_mask
    suffix = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
    return prefix + middle + suffix



if __name__ == '__main__':
    prefix = '''import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''

    suffix = '''        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False
'''
    model_path = "Dream-org/DreamOn-v0-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    input_ids = process_infilling_prompt(prefix, suffix, tokenizer, 4)
    input_ids = torch.LongTensor([input_ids]).to("cuda")

    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = model.to("cuda").eval()

    # 输出模型的实际路径
    print(f"✅ 模型加载完成")
    print(f"📂 模型本地路径: {model.name_or_path}")
    print(f"📂 Tokenizer 本地路径: {tokenizer.name_or_path}")

    # 如果需要更详细的缓存路径
    if os.path.exists(model.name_or_path):
        print(f"📁 模型文件存放在: {os.path.abspath(model.name_or_path)}")
    else:
        print("⚠️ 模型路径未在本地文件系统中找到（可能是远程加载）")

    # 找出模型在磁盘的实际缓存路径
    config_path = cached_file(model_path, "config.json")
    real_model_dir = os.path.dirname(config_path)

    print(f"✅ 模型加载完成")
    print(f"📂 模型缓存目录: {real_model_dir}")
    
    output = model.diffusion_generate(
        input_ids,
        temperature=0.2,
        alg = 'entropy',
        alg_temp = 0,
        top_p = 0.9,
        max_new_tokens = 64,
        return_dict_in_generate = True,
        output_history = True,
        number_transfer_tokens = 1
    )


    history = output.history
    for i, h in enumerate(history):
        print(f"########################")
        time.sleep(0.2)
        print(tokenizer.decode(h.tolist()), end="\n\n")   