#!/bin/bash
min_gen_lens=(4 8 16 32 64)
max_gen_lens=(64)
checkpoint_path=Dream-org/DreamOn-v0-7B
output_dir="results/$(basename "$(dirname "$checkpoint_path")")/$(basename "$checkpoint_path")"
mkdir -p ${output_dir}
for min_gen_len in "${min_gen_lens[@]}"; do
    for max_gen_len in "${max_gen_lens[@]}"; do
        complete_result_file="${output_dir}/${subtask}_dynamic_min${min_gen_len}_max${max_gen_len}_expand_santa.jsonl"

        echo "Evaluating: $checkpoint_path"
        echo "Result file: $complete_result_file"

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port 29595 \
            -m eval.evaluate --dotlist \
                max_prompt_len=2048 \
                min_gen_len=$min_gen_len \
                max_gen_len=$max_gen_len \
                batch_size=1 \
                steps=256 \
                pad_to_max_len=false \
                temperature=0.2 \
                mask_expansion=true \
                delete_eos_token=true \
                overwrite=true \
                alg=entropy \
                alg_temp=0.0 \
                top_p=0.9 \
                show_progress=false \
                ckpt=${checkpoint_path} \
                prediction_path=${complete_result_file} \
                task=santacoder-fim
    done
done
python eval/compute_em_santa.py --result_path ${output_dir}

