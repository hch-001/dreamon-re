#!/bin/bash
min_gen_lens=(4 8 16 32 64)
max_gen_lens=(64)
checkpoint_path=Dream-org/DreamOn-v0-7B

for min_gen_len in "${min_gen_lens[@]}"; do
    for max_gen_len in "${max_gen_lens[@]}"; do
        output_dir="results/$(basename "$(dirname "$checkpoint_path")")/$(basename "$checkpoint_path")"
        mkdir -p ${output_dir}
        complete_result_file="${output_dir}/humaneval_infill_dynamic_min${min_gen_len}_max${max_gen_len}_expand.jsonl"

        echo "Evaluating: $checkpoint_path"
        echo "Result file: $complete_result_file"

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port 29514 \
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
                task=humaneval_infill  && \
                evaluate_infilling_functional_correctness single-line ${complete_result_file}
    done
done

