#!/bin/bash

# models=("resnet152" "densenet121" "inception_v3" "vgg19" "bert")
models=("bert")
batch_sizes=(4 32)

for model_name in "${models[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        output_dir="./postprocessing/profile_test/models/${model_name}/batch_${batch_size}"
        mkdir -p "$output_dir"
        # 依次执行后续处理脚本
        echo "Processing profiling results for $model_name with batch size $batch_size"
        
        # 执行 process_ncu.py 脚本
        python ./postprocessing/process_ncu.py --results_dir "$output_dir"

        # 执行 get_num_blocks.py 脚本
        python ./postprocessing/get_num_blocks.py --results_dir "$output_dir"

        # 执行 joint_roofline_analysis.py 脚本
        python ./postprocessing/roofline_analysis.py --results_dir "$output_dir" --ai_threshold 26.15

        # 执行 generate_file_block.py 脚本
        model_type="vision"
        if [[ "$model_name" == "transformer" || "$model_name" == "bert" ]]; then
            model_type="$model_name"
        fi

        python ./postprocessing/generate_file.py --input_file_name "$output_dir/output_ncu_sms_roofline.csv" \
            --output_file_name "$output_dir/${model_name}_${batch_size}_fwd" \
            --model_type "$model_type"

        echo "Finished processing $model_name with batch size $batch_size"
    done
done
