#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate orion
# models=("mobilenet_v2" "bert" "resnet50" "resnet101")
models=("bert")
batch_sizes=(4 32)

for model_name in "${models[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        echo "Profiling model: $model_name with batch size: $batch_size"

        output_dir="./postprocessing/profile_test/models/${model_name}/batch_${batch_size}"
        mkdir -p "$output_dir"

        # 运行并生成 CSV
        ncu --csv --set detailed --nvtx --nvtx-include "start/" \
            `which python3` script.py --profile ncu --model_name "$model_name" --batch_size "$batch_size" > "$output_dir/output_ncu_${model_name}_batch_${batch_size}.csv"

        # 运行并生成 .ncu-rep
        ncu -o "$output_dir/output_ncu_${model_name}_batch_${batch_size}" --set detailed --nvtx --nvtx-include "start/" \
            `which python3` script.py --profile ncu --model_name "$model_name" --batch_size "$batch_size"

        # # 复制 CSV 文件并去掉前三行
        # tail -n +4 "$output_dir/output_ncu_${model_name}_batch_${batch_size}.csv" > "$output_dir/output_ncu.csv"

        echo "Finished profiling $model_name with batch size $batch_size"
    done
done
