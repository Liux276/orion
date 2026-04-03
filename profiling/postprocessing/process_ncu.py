import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, required=True,
                        help='path to directory containing the profiling files')
args = parser.parse_args()

df = pd.read_csv(f'{args.results_dir}/output_ncu.csv', index_col=0)
kernels = []
metrics_to_get = ['Memory [%]','Duration','SM [%]','Block Size', 'Grid Size',  'Registers Per Thread', 'Static Shared Memory Per Block']
unique_kernel_names = set()

type_conversion = {
    'Memory [%]': float,
    'Duration': int,
    'SM [%]': float,
    'Block Size': int,
    'Grid Size': int,
    'Registers Per Thread': int,
    'Static Shared Memory Per Block': int
}

# for index, row in df.iterrows():
#     kernel = row['Kernel Name']
#     metric_name = row['Metric Name']

#     if metric_name == 'DRAM Frequency':
#         kernels.append([kernel])
#         unique_kernel_names.add(kernel)
#     elif metric_name in metrics_to_get:
#         kernels[-1].append(row['Metric Value'])

for index, row in df.iterrows():
    kernel = row['Kernel Name']
    metric_name = row['Metric Name']

    if metric_name == 'DRAM Frequency':
        kernels.append([kernel])
        unique_kernel_names.add(kernel)
    elif metric_name in metrics_to_get:
        # 确保当前 kernel 数据已经存在
        
        if len(kernels) == 0 or kernels[-1][0] != kernel:
            kernels.append([kernel])
        
        # 找到对应 metric 在 metrics_to_get 中的索引
        metric_index = metrics_to_get.index(metric_name)+1
        
        # 根据索引位置添加 Metric Value
        while len(kernels[-1]) <= metric_index:  # 确保 kernel 列表的长度足够
            kernels[-1].append(None)  
        if metric_name in type_conversion:
            value = type_conversion[metric_name](row['Metric Value'].replace(",",""))
        else:
            value=row['Metric Value'] 
        kernels[-1][metric_index] = value  # 将值添加到对应位

for x in unique_kernel_names:
    print(x)
    print("------------------------------------")


for kernel in kernels:

    num_threads = int(kernel[-3]) * int(kernel[-4])
    num_registers = num_threads * int(kernel[-2])
    kernel += [num_threads, num_registers]


print(len(kernels))
print(kernels[0])
labels = ['Kernel_Name', 'DRAM_Throughput(%)', 'Duration(ns)', 'Compute(SM)(%)',  'Block', 'Grid', 'Registers_Per_Thread', 'Static_shmem_per_block', 'Number_of_threads', 'Number_of_registers']



df_new = pd.DataFrame(kernels, columns=labels)
print(df_new)
df_new.to_csv(f'{args.results_dir}/output_ncu_processed.csv')
