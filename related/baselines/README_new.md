# GPU Sharing Baselines
支持的baselins:`MPS`, `Streams`, `Isolated`, and `Sequential`.
如果需要重跑，可以修改`run_baselines.py`文件中对应的yaml的规则（比如修改代码开头的数组），执行后会自动在对应目录下生成对应格式的的yaml文件，按照顺序执行文件，并将执行的输入输出，对应的.log.json文件都copy到./result/对应的模型组合/策略/分布/batch_size下，如`./result/bert_mobilenet_v2/Isolated/poisson/batch_4/eval-berteval-mobilenet_v2.log.json`

可以在终端`ps -ef | grep baselines`查看执行情况。一般batch_size为32的模型或者包含bert的模型组合执行会比较慢。

结束后，可以使用`python collect_baselines.py --policy all`一次性处理所有的结果并生成csv。

## Supported Baselines
### MPS 运行前
MPS: [Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/index.html) is a feature of NVIDIA GPUs that allows multiple processes to share a single GPU.

**Caveat!** There are extra steps to do before executing the python program:
1. Execute `./start_MPS_control_daemon.sh` to start the MPS server.
2. Export these two environment variables:
```shell
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
```
3. Within the same shell session where you exported the environment variables, execute the python program normally.

### TICK-TOCK scheduling

This directory contains a basic implementation of TICK-TOCK scheduling using Python threads, and torch.cuda streams and events.
It is based on the description provided in [WAVELET: EFFICIENT DNN TRAINING WITH TICK-TOCK SCHEDULING (MLSys'21)](https://proceedings.mlsys.org/paper/2021/file/c81e728d9d4c2f636f067f89cc14862c-Paper.pdf).

What would be an interesting next step is implementing the memory management support described in [Zico: Efficient GPU Memory Sharing for
Concurrent DNN Training (ATC'21)](https://www.usenix.org/system/files/atc21-lim.pdf).

### Streams
GPU Streams provide a way to execute workloads concurrently on a single GPU.
One stream captures a linear sequence of operations to be executed, and multiple streams can be executed concurrently.

### Sequential
`Sequential` represents the temporal sharing baseline where the GPU is time-sliced between the two workloads.

### Isolated
To analyze the overhead of GPU sharing, we compare the performance of GPU sharing with the performance of executing 
the workload on a single GPU without sharing. For `Isolated` we first execute workload A and then workload B after A is finished.
