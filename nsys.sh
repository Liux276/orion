LD_PRELOAD="$HOME/orion/src/cuda_capture/libinttemp.so" nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none \
-o rnet101_rnet50_high_batch --cudabacktrace=true --capture-range=cudaProfilerApi -f true -x true \
python benchmarking/launch_jobs.py --algo orion --config_file $HOME/orion/artifact_evaluation/inf_inf/rnet101_rnet50.json