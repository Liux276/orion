This is a simple example to check that Orion has been installed correctly and can run.

Please follow the instructions in [INSTALL](INSTALL.md) to start a container with our image.
Then start the Orion process (server and client) by running:
* `cd /data1/home/jiangjz/code/orion_lx/benchmarking`
* `LD_PRELOAD="/data1/home/jiangjz/code/orion_lx/src/cuda_capture/libinttemp.so" python launch_jobs.py /data1/home/jiangjz/code/orion_lx/artifact_evaluation/example/config.json 1 1 1`