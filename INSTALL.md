### 环境配置
- 使用conda虚拟环境
- 经过反复实验，需使用以下配置，否则可能导致无法编译或卡住：
    - cuda=11.3.1，由于作者依赖一个版本的pytorch并对源码做了修改，所以使用该版本
    - cudnn=8.x
    - gcc=9
    - gxx=9
- 假设工作目录为: `$HOME`
1. 创建conda环境后使用以下命令安装，注意一定要严格按照以下命令:
    ```
    conda install cuda-toolkit cudnn=8 -c nvidia/label/cuda-11.3.1
    conda install cudatoolkit=11.3 gcc=9 gxx=9
    ```
2. Install PyTorch from source:
    * `git clone --recursive https://github.com/pytorch/pytorch`
    * `cd pytorch`
    * `git reset --hard 67ece03c8cd632cce9523cd96efde6f2d1cc8121`
    * Apply a patch of changes for Orion: `git apply orion-torch-changes.patch`
    * `git submodule sync`
    * `git submodule update --init --recursive --jobs 0`
    * `python setup.py develop`
3. Install Torchvision from source:
    * `git clone https://github.com/pytorch/vision.git`
    * `cd vision`
    * `git reset --hard da3794e90c7cf69348f5446471926729c55f243e`
    * `python setup.py develop`
4. Download the Orion repo and install:
    * `git clone http://github.com/Liux276/orion.git`
    * `cd orion`
    * `bash compile.sh`
    * `pip install -e .`
5. 测试
    ```
    LD_PRELOAD="$HOME/orion/src/cuda_capture/libinttemp.so"  python benchmarking/launch_jobs.py --algo orion --config_file $HOME/orion/artifact_evaluation/example/config.json
    ```

### Use Docker image

We have set up a docker image: [fotstrt/orion-ae](https://hub.docker.com/repository/docker/fotstrt/orion-ae/general) with all packages pre-installed. We assume NVIDIA drivers are installed in the source machine, and that docker containers can use the host machine's GPUs.

* Start a container with `docker run --gpus=1 -it fotstrt/orion-ae:v1 bash`
* Download the Orion repo and install:
    * `git clone https://github.com/eth-easl/orion.git`
    * `cd orion`
    * `bash compile.sh`
    * `pip install -e .`


### Without Docker image

In order to use Orion without our pre-built image, a user must install:
* [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit). We have tested Orion with CUDA 10.2 and CUDA 11.3
* (optionally) [NVIDIA CUDNN](https://developer.nvidia.com/cudnn)
* Pytorch (from source) + TorchVision
* Download the Orion repo and install:
    * `git clone https://github.com/eth-easl/orion.git`
    * `cd orion`
    * `bash compile.sh`
    * `pip install -e .`
