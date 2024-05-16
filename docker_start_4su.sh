PROJECT_ROOT=/aifs4su/mmcode/codeclm/MARBLE-Benchmark
# DATA_ROOT_old=/raid
DOCKER_IMG=registry-intl.cn-hongkong.aliyuncs.com/sixpublic/pytorch:23.10-py3

# 4su的data都在dgx-018的/raid里，不在ddn里。记得login 4su进行debug。

docker run --name marble --net=host --ipc=host --rm -it --shm-size=1024g --ulimit memlock=-1 --privileged \
     -e NVIDIA_VISIBLE_DEVICES=all \
     -e NCCL_SOCKET_IFNAME=ibp \
     -e NCCL_IB_HCA=mlx5 \
     -e NCCL_DEBUG_SUBSYS=ALL \
     -e MASTER_PORT=6000 \
     -e PROJECT_ROOT=/workspace/marble \
     -v $PROJECT_ROOT:/workspace/marble \
     -v /run/mellanox/drivers:/run/mellanox/drivers:shared \
     -v /etc/network:/etc/network \
     -v /etc:/host/etc \
     -v /lib/udev:/host/lib/udev \
     -w /workspace/marble \
     $DOCKER_IMG

# --gpus all
