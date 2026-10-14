#!/bin/bash
set -e

export DISPLAY=:1

# 確認 X11 socket
if [ ! -S /tmp/.X11-unix/X1 ]; then
    echo "ERROR: X11 socket X1 不存在"
    exit 1
fi

# 授權本機 root 連接 X11
xhost +si:localuser:root

# 啟動 Docker
sudo docker run --rm -it \
  --runtime=nvidia \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device=/dev/video0 \
  --device=/dev/video2 \
  -e DISPLAY=:1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/yannichang/yanni_test_ai/llm:/workspace \
  -w /workspace \
  yolo-jp72-ui:latest \
  python3 vehicle_lane12_ui_source_switch.py \
    --front-camera 0 \
    --rear-camera 2