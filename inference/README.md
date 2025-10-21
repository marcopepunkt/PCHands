## Setup

### Get source
git clone https://github.com/NVlabs/FoundationPose.git
 
### Setup container
```commandline
cd FoundationPose/docker
docker pull wenbowen123/foundationpose && docker tag wenbowen123/foundationpose foundationpose
cd ..
# add --privileged to docker/run_container.sh
bash docker/run_container.sh
cd FoundationPose
bash build_all.sh
```

### Install pkgs
```commandline
apt update
apt install apt-transport-https
mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo focal main" | tee /etc/apt/sources.list.d/librealsense.list
apt update
apt install librealsense2-udev-rules=2.54.2-0~realsense.10772 librealsense2=2.54.2-0~realsense.10772 librealsense2-gl=2.54.2-0~realsense.10772 librealsense2-utils=2.54.2-0~realsense.10772
pip install -r requirements.txt
wget https://github.com/TimSchneider42/franky/releases/download/v0.9.1/libfranka_0-8-0_wheels.zip
unzip libfranka_0-8-0_wheels.zip
pip install --no-index --find-links=./dist franky-panda
```

## Usage

### Start container
```commandline
docker start foundationpose -i
```

### Connect running container
```commandline
docker exec -it foundationpose bash
```


### Todo

1. Permission to access usb port from robotiq gripper:
   ```commandline
   sudo chmod 666 /dev/ttyUSB0

2. Opencv display permission; run on host terminal:
   ```commandline
   xhost + local:
   ```
