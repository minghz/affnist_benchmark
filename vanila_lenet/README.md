# LeNet model from Tensorflow tutorials
Original: https://raw.githubusercontent.com/tensorflow/models/master/tutorials/image/mnist/convolutional.py
Taken June 8th, 2018

* Modify script to take in 40x40 affNIST input
* Add ability to write results (training/accuracy) to file

# NVIDIA CUDA container extra care
Reference: https://hub.docker.com/r/tensorflow/tensorflow/

### Install nvidia-docker
Read the official reference documents throughly!!! This doc may be outdated by the time you are reading it.
Reference: https://github.com/NVIDIA/nvidia-docker

Quick command reference

* `sudo apt-get update`

* `sudo apt-get install cuda-drivers`

```
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
```

* Add your username to the docker group (so you don't have to sudo every time)

* Reboot

* Test with `docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi`

# Tensorflow docker setup for NVIDIA CUDA container
Sample repository to spin up a tensorflow nvidia-docker container

* Build the image defined by the Dockerfile

  `nvidia-docker build -t tensorflow_nvidia_docker_setup .`

* Run an interactive bash sell within the container

  `nvidia-docker run --rm -it tensorflow_nvidia_docker_setup bash`

* Run tensorflow python script
