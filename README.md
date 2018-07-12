# affNIST Benchmark #

There are two models in this project.
1. LeNet (CNN)
2. CapsNet

Both aim to train for the affNIST dataset (distorted and tralsated MNIST dataset, of sixe 40x40)

# Description of content #
`affNIST_data/` - Contains all affNIST input data to train the models

`capsnet_affnist` - CapsNet code. Modified from [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)

`CapsNet-Tensorflow` - CapsNet code with fixed-point accuracy adjustments. From my [my other repo](https://github.com/minghz/CapsNet-Tensorflow)

`lenet/` - LeNet-5 model with a few modifications

`Dockerfile.gpu run_gpu.sh` - Used to create Docker images and run on NVIDIA CUDA GPU

`Dockerfile.cpu run_cpu.sh` - Used to create Docker images and run on CPU

# Benchmarks #
1. We want to find the rate of learning from both networks. How fast do they learn to an acceptable accuracy?
    * Train with just_centered input, evaluate with transformed input
    * Train with just_centered input in addition to some "peppered" transformed input, evaluate with transformed input

2. Accuracy with less training samples.
    * We may start with the default 55000 training samples
    * Recude it to 40000, 30000, 20000
    * Compare accuracy numbers for BOTH models

# Results #
1. CapsNet learns in less steps than LeNet, but computation time is higher.

2. ?

# Docker env setup
Added custom modifications to run with affnist dataset

# Docker setup
* Build the image defined by the Dockerfile

  `docker build -f Dockerfile.cpu -t tensorflow_nvidia_docker_setup .`

* Run an interactive bash sell within the container

  `docker run --rm -it tensorflow_docker_setup bash`

# Docker setup for nvidia-cuda
* Build the image defined by the Dockerfile

  `nvidia-docker -f Dockerfile.gpu build -t tensorflow_nvidia_docker_setup .`

* Run an interactive bash sell within the container

  `nvidia-docker run --rm -it tensorflow_nvidia_docker_setup bash`
