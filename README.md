# affNIST Benchmark #

There are two models in this project.
1. LeNet (CNN)
2. Caps Net

Both aim to train for the affNIST dataset (distorted and tralsated MNIST dataset, of sixe 40x40)

# Benchmarks #
1. We want to find the rate of learning from both networks. How fast do they learn to an acceptable accuracy?

2. We want to know if they can train accurately with less training samples.
    * We may start with the default 55000 training samples
    * Recude it to 40000, 30000, 20000
    * Compare accuracy numbers for BOTH models

# Docker env setup
Added custom modifications to run with affmnist dataset

# Docker setup for nvidia-cuda

* Build the image defined by the Dockerfile

  `nvidia-docker build -t tensorflow_nvidia_docker_setup .`

* Run an interactive bash sell within the container

  `nvidia-docker run --rm -it tensorflow_nvidia_docker_setup bash`
