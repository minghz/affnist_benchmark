#!/bin/bash
nvidia-docker build -f Dockerfile.gpu -t tensorflow_nvidia_docker_setup .
