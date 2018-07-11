#!/bin/bash
nvidia-docker -f Dockerfile.gpu build -t tensorflow_nvidia_docker_setup .
