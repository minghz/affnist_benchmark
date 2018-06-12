#!/bin/bash
nvidia-docker run -v `pwd`:/app -u $(id -u) --rm -it tensorflow_nvidia_docker_setup bash
