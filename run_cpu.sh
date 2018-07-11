#!/bin/bash
docker run -p 6006:6006 -v `pwd`:/app -u $(id -u) --rm -it tensorflow_docker_setup bash
