#!/bin/sh

LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libv4l/v4l2convert.so python3 ./webcam$1.py
