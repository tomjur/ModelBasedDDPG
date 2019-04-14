#!/usr/bin/env bash
set -e

apt-get update -q
apt-get install -qy build-essential cmake \
                    ann-tools libann-dev libassimp-dev \
                    libavcodec-dev libavformat-dev libboost-python-dev \
                    libboost-all-dev libccd-dev libcollada-dom2.4-dp-dev \
                    libeigen3-dev libflann-dev liblapack-dev liblog4cxx-dev \
                    libminizip-dev liboctave-dev libode-dev libpcre3-dev \
                    libqhull-dev libsoqt-dev-common libsoqt4-dev \
                    libswscale-dev libtinyxml-dev libxml2-dev octomap-tools \
                    libjsoncpp-dev
