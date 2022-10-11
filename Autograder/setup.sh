#!/usr/bin/env bash                                                             
apt-get install -y python3 python3-pip python3-dev
apt-get install -y python3-setuptools
apt-get install -y build-essential libssl-dev libffi-dev \
    python3-dev cargo
pip3 install setuptools_rust
pip3 install numpy
pip3 install scipy
pip3 install vpython
pip3 install matplotlib
pip3 install casadi