#!/bin/bash

# osx
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
#bash Miniconda3-latest-Linux-aarch64.sh -b -p $HOME/miniconda

# intel/amd
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda

$HOME/miniconda/bin/conda init
source $HOME/.bashrc
# Will only work for intel/amd because conda-forge has no aarch64 for TRIQS
conda install -c conda-forge triqs
python3 -m pip install -e .
python3 -m pip install -e .[test]
python3 -m pip install -e .[docs]
