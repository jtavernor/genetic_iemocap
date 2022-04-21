#!/bin/bash
cp PyTorch-NEAT-setup.py PyTorch-NEAT/setup.py
cd PyTorch-NEAT/
pip install .
cd - >> /dev/null
echo "PyTorch-NEAT installed"

cd pytorch-NEAT
pip install .
cd - >> /dev/null
echo "pytorch-neat installed (I know very distinctive names - this is the feedforward only network one the other is the recurrent/cppn one :))"