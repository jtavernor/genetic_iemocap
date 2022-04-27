#!/bin/bash
# Reset to default in case this is a reinstall
cd pytorch-neat/
rm -rf neat_pytorch_linear
git stash >> /dev/null
cd - >> /dev/null

cp submodule_updated_files/PyTorch-NEAT-setup.py PyTorch-NEAT/setup.py
cd PyTorch-NEAT/
pip install -e .
cd - >> /dev/null
echo "pytorch-neat-rnn (PyTorch-NEAT) installed as pytorch_neat"

cp submodule_updated_files/pytorch_neat_setup.py pytorch-neat/setup.py
cp submodule_updated_files/population.py pytorch-neat/neat/population.py
cp submodule_updated_files/species.py pytorch-neat/neat/species.py
mv pytorch-neat/neat pytorch-neat/neat_pytorch_linear
find pytorch-neat/neat_pytorch_linear -type f -exec sed -i 's/neat\./neat_pytorch_linear./g' {} \;
cd pytorch-neat
pip install -e .
cd - >> /dev/null
echo "pytorch-neat (pytorch-neat) installed as neat_pytorch_linear (I know very distinctive names - this is the feedforward only network one the other is the recurrent/cppn one :))"


# Finally a nice easy install! 
cd Tensorflow-Neuroevolution/
git stash >> /dev/null
cd - >> /dev/null
cp submodule_updated_files/codeepneat.py Tensorflow-Neuroevolution/tfne/algorithms/codeepneat/codeepneat.py
cp submodule_updated_files/codeepneat_module_association.py Tensorflow-Neuroevolution/tfne/encodings/codeepneat/modules/codeepneat_module_association.py
cd Tensorflow-Neuroevolution
pip install -e .
cd - >> /dev/null
echo "Tensorflow-Neuroevolution installed as tfne"

# # Also need to reset in case of reinstall
# cd neuroevolution-deepneat
# rm -rf deepneat
# git stash >> /dev/null
# cd - >> /dev/null

# # This is such an annoying one to install, they refer to their own code and the base code as neat in both, so need to 
# # address these cases where it should be referring to itself and when it should be referring to original neat 
# cp submodule_updated_files/deepneat-setup.py neuroevolution-deepneat/setup.py
# mv neuroevolution-deepneat/neat neuroevolution-deepneat/deepneat
# find neuroevolution-deepneat/deepneat -type f -exec sed -i 's/neat\./deepneat./g' {} \;
# find neuroevolution-deepneat/deepneat -type f -exec sed -i 's/import neat/import deepneat/g' {} \;
# find neuroevolution-deepneat/deepneat -type f -exec sed -i 's/from neat/from deepneat/g' {} \;
# # Looks like they only need the base neat implementation for iznn ctrnn and nn as they provide files for the rest 
# find neuroevolution-deepneat/deepneat -type f -exec sed -i 's/deepneat\.ctrnn/neat.ctrnn/g' {} \;
# find neuroevolution-deepneat/deepneat -type f -exec sed -i 's/deepneat\.iznn/neat.iznn/g' {} \;
# find neuroevolution-deepneat/deepneat -type f -exec sed -i 's/deepneat\.nn/neat.nn/g' {} \;

# cd neuroevolution-deepneat
# pip install -e .
# cd - >> /dev/null
# echo "neuroevolution-deepneat installed as deepneat"