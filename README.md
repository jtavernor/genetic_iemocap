# genetic_iemocap
A genetic algorithm to topologically evolve an emotion recognition model for IEMOCAP.

# Setup
1. Ensure you are in a virtual environment.
2. Setup up submodules
    - This sets up the [PyTorch-NEAT](https://github.com/uber-research/PyTorch-NEAT/tree/master) code that we will use as the NEAT implementation. 
    - Run `git submodule update --init --recursive`
    - Again - double check you're in a virtual environment as pip commands will be run in this script - now run `./install-neat.sh`
3. Change directory into the code directory `cd src`
4. Run `pip install .`
    - If doing development run `pip install -e .` as this will allow changes to be immediately reflected without reinstalling the package.
 
# Reproduction/Viewing (some) results
Inside the `src/genetic_iemocap/training` are the folders that correspond to the training script. They should run without further hassle - though you will need a local installation of the (IEMOCAP dataset from the SLED group at USC)[https://sail.usc.edu/iemocap/]

The folders correspond to the model names in the paper/presentation as 
- `neural_network` - baseline
- `pytorch_neat` - pytorch-neat
- `pytorch_neat_rnn` - pytorch-neat-rnn
- `codeepneat` - CoDeepNEAT

The file namings inside are a little confused - this project certainly felt like a time crunch with how slow some of the evolutions were taking, but I really enjoyed working on it and would be happy to answer any questions about the codebase. :)
