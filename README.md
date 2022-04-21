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