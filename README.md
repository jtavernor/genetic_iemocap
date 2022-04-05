# genetic_iemocap
A genetic algorithm to topologically evolve an emotion recognition model for IEMOCAP.

# Setup
1. Ensure you are in a virtual environment.
2. Run `pip install .`
    - If doing development run `pip install -e .` as this will allow changes to be immediately reflected without reinstalling the package.

## If doing Alignment
1. Download [Gentle Aligner](https://github.com/lowerquality/gentle) somewhere on your system and install using the `install.sh`. 
2. Use the same virtual environment as described in the above setup. 
3. Ensure the directory to gentle aligner is setup in the `consts.py`.
3. Run `pip install .` in the forced aligner directory.