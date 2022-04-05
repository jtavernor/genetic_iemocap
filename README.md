# genetic_iemocap
A genetic algorithm to topologically evolve an emotion recognition model for IEMOCAP.

# Setup
1. Ensure you are in a virtual environment.
2. Setup up submodules
    - This is only needed if running forced alignment - a git submodule of [Gentle Aligner](https://github.com/lowerquality/gentle) is included for this purpose
    - Run `git submodule update --init --recursive`
    - Now go into the gentle aligner directory `cd gentle`
    - Install the gentle code using `install.sh`, this may need to be run as `sudo`
    - You can test that the aligner code is setup correctly by running `python align.py examples/data/lucier.mp3 examples/data/lucier.txt` from within the `gentle` directory
    - Ensure that the directory to gentle aligner is setup in the `consts.py` file
3. Change directory into the code directory `cd src`
4. Run `pip install .`
    - If doing development run `pip install -e .` as this will allow changes to be immediately reflected without reinstalling the package.