# We want to generate all the pickled label files here. 
# All data will be extracted to a single pickle file to reduce the overhead of usage. 
# IEMOCAP Utterance key -> (labels, word_timings)
# If flag --mmfusion-legacy then 3 files will be generated in the format specified for mmfusion. 

# Also ensure consts are set correctly
from genetic_iemocap.consts import IEMOCAP_directory, gentle_directory

# I don't like modifying the sys.path, however, there's no easy way to access the code in git submodule
# the only other choice would be using a symlink or something, it doesn't seem possible to include
# as a submodule in the python package
import sys
import os
sys.path.append(gentle_directory)
import gentle

resources = gentle.Resources()

with open('/home/tavernor/genetic_iemocap/gentle/examples/data/lucier.txt', encoding="utf-8") as fh:
    transcript = fh.read()

with gentle.resampled('/home/tavernor/genetic_iemocap/gentle/examples/data/lucier.mp3') as wavfile:
    aligner = gentle.ForcedAligner(resources, transcript)
    result = aligner.transcribe(wavfile)

print(result.to_json(indent=2))