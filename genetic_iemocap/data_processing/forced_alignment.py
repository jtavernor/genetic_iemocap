# We want to generate all the pickled label files here. 
# All data will be extracted to a single pickle file to reduce the overhead of usage. 
# IEMOCAP Utterance key -> (labels, word_timings)
# If flag --mmfusion-legacy then 3 files will be generated in the format specified for mmfusion. 

# Also ensure consts are set correctly
print(__package__)
print('running')
import sys
print(sys.path)
from genetic_iemocap.consts import IEMOCAP_directory
import gentle

resources = gentle.Resources()
logging.info("converting audio to 8K sampled wav")

with open('/home/tavernor/gentle/example/data/lucier.txt', encoding="utf-8") as fh:
    transcript = fh.read()

with gentle.resampled('/home/tavernor/gentle/example/data/lucier.mp3') as wavfile:
    logging.info("starting alignment")
    aligner = gentle.ForcedAligner(resources, transcript, nthreads=args.nthreads, disfluency=args.disfluency, conservative=args.conservative, disfluencies=disfluencies)
    result = aligner.transcribe(wavfile, progress_cb=on_progress, logging=logging)

print(result.to_json(indent=2))