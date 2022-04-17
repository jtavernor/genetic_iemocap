# We want to generate all the pickled label files here. 
# All data will be extracted to a single pickle file to reduce the overhead of usage. 
# IEMOCAP Utterance key -> (labels, word_timings)
# If flag --mmfusion-legacy then 3 files will be generated in the format specified for mmfusion. 

# Also ensure consts are set correctly
from genetic_iemocap.consts import IEMOCAP_directory, gentle_directory

# I don't like modifying the sys.path, however, there's no easy way to access the code in git submodule
# the only other choice would be using a symlink or something, it doesn't seem possible to include
# as a submodule in the python package
# The alternative would be fully installing gentle but I think that in order to maintain this as a git
# submodule this is the only real choice 
import sys
import os
sys.path.append(gentle_directory)
import gentle

# We now want to load the IEMOCAP wav directories and transcripts for every utterance
def read_IEMO_lbl():
    evaluator_columns = {'act': set(), 'val': set(), 'dom': set()}
    lab_file = os.path.join(IEMOCAP_directory, "IEMOCAP_EmoEvaluation.txt")
    get_evaluator_scores = re.compile(r'.*val\s+(\d);\s+act\s+(\d);\s+dom\s+(\d?);.*')
    get_times = re.compile(r'.*\[([^\s]+)\s+-\s+([^\]]+)\].*')
    df_lst = []
    with open(lab_file, 'r') as r:
        for line in r.readlines():
            line=line.rstrip()
            cur_df_loc = None
            if line.startswith("[") and line.endswith("]"):
                time = line.split("\t")[0]
                t1,t2 = get_times.match(time).groups()
                time = np.array([t1,t2], dtype=np.float32)
                utt_id = line.split("\t")[1]
                cat_lbl = line.split("\t")[2]

                val_lbl = float(line.split("\t")[3][1:-1].split(", ")[0])
                act_lbl = float(line.split("\t")[3][1:-1].split(", ")[1])
                dom_lbl = float(line.split("\t")[3][1:-1].split(", ")[2])
                if cur_df_loc is not None: # use this cur df loc approach so individual evaluator results can be appended
                    df_lst.append(cur_df_loc)
                df_loc = pd.DataFrame({
                    'time':[time],
                    'utt_id':[utt_id],
                    'cat_lbl':[cat_lbl],
                    'val_lbl':[val_lbl],
                    'act_lbl':[act_lbl],
                    'dom_lbl':[dom_lbl],
                })
                df_lst.append(df_loc)
            if line.startswith('A-'): # Added code to get evaluator specific labels but these probably won't be used 
                evaluator = line[:4]
                res = get_evaluator_scores.match(line)
                group_names = ['val', 'act', 'dom']
                for i, val in enumerate(res.groups()):
                    name = '{}_{}'.format(evaluator, group_names[i])
                    evaluator_columns[group_names[i]].add(name)
                    df_loc[name] = float(val) if val != '' else val

# Load IEMOCAP labels 
all_labels = read_IEMO_lbl()



# We're going to save the extracted JSONs to a json file here so that we can process them into labels
# separately without re-running gentle aligner
json_save_loc = os.path.dirname(os.path.realpath(__file__)) + 'gentle_outputs.json'
print(json_save_loc)
if not os.path.isfile(json_save_loc): # TODO Loop iemocap wavs and save to json 
    resources = gentle.Resources()

    with open('/home/tavernor/genetic_iemocap/gentle/examples/data/lucier.txt', encoding="utf-8") as fh:
        transcript = fh.read()

    with gentle.resampled('/home/tavernor/genetic_iemocap/gentle/examples/data/lucier.mp3') as wavfile:
        aligner = gentle.ForcedAligner(resources, transcript)
        result = aligner.transcribe(wavfile)

    print(result.to_json(indent=2))


# Process json into pickle files
