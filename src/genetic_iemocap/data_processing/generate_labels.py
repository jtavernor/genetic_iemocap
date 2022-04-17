from genetic_iemocap.consts import IEMOCAP_directory, gentle_directory

import os
import re
import numpy as np
import pandas as pd
import pickle

# The forced alignment code isn't actually necessary as IEMOCAP has aligned word timings already 
def read_IEMO_lbl():
    utt_id_to_word_timings = {}
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
                utt_id_to_word_timings[utt_id] = read_word_timings(utt_id, time[0])
                df_lst.append(df_loc)
            if line.startswith('A-'): # Added code to get evaluator specific labels but these probably won't be used 
                evaluator = line[:4]
                res = get_evaluator_scores.match(line)
                group_names = ['val', 'act', 'dom']
                for i, val in enumerate(res.groups()):
                    name = '{}_{}'.format(evaluator, group_names[i])
                    evaluator_columns[group_names[i]].add(name)
                    df_loc[name] = float(val) if val != '' else val
    lbl_df = pd.concat(df_lst)
    return lbl_df, evaluator_columns, utt_id_to_word_timings

def read_word_timings(utt_id, start_time):
    # Read corresponding word timings for this utterance 
    # Find the path
    word_timings = {'features': [], 'local_intervals': [], 'intervals': []}
    # print(utt_id)
    utt_details = re.search(r'.*(?P<sess_id>Ses0(?P<session_num>\d)[FM]_(impro|script)0\d[ab]?(_\d[ab]?)?)_[FM]\d+.*', utt_id)
    sess_id = utt_details.group('sess_id')
    # print(sess_id)
    session_num = utt_details.group('session_num')
    # print(session_num)
    timing_file = os.path.join(IEMOCAP_directory, f'Session{session_num}/sentences/ForcedAlignment/{sess_id}/{utt_id}.wdseg')
    # print('Based on the above the wdseg file is at', timing_file)
    timing_file_exists = os.path.exists(timing_file) and os.path.isfile(timing_file)
    if timing_file_exists:
        timing_extraction = re.compile(r'[^\d]*(?P<start_frame>\d+)\s+(?P<end_frame>\d+)\s+(?P<segascr>-?\d+)\s+(?P<word>[^\s]+).*')
        # print('Estimate output for utterance:')
        file_error = False

        with open(timing_file, 'r') as r:
            for i, line in enumerate(r.readlines()):
                if i == 0:
                    continue
                line = line.rstrip()
                matches = timing_extraction.match(line)
                if matches is None:
                    if file_error:
                        raise IOError(f'Failed to read values from timing string. Error occured parsing following lines\n{file_error}\n{line}\nOccured in timings file {timing_file}')
                    file_error = line # We can allow this once in a file as it's the last line in the file
                    continue
                start_frame = float(matches.group('start_frame'))
                end_frame = float(matches.group('end_frame'))
                word = matches.group('word')
                # print(start_frame/100, start_time)
                # print(end_frame/100, start_time)
                word_start_time = (start_frame/100) + start_time
                word_end_time = (end_frame/100) + start_time
                # print(f'{word_start_time}-{word_end_time} {word}')
                word_timings['features'].append(word)
                word_timings['intervals'].append([word_start_time, word_end_time])
    else:
        print('Warning - timing file doesn\'t exist, skipping', timing_file)
    return word_timings

# Load IEMOCAP labels 
all_labels, evaluator_columns, id_to_word_timings = read_IEMO_lbl()

# print(all_labels['utt_id'])
# print(all_labels['utt_id'].iloc[0])
# print(id_to_word_timings[all_labels['utt_id'].iloc[0]])

# Now apply processing and combine the files dependent on the type of processing selected
# mmfusion is legacy file processing for mmfusion model, if nothing is set then labels should be processed for this 
# project specifically

# How we will create these files is going to depend on if we are making legacy mmfusion labels
mmfusion_legacy = False
if mmfusion_legacy:
    raise NotImplemented()
else:
    label_file = {}
    transcript = {}
    word_matcher = re.compile(r'(?P<word>[a-z\']+)(?P<num>\(\d+\))?')
    for index, row in all_labels.iterrows():
        # print(row['utt_id'], row['act_lbl'], row['val_lbl'], row['dom_lbl'])
        label_file[row['utt_id']] = {
            'act': row['act_lbl'],
            'val': row['val_lbl'],
            'dom': row['dom_lbl'],
        }
        transcript[row['utt_id']] = {
            'features': [],
            'intervals': [],
        }
        timings = id_to_word_timings[row['utt_id']]
        skip_words = ['<sil>','<s>','</s>','++garbage++','++laughter++','++breathing++','++lipsmack++']
        for i, word in enumerate(timings['features']):
            # Skip words that represent features like silence <sil> 
            word = word.lower()
            if word in skip_words:
                continue
            matches = word_matcher.match(word)
            if matches is None:
                print('bad word:', word, 'in', row['utt_id'])
                continue
            transcript[row['utt_id']]['features'].append(matches.group('word'))
            transcript[row['utt_id']]['intervals'].append(timings['intervals'][i])

    label_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'labels')
    if not os.path.isdir(label_path):
        os.mkdir(label_path)

    with open(os.path.join(label_path, 'emo_labels.pk'), 'wb') as f:
        pickle.dump(label_file, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(label_path, 'transcripts.pk'), 'wb') as f:
        pickle.dump(transcript, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('WARNING - timings use a session timing, not in the utterance audio file, this may not be correct but will find out when implementing the specific model')