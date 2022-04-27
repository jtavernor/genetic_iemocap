from genetic_iemocap.consts import IEMOCAP_directory
from torch.utils.data import Dataset
from tqdm import tqdm

import os
import re
import librosa
import pickle
import math
import random
import numpy as np
import pandas as pd

class IEMOCAPDataset(Dataset):
    def __init__(self, saved_labels_path, bin_labels=True, iemocap_root_dir=IEMOCAP_directory, avg_audio_features=False, dim_size=40):
        self.IEMOCAP_directory = iemocap_root_dir
        self.bin_labels = bin_labels
        self.dim_size = dim_size
        self.mel_spec_stats = {
            'mean': [], 'std': []
        }
        # Load IEMOCAP labels 
        if saved_labels_path is None or not os.path.isfile(saved_labels_path):
            print('No labels found, regenerating them...')
            if saved_labels_path == None:
                saved_labels_path = 'iemocap_labels.pk'
            self.test_keys = []
            self.train_keys = []
            self.label_file = {}
            self.features = {}
            all_labels, evaluator_columns, id_to_word_timings = self.read_IEMO_lbl()

            word_matcher = re.compile(r'(?P<word>[a-z\']+)(?P<num>\(\d+\))?')
            print('dropping non-word audio/text features')
            for index, row in tqdm(all_labels.iterrows()):
                # print(row['utt_id'], row['act_lbl'], row['val_lbl'], row['dom_lbl'])
                self.label_file[row['utt_id']] = {
                    'act': row['act_lbl'],
                    'val': row['val_lbl'],
                    'dom': row['dom_lbl'],
                }
                self.features[row['utt_id']] = {
                    'text_features': [],
                    'audio_features': [],
                    'intervals': [],
                }
                timings = id_to_word_timings[row['utt_id']]
                skip_words = ['<sil>','<s>','</s>','++garbage++','++laughter++','++breathing++','++lipsmack++']
                for i, word in enumerate(timings['text_features']):
                    # Skip words that represent features like silence <sil> 
                    word = word.lower()
                    if word in skip_words:
                        continue
                    matches = word_matcher.match(word)
                    if matches is None:
                        print('bad word:', word, 'in', row['utt_id'])
                        continue
                    self.features[row['utt_id']]['text_features'].append(matches.group('word'))
                    self.features[row['utt_id']]['intervals'].append(timings['intervals'][i])
                    self.features[row['utt_id']]['audio_features'].append(timings['audio_features'][i])
                
                # Verify any features that only contained items like ++breathing++ are not in the dataset
                if len(self.features[row['utt_id']]['text_features']) == 0:
                    del self.features[row['utt_id']]
                    if row['utt_id'] in self.test_keys:
                        self.test_keys.remove(row['utt_id'])
                    if row['utt_id'] in self.train_keys:
                        self.train_keys.remove(row['utt_id'])
            
            print('Normalising/standardising the mel spectrogram values')
            mel_mean = np.mean(self.mel_spec_stats['mean'])
            mel_std = np.mean(self.mel_spec_stats['std'])
            for utt_id in self.features:
                for i, interval in enumerate(self.features[utt_id]['audio_features']):
                    for j, mel_spectrogram in enumerate(interval):
                        self.features[utt_id]['audio_features'][i][j] = (self.features[utt_id]['audio_features'][i][j] - mel_mean) / mel_std

            saved_labels = {
                'test_keys': self.test_keys,
                'train_keys': self.train_keys,
                'label_file': self.label_file,
                'features': self.features,
            }

            with open(saved_labels_path, 'wb') as f:
                pickle.dump(saved_labels, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            saved_labels = None
            with open(saved_labels_path, 'rb') as f:
                saved_labels = pickle.load(f)
            
            self.test_keys = saved_labels['test_keys']
            self.train_keys = saved_labels['train_keys']
            self.label_file = saved_labels['label_file']
            self.features = saved_labels['features']
        
        self.full_dataset_keys = self.train_keys
        self.active_dataset_keys = self.full_dataset_keys

        if avg_audio_features:
            self.avg_audio_features_fn()

    def __len__(self):
        return len(self.active_dataset_keys)
    
    def max_len(self):
        return len(self.full_dataset_keys)
    
    def __getitem__(self, idx):
        key = self.active_dataset_keys[idx]
        labels = self.label_file[key]

        return self.features[key], {label: self.bin(self.label_file[key][label]) for label in self.label_file[key]}
    
    def bin(self, label):
        newlbl = math.floor(label)
        binning = mapping = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
        return binning[newlbl]

    def read_IEMO_lbl(self):
        utt_id_to_word_timings = {}
        utt_id_to_audio = {}
        evaluator_columns = {'act': set(), 'val': set(), 'dom': set()}
        lab_file = os.path.join(self.IEMOCAP_directory, "IEMOCAP_EmoEvaluation.txt")
        get_evaluator_scores = re.compile(r'.*val\s+(\d);\s+act\s+(\d);\s+dom\s+(\d?);.*')
        get_times = re.compile(r'.*\[([^\s]+)\s+-\s+([^\]]+)\].*')
        df_lst = []
        with open(lab_file, 'r') as r:
            print('Reading labels and generating/aligning audio features to textual features')
            for line in tqdm(r.readlines()):
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
                    return_val = self.get_transcript_and_audio(utt_id, time[0])
                    if return_val is None:
                        continue
                    timings, session_id = return_val
                    if len(timings['text_features']) and len(timings['audio_features']):
                        if int(session_id) == 5:
                            self.test_keys.append(utt_id)
                        else:
                            self.train_keys.append(utt_id)
                        utt_id_to_word_timings[utt_id] = timings
                    else:
                        print('Failed to load text or audio features, not using utt id:', utt_id)
                        continue
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

    def get_transcript_and_audio(self, utt_id, start_time):
        # Read corresponding word timings for this utterance 
        # Find the path
        word_timings = {'text_features': [], 'audio_features': [], 'local_intervals': [], 'intervals': []}
        # print(utt_id)
        utt_details = re.search(r'.*(?P<sess_id>Ses0(?P<session_num>\d)(?P<sess_spkr>[FM])_(impro|script)0\d[ab]?(_\d[ab]?)?)_(?P<utt_spkr>[FM])\d+.*', utt_id)
        # if utt_details.group('sess_spkr') != utt_details.group('utt_spkr'):
        #     return None # The first F/M refers to who has the cameras/mocap/recording focused on them, so we only use this speaker's utterances
        sess_id = utt_details.group('sess_id')
        # print(sess_id)
        session_num = utt_details.group('session_num')
        audio_file = os.path.join(IEMOCAP_directory, f'Session{session_num}/sentences/wav/{sess_id}/{utt_id}.wav')

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
                    word_timings['text_features'].append(word)
                    word_timings['intervals'].append([word_start_time, word_end_time])
                    word_timings['local_intervals'].append([start_frame/100, end_frame/100])
            
            # Now load the audio mfb and align with the intervals seen already, using local intervals as the IEMOCAP on oolong has the audio per utterance not the audio per session
            word_timings['text_features'] = np.array(word_timings['text_features'])
            word_timings['local_intervals'] = np.array(word_timings['local_intervals'])
            word_timings['audio_features'] = self.load_audio_file(audio_file, word_timings['local_intervals'])
            
        else:
            print('Warning - timing file doesn\'t exist, skipping', timing_file)
            return None
        return word_timings, session_num

    def load_audio_file(self, audio_file_path, timings):
        # get_mfb -- this is stored in the tensors file in mmfusion 
        # the mfb are unaligned but the mfb timing can be found, so I would presume the timing of the mfb is then aligned with the timing of the words from forced alignment 
        # that should be easy enough to implement here.
        # mmfusion also calculates mean and std to do some kind of standardisation on the audio files but I want to keep implementation simple here so will just load melspectograms 
        
        ### CONSTANTS ###
        SR = 16000 # 16,000 samples per second
        n_fft = 2048 # length of the FFT window
        n_mels = self.dim_size # size of mel vector 
        hop_length = 160 # number of samples between frames (i.e. each frame is 160 samples, or 0.01 seconds)
        interval_length = hop_length/SR
        fmin = 0
        fmax = None
        ### END CONSTANTS ###

        y, sr = librosa.load(audio_file_path, sr=SR)
        # y = librosa.effects.preemphasis(y, coef=0.97)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, fmin=fmin, fmax=fmax, htk=False)

        mel_spect = mel_spec.T # Reshape so timestep is now the first dimension i.e. shape is (timesteps, hop_length)

        self.mel_spec_stats['mean'].append(np.mean(mel_spect))
        self.mel_spec_stats['std'].append(np.std(mel_spect))

        # Now determine the time intervals for each of the mel spectograms
        start_times = np.arange(0,mel_spect.shape[0]*interval_length - interval_length,interval_length)
        end_times = np.arange(interval_length,mel_spect.shape[0]*interval_length,interval_length)
        mel_intervals = np.vstack([start_times, end_times]).T

        # Now determine which interval the spectrogram is part of, and then store this so that the textual data and audio data are aligned to the same segments (with each segment being a word in speech)
        audio_features = [[] for i in timings]
        # print('given', timings)
        for i, interval in enumerate(mel_intervals):
            start, end = interval
            all_starts = timings[:,0]
            all_ends = timings[:,1]
            # To find the appropriate alignment for the current segment, we should first check for values that are definitely in the original interval range
            valid_start = np.argwhere(all_starts <= start)
            valid_end = np.argwhere(all_ends >= end)
            # If end < all_starts, or all_ends < start, then this audio feature is outside of the actual speech so shouldn't be used
            invalid = (end < all_starts).all() or (all_ends < start).all()
            # print(interval)
            if invalid:
                # print(timings)
                # print('interval', interval, 'outside valid range so skipping')
                continue
            valid_idxs = np.intersect1d(valid_start,valid_end)
            best_interval_idx = None
            if len(valid_idxs):
                best_interval_idx = valid_idxs.min() # Just pick the minimum index possible, this should only return 1 valid index anyway
            else:
                # In this case the audio feature crosses a boundary, so we just pick the index with the most overlap 
                if len(valid_start) == 0:
                    best_interval_idx = valid_end.min()
                elif len(valid_end) == 0:
                    best_interval_idx = valid_start.min()
                else:
                    # print(valid_start)
                    min_idx = valid_start.min()
                    max_idx = valid_end.min()
                    # look at each interval
                    min_idx_time = all_ends[min_idx] - start
                    max_idx_time = end - all_starts[max_idx]
                    # print('min idx', min_idx, min_idx_time)
                    # print('max idx', max_idx, max_idx_time)
                    if min_idx_time > max_idx_time:
                        best_interval_idx = min_idx
                    else:
                        best_interval_idx = max_idx
                
            best_interval_idx = np.abs((all_starts - start) + (all_ends - end)).argmin()
            audio_features[best_interval_idx].append(mel_spect[i])
            # print('adding features from interval', start, end, 'to audio feature interval number', best_interval_idx)
        
        return audio_features
    
    def train(self):
        self.full_dataset_keys = self.train_keys
        self.active_dataset_keys = self.full_dataset_keys
    
    def test(self):
        self.full_dataset_keys = self.test_keys
        self.active_dataset_keys = self.full_dataset_keys

    def use_subset(self, subset_size):
        self.active_dataset_keys = random.choices(self.full_dataset_keys, k=subset_size)
    
    def truncate(self, max_seq_len):
        cur_dataset_is_train = self.active_dataset_keys[0] in self.train_keys
        all_keys = self.train_keys + self.test_keys
        for key in all_keys:
            seq_len = self.get_seq_len(key)
            if seq_len > max_seq_len:
                for key_list in [self.train_keys, self.test_keys, self.full_dataset_keys, self.active_dataset_keys]:
                    while key in key_list:
                        key_list.remove(key)
        
        if cur_dataset_is_train:
            self.train()
        else:
            self.test()
    
    def print_stats(self):
        print(f'Dataset train size {len(self.train_keys)}')
        print(f'Dataset test size {len(self.test_keys)}')

    def get_seq_len(self, key):
        features = self.features[key]
        audio_features = features['audio_features']
        flat_audio_features = []
        for interval_list in audio_features:
            # interval_list is of shape num_spectrograms x 40 
            for spectrogram in interval_list:
                # this is the length 40 spectrogram
                flat_audio_features.append(spectrogram)
        flat_audio_features = np.array(flat_audio_features)
        return flat_audio_features.shape[0]

    def avg_audio_features_fn(self):
        print('Averaging audio features per word')
        for key in tqdm(self.train_keys + self.test_keys):
            # will be of shape num_intervals x num_mels x 40 
            # want to avg to get num_intervals x 40 i.e. an averaged mel spectrogram for each word 
            audio_features = self.features[key]['audio_features']
            for i, interval_mels in enumerate(audio_features):
                # now want to average all mels in interval_mels to a single vector of 40
                np_mels = np.array(interval_mels)
                mean_mel = np.mean(np_mels, axis=0)
                self.features[key]['audio_features'][i] = [mean_mel]