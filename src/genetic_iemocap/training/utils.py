import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class CustomCollation:
    def __init__(self, max_len, batch_level_padding=False):
        # MAX_LEN = 2605 # full dataset
        self.max_len = max_len
        self.batch_level_padding = batch_level_padding # If false pad to MAX_LEN (which iemocap should have as max length - truncate must be called if less than 2605)

    def custom_collate_fn(self, batch):
        audio_features = [sample[0]['audio_features'] for sample in batch]
        text_features = [sample[0]['text_features'] for sample in batch]
        labels = {label_name: [sample[1][label_name] for sample in batch] for label_name in batch[0][1]}

        # Want to flatten the audio features for this simple baseline as we don't need to consider the text features
        # print(len(audio_features))
        flat_batch_audio_features = []
        dim_size = None
        # Audio features is of shape batch_size x num intervals x spectrograms in interval x spectrogram dim size
        for i, val in enumerate(audio_features):
            # val is of shape num intervals x num spectrograms x spectrogram dim size
            # so we want to flatten this to num intervals*num spectrograms x spectrogram dim size and add to above list
            # then we will pad this later 
            flat_audio_features = []
            for interval_list in val:
                # interval_list is of shape num_spectrograms x spectrogram dim size 
                for spectrogram in interval_list:
                    # this is the length spectrogram dim size spectrogram
                    flat_audio_features.append(spectrogram)
                    if dim_size is None:
                        dim_size = len(spectrogram)
            
            flat_audio_features = np.array(flat_audio_features)
            tensor_audio_features = torch.from_numpy(flat_audio_features)
            flat_batch_audio_features.append(tensor_audio_features)
        
        # Now find the maximum length in these features
        input_lengths = [len(b) for b in flat_batch_audio_features]
        # max_length = max(input_lengths)
        # print(max(input_lengths))

        # Code for padding to max length:
        if self.batch_level_padding:
            # Code for batch-level padding
            flat_batch_audio_features = pad_sequence(flat_batch_audio_features, batch_first=True)
        else:        
            for i, length in enumerate(input_lengths):
                padding = torch.zeros((self.max_len-length, dim_size))
                flat_batch_audio_features[i] = torch.cat([flat_batch_audio_features[i],padding])

            flat_batch_audio_features = torch.stack(flat_batch_audio_features, dim=0).float()
        return flat_batch_audio_features, input_lengths, labels
