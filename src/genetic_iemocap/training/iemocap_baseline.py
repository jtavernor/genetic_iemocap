from genetic_iemocap.data_providers.iemocap import IEMOCAPDataset
from genetic_iemocap.models.iemocap_baseline import Baseline

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

# TRAINING CONSTANTS #
bs = 32
num_workers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# END CONSTANTS # 

def custom_collate_fn(batch):
    global max_max_len
    audio_features = [sample[0]['audio_features'] for sample in batch]
    text_features = [sample[0]['text_features'] for sample in batch]
    labels = {label_name: [sample[1][label_name] for sample in batch] for label_name in batch[0][1]}

    # Want to flatten the audio features for this simple baseline as we don't need to consider the text features
    # print(len(audio_features))
    flat_batch_audio_features = []
    # Audio features is of shape batch_size x num intervals x spectrograms in interval x 40
    for i, val in enumerate(audio_features):
        # val is of shape num intervals x num spectrograms x 40
        # so we want to flatten this to num intervals*num spectrograms x 40 and add to above list
        # then we will pad this later 
        flat_audio_features = []
        for interval_list in val:
            # interval_list is of shape num_spectrograms x 40 
            for spectrogram in interval_list:
                # this is the length 40 spectrogram
                flat_audio_features.append(spectrogram)
        
        flat_audio_features = np.array(flat_audio_features)
        tensor_audio_features = torch.from_numpy(flat_audio_features)
        flat_batch_audio_features.append(tensor_audio_features)
    
    # Now find the maximum length in these features
    input_lengths = [len(b) for b in flat_batch_audio_features]
    # max_length = max(input_lengths)
    # print(max(input_lengths))
    # max_length = 2605
    
    # for i, length in enumerate(input_lengths):
    #     while length < max_length:
    #         flat_batch_audio_features[i].append([0]*40)
    #         length += 1

    # if max_length == 0:
        # print('0 length feature vector found?')
        # print(batch)
    # print(len(flat_batch_audio_features))
    # print(flat_batch_audio_features[0].size())
    flat_batch_audio_features = pad_sequence(flat_batch_audio_features, batch_first=True)
    # print(flat_batch_audio_features.size())
    # print(flat_batch_audio_features.size())
    return flat_batch_audio_features, input_lengths, labels

dataset = IEMOCAPDataset('../data_providers/labels.pk')
dataset.train()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn)

model = Baseline(40).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Training model with {params} parameters')
for epoch in range(1): # train for 5 epochs 
    avg_loss = 0
    for audio_features, input_lengths, labels in (pbar := tqdm(dataloader)):
        # TODO: Should we mask input lengths? Probably yes if there is time or results are terrible
        audio_features = audio_features.to(device, dtype=torch.float)
        activation_labels = torch.tensor(labels['act']).to(device)

        optimizer.zero_grad()

        outputs = model(audio_features)
        # print(activation_labels)
        loss = criterion(outputs, activation_labels)
        # print(loss.item())
        pbar.set_description(f'Batch loss {loss.item()}')
        # print(max_max_len)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch} avg loss {avg_loss/len(dataloader)}')


print('Finished training. Evaluating accuracy')

dataset.test()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn)

model.eval()
correct = 0
for audio_features, input_lengths, labels in tqdm(dataloader):
    audio_features = audio_features.to(device, dtype=torch.float)
    activation_labels = torch.tensor(labels['act']).to(device)
    
    outputs = model(audio_features)
    outputs = torch.argmax(F.softmax(outputs, dim=1), dim=1)
    correct += (outputs == activation_labels).float().sum()

accuracy = correct / len(dataset)

print(f'Activation Accuracy: {accuracy}')