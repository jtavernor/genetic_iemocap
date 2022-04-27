from genetic_iemocap.data_providers.iemocap import IEMOCAPDataset
from genetic_iemocap.models.iemocap_baseline import Baseline
from genetic_iemocap.training.utils import custom_collate_fn

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

# TRAINING CONSTANTS #
bs = 32
num_workers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# END CONSTANTS # 

dataset = IEMOCAPDataset('../data_providers/labels.pk')
dataset.train()

collater = CustomCollation(max_len=2605)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers, collate_fn=collater.collate_fn)

model = Baseline(2605).to(device)

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
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers, collate_fn=collater.custom_collate_fn)

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