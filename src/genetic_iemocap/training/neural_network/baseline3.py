from genetic_iemocap.data_providers.iemocap import IEMOCAPDataset
from genetic_iemocap.models.iemocap_baseline import Baseline
from genetic_iemocap.training.utils import CustomCollation

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import matplotlib.pyplot as plt


# TRAINING CONSTANTS #
bs = 32
num_workers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 2
num_times_to_evaluate = 70 # will get total train set accuracy 10 times total 
max_num_words = 5
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# END CONSTANTS # 

dataset = IEMOCAPDataset('../data_providers/labels_dim10.pk', avg_audio_features=True, dim_size=10) # average mels over words
dataset.truncate(max_num_words) # truncate dataset to only include utterances of length <= 10 words
dataset.print_stats()
dataset.train()

collater = CustomCollation(max_len=max_num_words)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers, collate_fn=collater.custom_collate_fn)

model = Baseline(max_num_words).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Training model with {params} parameters')

def get_train_accuracy(model, dataset):
    model.eval()
    dataset.train()
    eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers, collate_fn=collater.custom_collate_fn)
    correct = 0
    for audio_features, input_lengths, labels in (pbar := tqdm(eval_dataloader)):
        audio_features = audio_features.to(device, dtype=torch.float)
        activation_labels = torch.tensor(labels['act']).to(device)

        outputs = model(audio_features)
        outputs = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        correct += (outputs == activation_labels).float().sum()
    model.train()
    correct = correct.cpu().item()
    return correct/len(dataset)

accuracy = []
with torch.no_grad():
    accuracy.append(get_train_accuracy(model, dataset)) # accuracies
batch_its_evaled_at = [0] # what batch number the evaluation was done at
eval_every_n = num_epochs*len(dataloader)//num_times_to_evaluate
print('Evaluating on train set every', eval_every_n, 'batches')
iters_done = 0
for epoch in range(num_epochs):
    avg_loss = 0
    correct = 0
    for audio_features, input_lengths, labels in (pbar := tqdm(dataloader)):
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

        iters_done += 1
        if iters_done % eval_every_n == 0:
            # evaluate on total test set
            batch_its_evaled_at.append(iters_done)
            with torch.no_grad():
                accuracy.append(get_train_accuracy(model, dataset))

    print(f'Epoch {epoch} avg loss {avg_loss/len(dataloader)}')

plt.plot(batch_its_evaled_at, accuracy)
plt.xlabel('Batch Iterations Trained')
plt.ylabel('Train set accuracy')
plt.savefig('baseline3_accuracy.png')

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