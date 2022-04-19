from iemocap import IEMOCAPDataset

a = IEMOCAPDataset('labels.pk')
a.test()
print('Number of test utterances:', len(a))
a.train()
print('Number of train utterances:', len(a))

for features, label in a:
    print(features['text_features'][0])
    print(len(features['audio_features'][0]), features['audio_features'][0][0].shape)
    print(features['intervals'][0])
    print(label)
    crash