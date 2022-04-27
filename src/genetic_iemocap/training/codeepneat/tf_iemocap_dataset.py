import numpy as np
# import tensorflow as tf

def convert_iemocap_to_tensorflow(dataset, max_words=5, dim_size=10):
    dataset.train()
    train_features = []
    train_act_labels = []
    for key in dataset.active_dataset_keys:
        features = dataset.features[key]
        act_label = dataset.label_file[key]['act']
        features = np.array(features['audio_features']).squeeze(axis=1) # should now be words x 10 array
        padding_to_add = max_words - features.shape[0]
        if padding_to_add:
            padding = np.zeros((padding_to_add, dim_size))
            features = np.vstack([features, padding])
        train_features.append(features)
        train_act_labels.append(dataset.bin(act_label))
    train_features = np.array(train_features)
    train_act_labels = np.array(train_act_labels)

    # tf_train_set = tf.data.Dataset.from_tensor_slices((train_features, train_act_labels))

    dataset.test()
    test_features = []
    test_act_labels = []
    for key in dataset.active_dataset_keys:
        features = dataset.features[key]
        act_label = dataset.label_file[key]['act']
        features = np.array(features['audio_features']).squeeze(axis=1) # should now be words x 10 array
        padding_to_add = max_words - features.shape[0]
        if padding_to_add:
            padding = np.zeros((padding_to_add, dim_size))
            features = np.vstack([features, padding])
        test_features.append(features)
        test_act_labels.append(dataset.bin(act_label))
    test_features = np.array(test_features)
    test_act_labels = np.array(test_act_labels)
    
    # tf_test_set = tf.data.Dataset.from_tensor_slices((test_features, test_act_labels))

    return (train_features, train_act_labels), (test_features, test_act_labels)