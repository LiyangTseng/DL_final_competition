import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

MAX_SEQ_LEN = 1007
TRAIN_RATIO = 0.8

class MFCC_Dataset(Dataset):
    def __init__(self, features_dir, csv_file_path=None, mode='test'):
        super(MFCC_Dataset, self).__init__()
        self.features_dir = features_dir
        self.mode = mode
        if self.mode != 'test':
            df = pd.read_csv(csv_file_path)
            ''' Shuffle DataFrame !!'''
            df = df.sample(frac=1).reset_index(drop=True)
            self.label_map = {j: i for i, j in enumerate(df.scenario.unique())}
            if mode == 'train':
                self.data_df = df[:int(len(df)*TRAIN_RATIO)]
            elif mode == 'valid':
                self.data_df = df[int(len(df)*TRAIN_RATIO):]

    def __len__(self):
        if self.mode != 'test':
            return len(self.data_df)
        else:
            return len(os.listdir(self.features_dir))

    def __getitem__(self, index):
        if self.mode == 'test':
            feature_filename = os.path.join(self.features_dir, '{0:06}.npy'.format(index))
            feature = np.load(feature_filename)
            return torch.tensor(feature)

        elif self.mode == 'valid':
            index = index + int(len(os.listdir(self.features_dir))*TRAIN_RATIO)
        
        elif self.mode != 'train':
            raise Exception('Non-existent mode!')

        feature_filename = os.path.join(self.features_dir, '{0:06}.npy'.format(index))
        feature = np.load(feature_filename)
        label = self.label_map[ self.data_df.loc[index]['scenario'] ]
        # feature: (20, timestep)
        # label: (10, )
        return torch.tensor(feature), torch.tensor(label)

def custom_collate_fn(batch_data):
    '''
    input: list of tensors
    output: tensors
    '''
    # padding
    batch_inputs = [sample[0] for sample in batch_data]
    batch_labels = [sample[1] for sample in batch_data]
    max_len = MAX_SEQ_LEN

    seq_lens = []
    for idx in range(len(batch_data)):
        feature_len = batch_inputs[idx].size(1)
        seq_lens.append(feature_len)
        # padding
        batch_inputs[idx] = F.pad(batch_inputs[idx], (0, max_len-feature_len))

    padded_batch_inputs = torch.stack(batch_inputs)
    try:
       # pack sequence
        packed_batch_inputs = pack_padded_sequence(padded_batch_inputs, seq_lens, 
                            batch_first=True, enforce_sorted=False)
    except Exception as e:
        print(seq_lens)
        raise e

    return packed_batch_inputs, torch.tensor(batch_labels) # return as tensors


if __name__ == '__main__':
    from torch.utils.data import Dataset, DataLoader
    train_dataset = MFCC_Dataset(features_dir='Data/train_features', csv_file_path='Data/train.csv', mode='train')
    valid_dataset = MFCC_Dataset(features_dir='Data/train_features', csv_file_path='Data/train.csv', mode='valid')
    test_dataset = MFCC_Dataset(features_dir='Data/test_features', csv_file_path=None, mode='test')
    # print(len(train_dataset.data_df))
    # print(len(valid_dataset.data_df))
    # print(len(os.listdir(test_dataset.features_dir)))
    BATCH_SIZE = 32
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, collate_fn=custom_collate_fn)

    for _, data in enumerate(train_loader):
        features, labels = data
        print(labels.size())
        input()
        
