import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class MFCC_Dataset(Dataset):
    def __init__(self, features_dir, csv_file_path=None, train=False):
        super(MFCC_Dataset, self).__init__()
        self.features_dir = features_dir
        self.forTrain = train
        if csv_file_path:
            data_df = pd.read_csv(csv_file_path)
            self.one_hot_label_df = pd.get_dummies(data_df['scenario'])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        feature_filename = os.path.join(self.features_dir, '{0:06}.npy'.format(index))
        feature = np.load(feature_filename)
        if self.forTrain:
            label = self.one_hot_label_df.loc[index].values
            # feature: (20, timestep)
            # label: (10, )
            return feature, label
        else:
            return feature

if __name__ == '__main__':
    dataset = MFCC_Dataset(features_dir='Data/train_features', csv_file_path='Data/train.csv', train=True)
    print(dataset[0][0].shape, dataset[0][1].shape)
    dataset = MFCC_Dataset(features_dir='Data/test_features')
    print(dataset[0].shape)