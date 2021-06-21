from preprocess import DATA_DIR
import warnings
warnings.filterwarnings('ignore')
from model import MFCC_Dataset

DATA_DIR = 'Data'

def main():
    dataset = MFCC_Dataset(features_dir='Data/train_features', csv_file_path='Data/train.csv', train=True)
    print(dataset[0][0].shape, dataset[0][1].shape)
    dataset = MFCC_Dataset(features_dir='Data/test_features')
    print(dataset[0].shape)

if __name__ == '__main__':
    main()