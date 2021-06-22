import warnings
warnings.filterwarnings('ignore')
from dataset import MFCC_Dataset, custom_collate_fn
from model import LSTM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

DATA_DIR = 'Data'
BATCH_SIZE = 64
EPOCH_NUM = 1
LR = 1e-5

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device: ', device)

    train_dataset = MFCC_Dataset(features_dir='Data/train_features', csv_file_path='Data/train.csv', mode='train')
    valid_dataset = MFCC_Dataset(features_dir='Data/train_features', csv_file_path='Data/train.csv', mode='valid')
    test_dataset = MFCC_Dataset(features_dir='Data/test_features', csv_file_path=None, mode='test')
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, collate_fn=custom_collate_fn)
    
    model = LSTM(hidden_size=30, category_num=10, bidirectional=True, device=device).to(device).to(device)
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)
    loss_function = nn.CrossEntropyLoss()

    for epoch_idx in tqdm(range(EPOCH_NUM)):
        model.train()
        for _, data in enumerate(train_loader):
            optimizer.zero_grad()

            packed_features, labels = data
            packed_features, labels = packed_features.to(device), labels.to(device)
            
            output = model(packed_features)
            
if __name__ == '__main__':
    main()