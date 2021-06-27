import os
import pandas as pd
import re
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset

DATA_DIR = '../Data'
TRAIN_RATIO = 0.8

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '
    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')

text_transform = TextTransform()


train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)
valid_audio_transforms = torchaudio.transforms.MelSpectrogram()


def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (melspec, utterance) in data:
        if data_type == 'train':
            # spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            spec = melspec.squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            # spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            spec = melspec.squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths

class Audio_Dataset(Dataset):
    def __init__(self, csv_file_path, mode='train'):
        super(Audio_Dataset, self).__init__()
        self.mode = mode
        df = pd.read_csv(csv_file_path)
        # ''' Shuffle DataFrame !!'''
        # df = df.sample(frac=1).reset_index(drop=True)

        assert self.mode in ['train', 'valid'] # only for train and valid
        if self.mode == 'train':
            self.data_df = df[:int(len(df)*TRAIN_RATIO)]
        
        elif self.mode == 'valid':
            self.data_df = df[int(len(df)*TRAIN_RATIO):]

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # audio_filename = os.path.join('../Data/train', '{0:06}.wav'.format(self.data_df.iloc[index]['file']))
        # audio, _ = torchaudio.load(audio_filename)
        melspec = torch.load('Features/train/{:06}.pt'.format(self.data_df.iloc[index]['file']))
        label = self.data_df.iloc[index]['sentence']
        label = re.sub(r'[,:;\.\(\)-/"<>@_]', ' ', label)
        label = re.sub(r'[0-9]', ' ', label)

        
        return torch.tensor(melspec), label

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")


    train_dataset = Audio_Dataset(csv_file_path='../Data/train.csv', mode='train')
    valid_dataset = Audio_Dataset(csv_file_path='../Data/train.csv', mode='valid')

    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride":2,
        "dropout": 0.1,
        "learning_rate": 5e-4,
        "batch_size": 20,
        "epochs": 10
    }

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, 'train'),
                                **kwargs)
    valid_loader = DataLoader(dataset=valid_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid'),
                                **kwargs)

    
    for data in train_loader:
        print()
        print()
