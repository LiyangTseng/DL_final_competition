import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from tqdm import tqdm
import torchaudio
import torch

train_length = 18052
test_length = 4721

audio_transforms = torchaudio.transforms.MelSpectrogram()

''' train data '''
features_dir = 'Features/train'
if not os.path.exists(features_dir):
    os.makedirs(features_dir)

for index in tqdm(range(train_length), desc='Train'):
    audio_filename = os.path.join('../Data/train', '{0:06}.wav'.format(index))
    audio, _ = torchaudio.load(audio_filename)
    spec = audio_transforms(audio)
    torch.save(spec, os.path.join(features_dir, '{:06}.pt'.format(index)))    

''' test data '''
features_dir = 'Features/test'
if not os.path.exists(features_dir):
    os.makedirs(features_dir)

for index in tqdm(range(test_length), desc='Test'):
    audio_filename = os.path.join('../Data/test', '{0:06}.wav'.format(index))
    audio, _ = torchaudio.load(audio_filename)
    spec = audio_transforms(audio)
    torch.save(spec, os.path.join(features_dir, '{:06}.pt'.format(index)))    

