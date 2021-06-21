import os
import librosa
from librosa import feature
import numpy as np
from tqdm import tqdm

DATA_DIR = 'Data'

def preprocess():
    wav_folders_list = ['train', 'test']
    for wav_folder in wav_folders_list:
        audio_dir = os.path.join(DATA_DIR, wav_folder)
        features_dir = os.path.join(DATA_DIR, wav_folder+'_features')
        if not os.path.exists( features_dir):
            os.makedirs(features_dir)
        for audio_file in tqdm(os.listdir(audio_dir), desc='{} features'.format(wav_folder)):
            y, sr = librosa.load(os.path.join(audio_dir, audio_file))
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            feature_filename = audio_file[:-4]
            np.save(os.path.join(features_dir, feature_filename), mfcc)
            
if __name__ == '__main__':
    preprocess()