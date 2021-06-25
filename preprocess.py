import warnings
warnings.filterwarnings('ignore')
import os
import librosa
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import csv
import math

DATA_DIR = 'Data'
N_MFCC = 12
MODE_TYPE = ['train', 'test']

def estimate_pitch(segment, sr, fmin=50.0, fmax=2000.0):
    ''' 
        estimate pitch from audio segment
        ref: https://musicinformationretrieval.com/pitch_transcription_exercise.html 
    '''
    # Compute autocorrelation of input segment.
    r = librosa.autocorrelate(segment)

    # Define lower and upper limits for the autocorrelation argmax.
    i_min = sr/fmax
    i_max = sr/fmin
    r[:int(i_min)] = 0
    r[int(i_max):] = 0
    
    # Find the location of the maximum autocorrelation.
    i = r.argmax()
    f0 = float(sr)/i+1e-9
    return f0

def detect_pitch():
    ''' 
        detect segment pitch and write to Data/{}_pitches.csv 
    '''
    fmin, fmax = 150, 400
    
    for wav_folder in MODE_TYPE:
        audio_dir = os.path.join(DATA_DIR, wav_folder)
        pitch_dict = {}
        audio_num = len(os.listdir(audio_dir))
        for idx in tqdm(range(audio_num), desc='{} features'.format(wav_folder)):
            audio_filename =  '{:06}.wav'.format(idx)
            
            # y, sr = librosa.load(os.path.join(audio_dir, audio_file))
            y, sr = librosa.load(os.path.join(audio_dir, '{:06}.wav'.format(idx)))

            sos = butter(N=50, Wn=fmax, btype='lowpass', fs=sr, output='sos')
            y_filterd = sosfiltfilt(sos, y)          
            ''' ref: https://www.geeksforgeeks.org/noise-removal-using-lowpass-digital-butterworth-filter-in-scipy-python/ '''
            pitch = estimate_pitch(segment=y_filterd, sr=sr, fmin=fmin, fmax=fmax)
            pitch_dict[os.path.join(audio_dir, audio_filename)] = pitch
        
        with open(os.path.join(DATA_DIR, '{}_pitches.csv'.format(wav_folder)), 'w') as csvfile:
            writer = csv.writer(csvfile)
            for key, value in pitch_dict.items():
                writer.writerow([key, value])

def normalize_pitch():
    '''
        shift all audio files to the same pitch
    '''
    train_pitch_df = pd.read_csv(os.path.join(DATA_DIR, 'train_pitches.csv'))
    test_pitch_df = pd.read_csv(os.path.join(DATA_DIR, 'test_pitches.csv'))
    train_pitch_arr = train_pitch_df.iloc[:,1].values
    test_pitch_arr = test_pitch_df.iloc[:,1].values
    pitch_arr = np.concatenate((train_pitch_arr, test_pitch_arr), axis=0)
    pitch_arr = pitch_arr[np.where(pitch_arr != np.inf)]
    avg_pitch = pitch_arr.mean()
    '''
    shifted_audio_dir = os.path.join(DATA_DIR, 'train_shifted') 
    if not os.path.exists(os.path.join(DATA_DIR, 'train_shifted')):
        os.makedirs(shifted_audio_dir)
    for idx in tqdm(range(len(train_pitch_df)), desc='Training Data'):
        pitch_to_shift = train_pitch_df.iloc[idx, 1]
        in_file_path = train_pitch_df.iloc[idx, 0]  
        out_file_path = os.path.join(shifted_audio_dir, in_file_path[:-4].split('/')[-1]+'_shifted.wav')
        if pitch_to_shift != np.inf:
            cents_offset = 1200*math.log(pitch_to_shift/avg_pitch, 2)
            os.system('sox {infile} {outfile} pitch {shift}'.format(infile=in_file_path, outfile=out_file_path, shift=cents_offset))
        else:
            os.system('cp {infile} {outfile}'.format(infile=in_file_path, outfile=out_file_path))
    '''
    shifted_audio_dir = os.path.join(DATA_DIR, 'test_shifted')
    if not os.path.exists(os.path.join(DATA_DIR, 'test_shifted')):
        os.makedirs(shifted_audio_dir)
    for idx in tqdm(range(len(test_pitch_df)), desc='Testing Data'):
        pitch_to_shift = test_pitch_df.iloc[idx, 1]

        in_file_path = test_pitch_df.iloc[idx, 0]   
        out_file_path = os.path.join(shifted_audio_dir, in_file_path[:-4].split('/')[-1]+'_shifted.wav')
        if pitch_to_shift != np.inf:
            cents_offset = 1200*math.log(pitch_to_shift/avg_pitch, 2)
            os.system('sox {infile} {outfile} pitch {shift}'.format(infile=in_file_path, outfile=out_file_path, shift=cents_offset))
        else:
            os.system('cp {infile} {outfile}'.format(infile=in_file_path, outfile=out_file_path))


def preprocess():
    '''
        get MFCC from all (preprocessed) audio files
    '''
    for wav_folder in MODE_TYPE:
        audio_dir = os.path.join(DATA_DIR, wav_folder+'_shifted')
        features_dir = os.path.join(DATA_DIR, wav_folder+'_features')
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)
        len_dict = {}
        for audio_file in tqdm(os.listdir(audio_dir), desc='{} features'.format(wav_folder)):
            feature_filename = audio_file[:-4]
            
            y, sr = librosa.load(os.path.join(audio_dir, audio_file))
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
            np.save(os.path.join(features_dir, feature_filename), features)
            len_dict[os.path.join(features_dir, feature_filename)] = mfcc.shape[1]

        with open(os.path.join(DATA_DIR, '{}_details.csv'.format(wav_folder)), 'w') as csvfile:
            writer = csv.writer(csvfile)
            for key, value in len_dict.items():
                writer.writerow([key, value])

if __name__ == '__main__':
    # detect_pitch()
    # normalize_pitch()
    preprocess()