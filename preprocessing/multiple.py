import torch
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import librosa
import numpy as np

EMOTIONS = ['aggressive', 'relaxed', 'happy', 'sad']
data_folder = 'C:/Users/apurv/Desktop/project/Data'
csv_file = 'Data/acoustic-guitar/aannotations_acoustic-guitar.csv'


class AudioEmotionDatasetMultiple(Dataset):
    def __init__(self, data_folder, csv_files, fold_splits=None, fold_indices=None, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.fs = 16000
        self.window = 2048
        self.hop = 512
        self.mel = 512

        self.data = []
        self.labels = []

        for csv_file in csv_files:
            data, labels = self._load_data_from_csv(csv_file)
            self.data.extend(data)
            self.labels.extend(labels)

        if fold_splits is not None:
            combined_data = []
            combined_labels = []
            for fold_idx in fold_indices:
                if fold_idx < len(fold_splits):
                    fold_data = fold_splits[fold_idx]['data']
                    fold_labels = fold_splits[fold_idx]['labels']
                    combined_data.extend(fold_data)
                    combined_labels.extend(fold_labels)
            self.data = combined_data
            self.labels = combined_labels

    def _load_data_from_csv(self, csv_file):
        data = []
        labels = []
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            file_name = row['file_name']
            emotion = row['emotion']
            data.append(file_name)
            labels.append(emotion)
        return data, labels

    def __len__(self):
        return len(self.data)

    def filter_data_by_emotion(self, emotions):
        filtered_indices = [idx for idx, label in enumerate(self.labels) if label in emotions]
        self.data = [self.data[idx] for idx in filtered_indices]
        self.labels = [self.labels[idx] for idx in filtered_indices]

    def _compute_mel_spectrogram(self, file_path):
        x, sr = librosa.load(file_path, sr=16000, mono=True)

        # Compute mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=x, sr=self.fs, n_fft=self.window, hop_length=self.hop, n_mels=self.mel)
        mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram)

        max_length = 256
        if mel_spectrogram.shape[1] > max_length:
            mel_spectrogram = mel_spectrogram[:, :max_length]
        else:
            padding = max_length - mel_spectrogram.shape[1]
            mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, padding)))

        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)

        return mel_spectrogram

    def __getitem__(self, idx):
        file_name = self.data[idx]
        emotion = self.labels[idx]
        instruments = ['acoustic-guitar', 'piano', 'electric-guitar']
        instrument_type = 'acoustic-guitar'  # Default value

        # Determine the instrument type based on the file name
        for inst in instruments:
            if inst in file_name:
                instrument_type = inst
                break

        file_path = os.path.join(self.data_folder, instrument_type, emotion, f'{file_name}.wav')

        try:
            mel_spectrogram = self._compute_mel_spectrogram(file_path)

            # Convert emotion label to numerical index
            emotion_label = EMOTIONS.index(emotion)

            return file_name, mel_spectrogram, emotion_label

        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return torch.zeros((1, 96, 1000)), EMOTIONS.index(emotion)
