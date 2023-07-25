import torchaudio
import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

class MelSpectrogramTransform:
    def __init__(self, sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, waveform):
        # Convert waveform to torch tensor
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32)

        # Compute the mel spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )(waveform_tensor)

        # Convert the mel spectrogram to a logarithmic scale (dB)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        # Normalize the mel spectrogram to a range [0, 1]
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

        return mel_spec_normalized

# Usage example:
# Create an instance of MelSpectrogramTransform
mel_transform = MelSpectrogramTransform()


# Define emotion labels (modify as needed)
EMOTIONS = ['aggressive', 'relaxed', 'happy', 'sad']

class AudioEmotionDataset(Dataset):
    def __init__(self, data_folder, csv_file, transform=None):
        self.data_folder = data_folder
        self.transform = transform

        # Load data and emotion labels from CSV file
        self.data, self.labels = self._load_data_from_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load audio data and emotion label for the given index
        file_path = os.path.join(self.data_folder, self.data[idx])
        waveform, sample_rate = torchaudio.load(file_path)

        # Apply audio transformations (e.g., compute mel spectrogram)
        if self.transform:
            mel_spectrogram = self.transform(waveform)
        else:
            mel_spectrogram = waveform  # Replace with appropriate audio preprocessing if no transform

        # Convert emotion label to numerical index
        emotion_label = EMOTIONS.index(self.labels[idx])

        return mel_spectrogram, emotion_label

    def _load_data_from_csv(self, csv_file):
      data = []
      labels = []
      df = pd.read_csv(csv_file)
      #print(df)
      for _, row in df.iterrows():
          # Assuming the file_name is in the 'file_name' column and the emotion is in the 'emotion' column
          file_name = row['file_name']
          emotion = row['emotion']
          data.append(file_name)
          labels.append(emotion)
      return data, labels


def split_dataset_folds(dataset, num_folds=5, random_state=None):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=random_state)

    data = dataset.data
    labels = dataset.labels

    fold_splits = []
    for train_index, test_index in skf.split(data, labels):
        train_data = [data[idx] for idx in train_index]
        test_data = [data[idx] for idx in test_index]

        fold_splits.append({'train': train_data, 'test': test_data})

    return fold_splits

# Usage example:
# Assuming you have already created the AudioEmotionDataset instance 'audio_dataset'
# and defined the MelSpectrogramTransform 'mel_transform'
audio_dataset = AudioEmotionDataset('C:/Users/apurv/Desktop/project/Data/piano', 'C:/Users/apurv/Desktop/project/Data/piano/annotations_piano.csv', transform=mel_transform)
fold_splits = split_dataset_folds(audio_dataset)

