# main.py

from preprocessing.audiofeature import MelSpectrogramTransform
from preprocessing.audiofeature import AudioEmotionDataset
from preprocessing.audiofeature import split_dataset_folds
from torch.utils.data import DataLoader

# Define data folder and CSV file path
data_folder = 'Data'
csv_file = 'Data/piano/annotations_piano.csv'

# Define the MelSpectrogramTransform instance
mel_transform = MelSpectrogramTransform()

# Create an instance of AudioEmotionDataset
audio_dataset = AudioEmotionDataset(data_folder, csv_file, transform=mel_transform)

# Perform fold splits
fold_splits = split_dataset_folds(audio_dataset, num_folds=5)

# Create data loaders for each fold
data_loaders = []
for fold in fold_splits:
    train_dataset = AudioEmotionDataset(data_folder, csv_file, transform=mel_transform)
    train_dataset.data = fold['train']
    test_dataset = AudioEmotionDataset(data_folder, csv_file, transform=mel_transform)
    test_dataset.data = fold['test']

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    data_loaders.append({'train': train_loader, 'test': test_loader})

# Print the split information
for i, fold in enumerate(data_loaders, 1):
    print(f"Fold {i}:")
    print(f"Train data: {len(fold['train'].dataset)} samples")
    print(f"Test data: {len(fold['test'].dataset)} samples")
    print("-" * 30)