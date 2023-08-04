# main.py
import torchaudio

from preprocessing.audiofeature import MelSpectrogramTransform
from preprocessing.audiofeature import AudioEmotionDataset
from preprocessing.audiofeature import split_dataset_folds
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from training.model import MTSA

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


# Initialize your custom model
custom_model = MTSA()

# Load the state_dict of the pre-trained model
pretrained_model_path = 'C:/Users/apurv/Desktop/project/pre-trained model/ponswon.pth'

pretrained_model_state = torch.load(pretrained_model_path, map_location=torch.device('cpu'))

# Get the state_dict of the custom model
custom_model_dict = custom_model.state_dict()

# Filter out the layers that exist in both models
pretrained_dict = {k: v for k, v in pretrained_model_state.items() if k in custom_model_dict}

# Update the state_dict of the custom model
custom_model_dict.update(pretrained_dict)

# Load the updated state_dict into the custom model
custom_model.load_state_dict(custom_model_dict)


# Modify the last layer for your task
num_classes = 4
custom_model.classifier = nn.Linear(custom_model.classifier.in_features, num_classes)

# Freeze all layers except the last one
for param in custom_model.parameters():
    param.requires_grad = False

# Allow the last layer to be trainable
custom_model.classifier.requires_grad = True

# Move the model to the appropriate device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(custom_model.parameters(), lr=0.001)  # Adjust learning rate as needed

# Set the number of epochs
num_epochs = 10  # Adjust the number of epochs as needed

torchaudio.set_audio_backend("soundfile")

file_path = 'C:/Users/apurv/Desktop/project/Data/piano/sad/284_sad_n1_i1_piano_AlbLin_20230126.wav'
try:
    waveform, sample_rate = torchaudio.load(file_path)
    print(f"Loaded audio file: {file_path}")
except Exception as e:
    print(f"Error loading file {file_path}: {e}")




# Training loop
for epoch in range(num_epochs):
    for phase in ['train', 'test']:
        if phase == 'train':
            custom_model.train()
        else:
            custom_model.eval()

        running_loss = 0.0
        corrects = 0

        data_loader = data_loaders[0][phase] if phase == 'train' else data_loaders[1][phase]

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = custom_model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_accuracy = corrects.double() / len(data_loader.dataset)

        print(f'{phase} - Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}')