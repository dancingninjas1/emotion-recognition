# main.py
import torchaudio
from sklearn.model_selection import KFold

from preprocessing.audiofeature import AudioEmotionDataset
from preprocessing.audiofeature import split_dataset_folds
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing.foldsplit_instrument import fold_instrument
from preprocessing.multiple import AudioEmotionDatasetMultiple
from training.model import MTSA

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

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")  # Print the name of the GPU
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU")
# Move the model to the appropriate device (GPU/CPU)
# device = torch.device("cuda")
custom_model.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Only optimize parameters with requires_grad=True
optimizer = optim.Adam(filter(lambda p: p.requires_grad, custom_model.parameters()), lr=0.00001)

#Define data folder and CSV file path
data_folder = 'C:/Users/apurv/Desktop/project/Data'
csv_file = 'Data/acoustic-guitar/annotations_acoustic-guitar.csv'

# Create an instance of AudioEmotionDataset
audio_dataset = AudioEmotionDataset(data_folder, csv_file)

# Get the data and labels from the dataset
data = audio_dataset.data
labels = audio_dataset.labels

# Define the fold ranges as specified
instrument = "acoustic_guitar"
fold_ranges = fold_instrument(instrument)

fold_splits = split_dataset_folds(data, labels, fold_ranges)

# Print the number of samples in each fold
for fold_idx, fold_data in enumerate(fold_splits, start=1):
    num_samples = len(fold_data['data'])
    print(f"Fold {fold_idx}: {num_samples} samples")

# Example usage
folds = [0, 1, 2, 3]

# Training loop

# Set the number of epochs
num_epochs = 1

# Training loop using K-fold cross-validation
for epoch in range(num_epochs):

    for i in range(1, len(folds)):
        train_folds = folds[:i]
        test_folds = folds[i:]

        train_dataset = AudioEmotionDataset(data_folder, csv_file, fold_splits, fold_indices=train_folds)
        test_dataset = AudioEmotionDataset(data_folder, csv_file, fold_splits, fold_indices=test_folds)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

        for phase in ['train', 'test']:
            if phase == 'train':
                custom_model.train()
                data_loader = train_loader
            else:
                custom_model.eval()
                data_loader = test_loader

            running_loss = 0.0
            corrects = 0

            for file_name, inputs, labels in data_loader:
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

            print(
                f'Epoch {epoch + 1}, Train Fold {train_folds}, Test Fold {test_folds}, {phase} - Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}')


# Save the trained model
save_path = 'C:/Users/apurv/Desktop/project/custom_model_ac_guitar.pth'
torch.save(custom_model.state_dict(), save_path)
print(f"Trained model saved to {save_path}")

print("Unfreezing all layers")
# Unfreeze all layers of the custom model
for param in custom_model.parameters():
    param.requires_grad = True

# Move the model to the appropriate device (GPU/CPU)
custom_model.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer for the entire model
optimizer = optim.Adam(custom_model.parameters(), lr=0.000001)
data_folder = 'C:/Users/apurv/Desktop/project/Data'
csv_files = ['Data/acoustic-guitar/annotations_acoustic-guitar.csv', 'Data/piano/annotations_piano.csv',
             'Data/electric-guitar/annotations_electric-guitar.csv']

audio_datasets = AudioEmotionDatasetMultiple(data_folder, csv_files)

# Get the data and labels from the dataset
data = audio_datasets.data
labels = audio_datasets.labels

print(len(data))
# Define the fold ranges as specified
instrument = "all"
fold_ranges = fold_instrument(instrument)

fold_splits = split_dataset_folds(data, labels, fold_ranges)

# Print the number of samples in each fold
for fold_idx, fold_data in enumerate(fold_splits, start=1):
    num_samples = len(fold_data['data'])
    print(f"Fold {fold_idx}: {num_samples} samples")

# Set the number of epochs for re-training
num_epochs = 1
folds = [0, 1, 2, 3, 4, 5, 6, 7]
print("Training for 5 epochs......")
for epoch in range(num_epochs):

    for i in range(1, len(folds)):
        train_folds = folds[:i]
        test_folds = folds[i:]

        train_dataset = AudioEmotionDatasetMultiple(data_folder, csv_files, fold_splits, fold_indices=train_folds)
        test_dataset = AudioEmotionDatasetMultiple(data_folder, csv_files, fold_splits, fold_indices=test_folds)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

        for phase in ['train', 'test']:
            if phase == 'train':
                custom_model.train()
                data_loader = train_loader  # Use train_loader for training
            else:
                custom_model.eval()
                data_loader = test_loader  # Use test_loader for testing

            running_loss = 0.0
            corrects = 0

            for file_name, inputs, labels in data_loader:
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

            print(
                f'Epoch {epoch + 1}, Train Fold {train_folds}, Test Fold {test_folds}, {phase} - Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}')

# Save the re-trained model
retrained_model_path = 'C:/Users/apurv/Desktop/project/retrained_model_piano.pth'
torch.save(custom_model.state_dict(), retrained_model_path)
print(f"Retrained model saved to {retrained_model_path}")
