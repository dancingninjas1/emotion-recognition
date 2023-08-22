# main.py
import torchaudio
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR

from preprocessing.audiofeature import AudioEmotionDataset
from preprocessing.audiofeature import split_dataset_folds
from torch.utils.data import DataLoader
import torch
import csv
import torch.nn as nn
import torch.optim as optim
import torchviz

from preprocessing.foldsplit_instrument import fold_instrument
from preprocessing.multiple import AudioEmotionDatasetMultiple
from training.model import MTSA
from sklearn.metrics import precision_score, recall_score, f1_score

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

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")  # Print the name of the GPU
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU")

custom_model.to(device)

num_classes = 4
custom_model.classifier = nn.Linear(custom_model.classifier.in_features, num_classes)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Selectively unfreeze specific layers
for name, param in custom_model.named_parameters():
    if 'classifier' in name or 'encoder.layer.1.attention.self' in name or 'encoder.layer.0.attention.self' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Only optimize parameters with requires_grad=True
optimizer = optim.Adam(filter(lambda p: p.requires_grad, custom_model.parameters()), lr=0.00001)


# Define data folder and CSV file path
data_folder = 'C:/Users/apurv/Desktop/project/Data'
csv_file = 'Data/electric-guitar/annotations_electric-guitar.csv'

# Create an instance of AudioEmotionDataset
audio_dataset = AudioEmotionDataset(data_folder, csv_file)

# Get the data and labels from the dataset
data = audio_dataset.data
labels = audio_dataset.labels

# Define the fold ranges as specified
instrument = "electric_guitar"
fold_ranges = fold_instrument(instrument)

fold_splits = split_dataset_folds(data, labels, fold_ranges)

# Print the number of samples in each fold
for fold_idx, fold_data in enumerate(fold_splits, start=1):
    num_samples = len(fold_data['data'])
    print(f"Fold {fold_idx}: {num_samples} samples")

# Example usage
folds = [0, 1, 2, 3, 4]

# Training loop

# Set the number of epochs
num_epochs = 18
performance_metrics = []
# Training loop using K-fold cross-validation
for epoch in range(num_epochs):

    for i in range(len(folds)):
        test_folds = [folds[i]]
        train_folds = folds[:i] + folds[i + 1:]

        all_preds = []
        all_labels = []

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
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_accuracy = corrects.double() / len(data_loader.dataset)

            epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
            epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

            print(
                f'Epoch {epoch + 1}, Train Fold {train_folds}, Test Fold {test_folds}, {phase} - '
                f'Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, '
                f'Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}')

            epoch_metrics = {
                'epoch': epoch + 1,
                'train_folds': train_folds,
                'test_folds': test_folds,
                'phase': phase,
                'loss': epoch_loss,
                'accuracy': epoch_accuracy,
                'precision': epoch_precision,
                'recall': epoch_recall,
                'f1': epoch_f1,
            }
            performance_metrics.append(epoch_metrics)

# Save the trained model
save_path = '/results/custom_model-ele.pth'
torch.save(custom_model.state_dict(), save_path)
print(f"Trained model saved to {save_path}")

csv_filename = 'results/performance_metrics-ele-report.csv'

# Define the header for the CSV file
csv_header = ['epoch', 'train_folds', 'test_folds', 'phase', 'loss', 'accuracy', 'precision', 'recall', 'f1']

# Write the data to the CSV file
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=csv_header)
    writer.writeheader()
    writer.writerows(performance_metrics)

print(f'Performance metrics saved to {csv_filename}')

print("Unfreezing all layers")
# Unfreeze all layers of the custom model
for param in custom_model.parameters():
    param.requires_grad = True

# Move the model to the appropriate device (GPU/CPU)
custom_model.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer for the entire model
optimizer = optim.Adam(custom_model.parameters(), lr=0.00001)

# Decay the learning rate by a factor of 0.1 every epoch
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

# Print the number of samples in each fold
for fold_idx, fold_data in enumerate(fold_splits, start=1):
    num_samples = len(fold_data['data'])
    print(f"Fold {fold_idx}: {num_samples} samples")

performance_metrics = []
# Set the number of epochs for re-training
num_epochs = 12



for epoch in range(num_epochs):

    for i in range(len(folds)):
        test_folds = [folds[i]]
        train_folds = folds[:i] + folds[i + 1:]

        all_preds = []
        all_labels = []

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
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_accuracy = corrects.double() / len(data_loader.dataset)

            epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
            epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

            print(
                f'Epoch {epoch + 1}, Train Fold {train_folds}, Test Fold {test_folds}, {phase} - '
                f'Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, '
                f'Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}')

            epoch_metrics = {
                'epoch': epoch + 1,
                'train_folds': train_folds,
                'test_folds': test_folds,
                'phase': phase,
                'loss': epoch_loss,
                'accuracy': epoch_accuracy,
                'precision': epoch_precision,
                'recall': epoch_recall,
                'f1': epoch_f1,
            }
            performance_metrics.append(epoch_metrics)

csv_filename = 'results/performance_metrics-electric-report-full.csv'

save_path = '/results/custom_model-ele.pth'
torch.save(custom_model.state_dict(), save_path)
print(f"Trained model saved to {save_path}")

# Write the data to the CSV file
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=csv_header)
    writer.writeheader()
    writer.writerows(performance_metrics)

print(f'Performance metrics saved to {csv_filename}')
