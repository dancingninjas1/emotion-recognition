import numpy as np
import torch
import random
from training.model import MTSA
from preprocessing.audiofeature import AudioEmotionDataset


num_classes = 4  # Change this to the actual number of classes you have
model1 = MTSA(architecture='pons_won', num_class=num_classes, is_cuda=True)
model2 = MTSA(architecture='pons_won', num_class=num_classes, is_cuda=True)
model3 = MTSA(architecture='pons_won', num_class=num_classes, is_cuda=True)
model1.load_state_dict(torch.load('results/custom_model-pia.pth'))
model2.load_state_dict(torch.load('results/custom_model-aco.pth'))
model3.load_state_dict(torch.load('results/custom_model-ele.pth'))
model1.eval()
model2.eval()
model3.eval()

data_folder = 'C:/Users/apurv/Desktop/project/Data/test'
csv_file = 'Data/test/test.csv'
audio_dataset = AudioEmotionDataset(data_folder, csv_file)




random_idx = 1
file_name, mel_spectrogram, emotion_label = audio_dataset[random_idx]


# Convert the mel spectrogram to a tensor and add a batch dimension
input_tensor = mel_spectrogram.unsqueeze(0)

# Make predictions on the input tensor
with torch.no_grad():
    output1 = model1(input_tensor)
    predicted_class1 = torch.argmax(output1, dim=1)

# Make predictions on the input tensor
with torch.no_grad():
    output2 = model2(input_tensor)
    predicted_class2 = torch.argmax(output2, dim=1)

# Make predictions on the input tensor
with torch.no_grad():
    output3 = model3(input_tensor)
    predicted_class3 = torch.argmax(output3, dim=1)



EMOTION= ['aggressive', 'relaxed', 'happy', 'sad']  # Replace with your label classes
# Calculate the ensemble prediction by averaging the outputs of the models
ensemble_output = (output1 + output2 + output3) / 3
ensemble_predicted_class = torch.argmax(ensemble_output, dim=1)

# Print individual predicted classes
print('Predicted Classes:')
print(f'Model 1 Predicted Class: {EMOTION[predicted_class1.item()]}')
print(f'Model 2 Predicted Class: {EMOTION[predicted_class2.item()]}')
print(f'Model 3 Predicted Class: {EMOTION[predicted_class3.item()]}')

# Print the final ensemble predicted class

print(f'Ensemble Predicted Class: {EMOTION[ensemble_predicted_class.item()]}')
print('Annotated Emotion: ', EMOTION[emotion_label])
print('Prediction Scores:')
for label, score in zip(EMOTION, torch.softmax(ensemble_output, dim=1)[0]):
    print(f'{label}: {score:.4f}')

