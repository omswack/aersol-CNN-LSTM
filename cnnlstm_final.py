import torch
import os
import time
import numpy as np
import xarray as xr
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

#Parameters, you can change these. Sequence length is amound of timestamps in a sequence, each timestamp is 30 minutes
SEQUENCE_LEN = 72
LABEL_LEN = 144
HEIGHT = 25
WIDTH = 25

folder_path = '/project/dilkina_565/aerosol_data/2018/'
expected_timesteps = 48
shape_per_timestep = (25, 25, 1)

primer = np.empty((0, 25, 25, 1))

# Iterate through each month and day
for month in range(1, 13):
    month_folder = f'M{month:02d}'
    month_path = os.path.join(folder_path, month_folder)

    if not os.path.exists(month_path):
        continue

    for day in range(1, 32):
        day_file = f'day{day:02d}.npy'
        file_path = os.path.join(month_path, day_file)
        
        if not os.path.isfile(file_path):
            continue

        sequence = np.load(file_path)
        if sequence.shape[0] != expected_timesteps:
            print(f"Skipping file with unexpected number of timesteps: {file_path}")
            continue
        

        # Keep only the first feature along the last dimension for 25x25 files
        sequence_first_feature = sequence[:, :, :, 0:1]

        primer = np.concatenate((primer, sequence_first_feature), axis=0)

reshaped_primer = np.reshape(primer, (1, primer.shape[0], 25, 25, 1))
print (reshaped_primer.shape)
sequence_length = SEQUENCE_LEN
label_length = LABEL_LEN
total_length = SEQUENCE_LEN + LABEL_LEN
height = HEIGHT
width = WIDTH

sequences = []
labels = []






data = reshaped_primer.squeeze(0)

for start in range(0,len(data) - total_length + 1, sequence_length//4):
    end = start + total_length
    seq = data[start:start + sequence_length]
    label = data[start + sequence_length:end]
    sequences.append(seq)
    labels.append(label)

# Convert to numpy arrays
sequences = np.array(sequences)
labels = np.array(labels)

train_sequences, temp_sequences, train_labels, temp_labels = train_test_split(sequences, labels, test_size=0.3, random_state=42)

test_sequences, val_sequences, test_labels, val_labels = train_test_split( temp_sequences, temp_labels, test_size=0.5, random_state=42)

class CNNLSTMModel(nn.Module):
    def __init__(self, slider, lstm_hidden_size):
        super(CNNLSTMModel, self).__init__()
        self.slider = slider

        # Convolutional Layer
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=32 * 23 * 23, hidden_size=lstm_hidden_size, batch_first=True)

        self.relu = nn.ReLU()

        # Fully Connected Layer
        self.fc_layer = nn.Linear(lstm_hidden_size * sequence_length, label_length * height * width)


    def forward(self, x):
        batch_size, sequence_length, channels, height, width = x.size()
    
        # Convolutional layers
        x = x.view(batch_size * sequence_length, channels, height, width)
        x = self.conv_layer(x)
    
        # Reshape for LSTM
        conv_output_height, conv_output_width = x.size(-2), x.size(-1)
        x = x.view(batch_size, sequence_length, -1)
    
        # LSTM layer
        x, _ = self.lstm(x)
    
        # Flatten the output for the fully connected layer
        x = x.contiguous().view(batch_size, -1)
    
        # Fully connected layer
        x = self.fc_layer(x)
    
        # Reshape to get the final output in the shape [batch_size, label_length, height, width]
        x = x.view(batch_size, label_length, height, width)
    
        return x

class CDF(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        sequence_tensor = torch.from_numpy(sequence).float()
        label_tensor = torch.from_numpy(label).float()
        # print(sequence_tensor.shape)
        sequence_tensor = sequence_tensor.permute(0, 3, 1, 2)
        label_tensor = label_tensor.permute(0, 3, 1, 2)
        return sequence_tensor, label_tensor

def validate_model(model, val_loader, loss_function):
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    with torch.no_grad():  # No need to track gradients during validation
        for val_sequences, val_labels in val_loader:
            val_sequences, val_labels = val_sequences.to(device), val_labels.to(device)
            val_labels = val_labels.squeeze(2)
            predictions = model(val_sequences)
            val_loss = loss_function(predictions, val_labels)
            total_val_loss += val_loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss

def test_model(model, test_loader, loss_function):
    model.eval()  # Set the model to evaluation mode
    total_test_loss = 0
    with torch.no_grad():  # No need to track gradients during testing
        for test_sequences, test_labels in test_loader:
            test_sequences, test_labels = test_sequences.to(device), test_labels.to(device)
            test_labels = test_labels.squeeze(2)

            predictions = model(test_sequences)
            test_loss = loss_function(predictions, test_labels)
            total_test_loss += test_loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')
    return avg_test_loss
    


if __name__ == '__main__':
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    # Create an instance of the model
    model = CNNLSTMModel(slider=1, lstm_hidden_size=32)  # Adjust the slider and hidden size as needed
    model.to(device)
    # loss function
    lf = nn.MSELoss()

    learning_rate = 0.001

    # optimization algorithm (stochastic gradient descent)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = CDF(train_sequences, train_labels)
    val_dataset = CDF(val_sequences, val_labels)
    test_dataset = CDF(test_sequences, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(len(train_dataset))

    num_epochs = 10  # Modify the number of epochs as needed

    '''
    within the epoc we need to split the tensor from data_loader to get the train test split i think)
    '''
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for batch_idx, (sequence_tensor, label_tensor) in enumerate(train_loader):
            sequence_tensor, label_tensor = sequence_tensor.to(device), label_tensor.to(device)
            optimizer.zero_grad()
            # print(sequence_tensor[0].shape)
            predictions = model(sequence_tensor)
            #print(predictions)
            #target = target.view(-1, 1)
            label_tensor = label_tensor.squeeze(2)

            loss = lf(predictions, label_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}], Loss: {loss.item():.4f}')
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}] finished with average loss: {avg_loss:.4f}')
        val_loss = validate_model(model, val_loader, lf)
        print(f'Epoch [{epoch + 1}/{num_epochs}] finished with validation loss: {val_loss:.4f}')

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}], Loss: {loss.item():.4f}')

    test_model(model, test_loader, lf)