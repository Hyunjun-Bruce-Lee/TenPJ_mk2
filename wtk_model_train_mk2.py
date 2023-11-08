# model too heavy for light sail server

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import tqdm
import numpy as np
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

base_dir = os.getcwd()

with open(f'{base_dir}/train_dataset.bin', 'rb') as f:
    loaded_data = pickle.load(f)

loaded_data['labels'] = [i-1 for i in loaded_data['labels']]

wtk_data = np.array(loaded_data['data'])
wtk_labels = np.array(loaded_data['labels'])

wtk_data.shape
wtk_labels.shape

# torch dataset generator
class guansang_dataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        
    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index]
        
    def __len__(self): 
        return self.x_data.shape[0]

x_data = torch.tensor(wtk_data, dtype = torch.float32)
y_data = wtk_labels # should be changed to actual labels
dataset = guansang_dataset(x_data, y_data)

guansang_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

next(iter(guansang_dataloader))

# you can use this formula [(W−K+2P)/S]+1.
# 
# W is the input volume - in your case 128
# K is the Kernel size - in your case 5
# P is the padding - in your case 0 i believe
# S is the stride - which you have not provided.


class wtk_model(nn.Module):
    def __init__(self):
        super().__init__()        # input shape (478,478)
        self.conv_1st = nn.Conv2d(1, 3, 2**8, padding = 10)   # out shape (243,243)
        self.pool_1st = nn.MaxPool2d((9,9),stride = 2) # out shape (118,118)
        self.conv_2nd = nn.Conv2d(3,3,2**6, padding = 8) # out shape (65,65)
        self.pool_2nd = nn.MaxPool2d((5,5), stride = 5) # out shape (13,13)
        self.ffn_1st = nn.Linear(3*13*13, 2**7)
        self.ffn_2nd = nn.Linear(2**7, 2**5)
        self.ffn_fin = nn.Linear(2**5, 4)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool_1st(F.relu(self.conv_1st(x)))
        x = self.pool_2nd(F.relu(self.conv_2nd(x)))
        x = torch.flatten(x,1)
        x = self.dropout(F.relu(self.ffn_1st(x)))
        x = self.dropout(F.relu(self.ffn_2nd(x)))
        x = self.ffn_fin(x)
        return x 
    
wtk = wtk_model().to(device)


# Training
# loss_fc = nn.CrossEntropyLoss().cuda() # when using gpu
loss_fc = nn.CrossEntropyLoss()
optimizer = optim.Adam(wtk.parameters(), lr = 1e-3)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(guansang_dataloader, 0):
        inputs, labels = data
        # labels = labels.type(torch.LongTensor) # casting to long tensor (when using gpu int is not implemented)
        # inputs, labels = inputs.to(device), labels.to(device) # only when using gpu
        optimizer.zero_grad()
        outputs = wtk(inputs)
        loss = loss_fc(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i != 0: # to prevent zero division error
            print(f'{epoch + 1}, mean_loss : {running_loss/i}')
            running_loss = 0.0


# predicting test
test_input = np.expand_dims(wtk_data[0], 0)
test_input = torch.tensor(test_input, dtype = torch.float32)
test_input = test_input.to(device)
pred_probab = nn.Softmax(dim=1)(wtk(test_input))
pred_probab.argmax(1)

# from gpu model to cpu
# torch.save(model.state_dict(), PATH)
# model.load_state_dict(torch.load(PATH, map_location=device))