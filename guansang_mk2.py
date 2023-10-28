import cv2
import mediapipe as mp
import math
from tqdm import tqdm
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# generator for face mesh
class face_mesh_generator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks = True, static_image_mode = True, max_num_faces = 1)
    
    # make face mesh
    def run_face_mesh(self, img_dir):
        image = cv2.imread(img_dir)
        results = self.face_mesh.process(image)
        return results.multi_face_landmarks[0].landmark
    
    # distance between two points
    def get_length(self, x,y,z):
        return math.sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2)

    # calculate every single length between the dots in face mesh
    def data_processing(self, img_dir):
        landmark_info = self.run_face_mesh(img_dir)
        cal_arr = list()
        for i in range(len(landmark_info)):
            one = landmark_info[i]
            temp_arr = list()
            for j in range(len(landmark_info)):
                two = landmark_info[j]
                temp_arr.append(self.get_length([one.x, two.x], [one.y, two.y], [one.z, two.z]))
            cal_arr.append(temp_arr)
        cal_arr = np.expand_dims(np.array(cal_arr), axis = 0)
        return cal_arr

# generating dataset
generator = face_mesh_generator()

base_dir = os.getcwd()
test_data_names = os.listdir(base_dir + '/test_img')

processed_data = list()
for img_name in tqdm(test_data_names):
    processed_data.append(generator.data_processing(base_dir + '/test_img/' + img_name))

processed_data = np.array(processed_data)

# torch dataset generator
class guansang_dataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        
    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index]
        
    def __len__(self): 
        return self.x_data.shape[0]

x_data = torch.tensor(processed_data, dtype = torch.float32)
y_data = [1]*len(x_data) # should be changed to actual labels
dataset = guansang_dataset(x_data, y_data)

guansang_dataloader = DataLoader(dataset, batch_size=1)


# guansang model  (needs update, shapes doesnt match)
# initial shape = (478,478)
#                 > 478 nodes in face mesh, length between every single nodes
#                 > column & row == nodes / values == length
# 1st conv layer =  in_chanel : 1 / out_chanel : 3 / kernel_size = 239(greatest devisor & x/2) / stride = 1 / padding = same / padding_mode = default(zero)
#                   > set the kernel_size in 239 expecting it could extract distinctive feature based on 1/4 of face
#                   > padding = same & padding_mode = zero, so that the layer can extract distictive feature while preserving the out line of the face
#                   > out_shape = (478,478)
# max_pooling = pool size = (2,2)
#               > guansnag aims to classify faces base on the orintal guansnag method. in order to do so, by extracting the max value expecting to extract the greatest cahricteristics form the out metrix
#               > as the in put data r in order, extract greater characteristic among near features (kernel == (2,2))
#               > out_shape = (239,239)
# 2nd conv layer = in chanel : 3 / out chanel : 3 / kernel_size = 64 / stride = 1 / padding = 2 / padding_mode = default(zero)
#                              > kernel_size as 64 expecting to extract more spesific features compared to 1st conv layer
#                              > stide == 1 for catching more detailded feature
#                              > padding = 2 & padding_method = zero to preserve the out line & to make the out put shape in even num
#                              > out_shape = (180,180)
# max_pooling = pool size : (10,10)
#               > extract features with wide point of view (kernel == (10,10))
#               > out_shape = (18,18)
# flatten = flatten the matrix
#           > out_shape = (3*18*18) (1d tensor)
# feed foward network = from 1d tensor (3*18*18) to out put (5) for sotmax function calculate the values
#                       > uses drop out layer to add some unsertainty (prevent overfitting) (drop out rate = 0.3)

class wtk_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1st = nn.Conv2d(1, 3, 239, padding = 'same')
        self.pool_1st = nn.MaxPool2d(2,2)
        self.conv_2nd = nn.Conv2d(3,3,64, padding = 2)
        self.pool_2nd = nn.MaxPool2d(10,10)
        self.ffn_1st = nn.Linear(int(3*18*18), int((3*18*18)/2))
        self.ffn_2nd = nn.Linear(int((3*18*18)/2), int((3*18*18)/4))
        self.ffn_3rd = nn.Linear(int((3*18*18)/4), 128)
        self.ffn_4th = nn.Linear(128, 64)
        self.ffn_5th = nn.Linear(63,32)
        self.ffn_fin = nn.Linear(32,5)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool_1st(F.relu(self.conv_1st(x)))
        x = self.pool_2nd(F.relu(self.conv_2nd(x)))
        x = torch.flatten(x,1)
        x = self.dropout(F.relu(self.ffn_1st(x)))
        x = self.dropout(F.relu(self.ffn_2nd(x)))
        x = self.dropout(F.relu(self.ffn_3rd(x)))
        x = self.dropout(F.relu(self.ffn_4th(x)))
        x = self.dropout(F.relu(self.ffn_5th(x)))
        x = self.ffn_fin(x)
        return x 
    
wtk = wtk_model()

# Training
loss_fc = nn.CrossEntropyLoss()
optimizer = optim.Adam(wtk.parameters(), lr = 1e-3)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(guansang_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = wtk(inputs)
        loss = loss_fc(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i != 0: # to prevent zero division error
            print(f'{epoch + 1}, mean_loss : {running_loss/i}')
            running_loss = 0.0


# predicting
test_input = iter(guansang_dataloader).next()
pred_probab = nn.Softmax(dim=1)(wtk(test_input[0]))