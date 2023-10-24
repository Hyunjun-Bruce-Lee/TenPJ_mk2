import cv2
import mediapipe as mp
import math
from tqdm import tqdm

# pip3 install torch torchvision torchaudio v(2.1.0)

### data generator for training data
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
        return cal_arr


generator = face_mesh_generator()
image_dir = '/Users/hyunjun_bruce_lee/Documents/GIT/TenPJ_mk2/test_img/41e5d467-09f5-485a-b241-8f9f860a3b0f.jpg'
test = generator.data_processing(image_dir)
image_dir = 'test_img/news-p.v1.20230829.e92fb8ed6d954b70b9b621b20b8ab443_P1.jpg'
test2 = generator.data_processing(image_dir)

test # 478*478 matrix


import os
import numpy as np

base_dir = os.getcwd()
test_data_names = os.listdir(base_dir + 'test_img')

processed_data = list()
for img_name in tqdm(test_data_names):
    processed_data.append(generator.data_processing(base_dir + 'test_img/' + img_name))

processed_data = np.array(processed_data)

processed_data.shape


### dummy model with pytorch for testing !!!
### 1st time building model with pytorch lol
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class guansang_temp_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(478*478, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,5)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = guansang_temp_model().to(device)
print(model)

test_data = processed_data[0]
test_data = np.expand_dims(test_data, axis = 0)
X = torch.tensor(test_data, device=device, dtype = torch.float32)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)


# bellow method doesnt have batch processing, it may lack in memory use (only use for test run)
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class guansang_dataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        
    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index]
        
    def __len__(self): 
        return self.x_data.shape[0]

processed_data = torch.tensor(processed_data, device = device, dtype = torch.float32)

x_data = processed_data
y_data = [1,1]
dataset = guansang_dataset(x_data, y_data)

guansang_dataloader = DataLoader(dataset, batch_size=1)


### setting hyper parmas
learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs = 5

### train & test function
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X) # predict with input data
        loss = loss_fn(pred, y) # cal loss with given loss function

        # backpropagation
        optimizer.zero_grad() # initialize optimizer (set to 0, as the gradient is added to existing value)
        loss.backward() # backpropagate, save gradients of loss for the parametors
        optimizer.step() # tune the params based on the collected gradients above

        if batch % 100 == 0: # to print the progress
            loss, current = loss.item(), (batch + 1) * len(X) # loss for current loss, current for current data used
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0 # initialize loss and correct cnt (it will be done batch wise)

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


print('\n'*20)

for t in range(epochs):
    print(f"\n[Epoch {t+1}]\n-------------------------------")
    train_loop(guansang_dataloader, model, loss_fn, optimizer)
    test_loop(guansang_dataloader, model, loss_fn)


## save weights
model_weight_path = base_dir + 'weights/model_weights.pth'
torch.save(model.state_dict(), model_weight_path)

## load weights
model = guansang_temp_model()
model.load_state_dict(torch.load(model_weight_path))