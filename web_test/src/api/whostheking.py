from io import BytesIO
from PIL import Image
import base64
from fastapi import APIRouter, UploadFile
import numpy as np  # 1.24.3
import torch  # ver 2.0.1+cu118
import torch.nn as nn
import cv2
import torch.nn.functional as F
import mediapipe as mp
import math

# data parsing
class data_parser :
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks = True, static_image_mode = True, max_num_faces = 4)
    
    def __call__(self, img):
        result = self.generate_face_mesh(img)
        if result == '-1':
            return '-1'
        result = self.get_landmarks_by_face(result)
        data_holder = [self.generate_matrix(result[key]) for key in result.keys()]
        return np.expand_dims(np.array(data_holder), axis = 1)

    def get_length(self, x,y,z):
        return math.sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2)

    def generate_face_mesh(self, img):
        mesh_data = self.face_mesh.process(img)
        return -1 if mesh_data.multi_face_landmarks == None else mesh_data

    def get_landmarks_by_face(self, mesh_data):
        landmark_holder = dict()
        for i, j in enumerate(mesh_data.multi_face_landmarks):
            landmark_holder[str(i+1)] = j.landmark
        return landmark_holder
    
    def generate_matrix(self, landmark_info):
        cal_arr = list()
        for i in range(len(landmark_info)):
            one, temp_arr = landmark_info[i], list()
            for j in range(len(landmark_info)):
                two = landmark_info[j]
                temp_arr.append(self.get_length([one.x, two.x], [one.y, two.y], [one.z, two.z]))
            cal_arr.append(temp_arr)
        return cal_arr
    

# whos the king model
class wtk_model(nn.Module):
    def __init__(self):
        super().__init__()        # input shape (478,478)
        self.conv_1st = nn.Conv2d(1, 3, 2**8, padding = 2**6)   # out shape (351,351)
        self.pool_1st = nn.MaxPool2d((9,9),stride = 2) # out shape (172,172)
        self.conv_2nd = nn.Conv2d(3,3,2**6, padding = 8) # out shape (125,125)
        self.pool_2nd = nn.MaxPool2d((5,5), stride = 5) # out shape (25,25)
        self.ffn_1st = nn.Linear(3*25*25, 2**9)
        self.ffn_2nd = nn.Linear(2**9, 2**7)
        self.ffn_3rd = nn.Linear(2**7, 2**5)
        self.ffn_4th = nn.Linear(2**5, 2**4)
        self.ffn_fin = nn.Linear(2**4,4)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool_1st(F.relu(self.conv_1st(x)))
        x = self.pool_2nd(F.relu(self.conv_2nd(x)))
        x = torch.flatten(x,1)
        x = self.dropout(F.relu(self.ffn_1st(x)))
        x = self.dropout(F.relu(self.ffn_2nd(x)))
        x = self.dropout(F.relu(self.ffn_3rd(x)))
        x = self.dropout(F.relu(self.ffn_4th(x)))
        x = self.ffn_fin(x)
        return x 

  
class response_generator:
    def __init__(self):
        self.model = wtk_model()
    
    def predict(self, single_data):
        single_data = np.expand_dims(single_data, axis = 0)
        single_data = torch.tensor(single_data, dtype = torch.float32)
        with torch.no_grad():
            proba = nn.Softmax(dim=1)(self.model(single_data))
        return proba.numpy()
    
    def predict_all(self, whole_data):
        proba_holder = list()
        for single_data in whole_data: # 4d shape
            proba_holder.append(self.predict(single_data))
        return proba_holder


# test run
test_img = cv2.imread('/Users/hyunjun_bruce_lee/Documents/GIT/TenPJ_mk2/test_img/p179506928231510_590.jpg')

test_obj = data_parser()

resp_gen = response_generator()

test_temp = test_obj(test_img)

proba_list = resp_gen.predict_all(test_temp)

for i in range(len(proba_list)):
    print(proba_list[i])

rank_dict = {1:0,2:0,3:0,4:0}

for label_num in range(0,4):    
    temp_list = [proba_list[person][0][label_num] for person in range(len(proba_list))]

    highest_person = temp_list.index(max(temp_list))

    rank_dict[highest_person+1] = label_num

rank_dict


# API
router = APIRouter()

@router.post("/whos_the_king")
async def upload_result(file: UploadFile):
    model, parser = wtk_model(), data_parser()
    model.load_state_dict(torch.load('PATH'), map_location=torch.device('cpu'))
    content = await file.read()
    content = np.array(Image.open(BytesIO(content)))[:, :, ::-1]
    img = await parser(content)
    
 
    return 'result'
