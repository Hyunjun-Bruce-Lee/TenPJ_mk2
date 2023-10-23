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
base_dir = '/Users/hyunjun_bruce_lee/Documents/GIT/TenPJ_mk2/'
test_data_names = os.listdir(base_dir + 'test_img')

processed_data = list()
for img_name in tqdm(test_data_names):
    processed_data.append(generator.data_processing(base_dir + 'test_img/' + img_name))




import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

