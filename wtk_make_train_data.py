import os
from copy import deepcopy

# to find out the format of collected imgs
base_path = os.getcwd()
label_dir = os.listdir(base_path + '/train_imgs/labels')
label_dir.sort()
label_dir = label_dir[1:] # not to include .ds_store

label_dict = dict()
for label_file in label_dir:
    label_img_nms = os.listdir(base_path + '/train_imgs/labels/' + label_file)
    temp_list = list()
    for i in label_img_nms:
        if i != '.DS_Store':
            temp_list.append(i)
    label_dict[label_file] = deepcopy(temp_list)

formats = list()
for key in label_dict.keys():
    for img_nm in label_dict[key]:
        formats.append(img_nm.split('.')[-1])

set(formats) # {'png', 'jpg', 'jpeg', 'avif', 'webp'}

# change image format to jpg
from PIL import Image
import pillow_avif # pip install pillow-avif-plugin Pillow

for key in label_dict.keys():
    for i, file_name in enumerate(label_dict[key]):
        f_nm, f_format = file_name.split('.')[0], file_name.split('.')[-1]
        img = Image.open(base_path + f'/train_imgs/labels/{key}/' + file_name)
        if f_format == 'png':
            img = img.convert('RGB')
        img.save(base_path + f'/train_imgs/labels_jpg/{key}/' + str(i) + '_' + f_nm + '.jpg')









# generate data
import cv2
import mediapipe as mp
import math
import numpy as np
from tqdm import tqdm

class face_mesh_generator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks = True, static_image_mode = True, max_num_faces = 1)
    
    # distance between two points
    def get_length(self, x,y,z):
        return math.sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2)

    # calculate every single length between the dots in face mesh
    def data_processing(self, img_dir):
        image = cv2.imread(img_dir)
        results = self.face_mesh.process(image)
        if results.multi_face_landmarks == None:
            return '-1'
        
        landmark_info = results.multi_face_landmarks[0].landmark
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

generator = face_mesh_generator()

base_dir = os.getcwd()
data_file_nms = os.listdir(base_dir + '/train_imgs/labels_jpg')
data_file_nms.sort()
data_file_nms = data_file_nms[1:]

temp_data_dict = dict()
for key in data_file_nms:
    temp_list = list()
    temp_file_nms = os.listdir(base_dir + '/train_imgs/labels_jpg/' + key)
    for i in temp_file_nms:
        if i != '.DS_Store':
            temp_list.append(i)
    temp_data_dict[key] = deepcopy(temp_list)

for key in temp_data_dict.keys():
    print(f'{key} {len(temp_data_dict[key])}')



mat_holder = list()
label_holder = list()
for key in temp_data_dict.keys():
    label = int(key.split('_')[-1])
    for img in temp_data_dict[key]:
        mat = generator.data_processing(f'{base_dir}/train_imgs/labels_jpg/{key}/{img}')
        if mat != '-1':
            mat_holder.append(mat)
            label_holder.append(label)
        else:
            print(f'{key} : {img}')

train_dataset = {'data' : mat_holder, 'labels' : label_holder}

import pickle

with open(f'{base_dir}/train_dataset.bin', 'wb') as f:
    pickle.dump(train_dataset, f)