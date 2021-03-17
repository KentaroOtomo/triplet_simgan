from __future__ import print_function, division
import os
from typing import Optional

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import os
import cv2
import json
from preprocess import preprocess_unityeyes_image


class UnityEyesDataset(Dataset):

    def __init__(self, img_dir: Optional[str] = None):

        if img_dir is None:
            img_dir = os.path.join(os.path.dirname(__file__), '../data/imgs_json')

        self.img_paths = glob.glob(os.path.join(img_dir, '*.png'))
        self.img_paths = sorted(self.img_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        print(len(self.img_paths))
        self.json_paths = []
        for img_path in self.img_paths:
            idx = os.path.splitext(os.path.basename(img_path))[0]
            self.json_paths.append(os.path.join(img_dir, f'{idx}.json'))

    def __len__(self):
        return len(self.img_paths)

    def scale_to_width(img, width):
        scale = width / img.shape[1]
        return cv2.resize(img, dsize=None, fx=scale, fy=scale)


    def __getitem__(self, idx):
        def scale_to_width(img, width):
            scale = width / img.shape[1]
            return cv2.resize(img, dsize=None, fx=scale, fy=scale)
        
        if torch.is_tensor(idx):
            idx = idx.tolist()    
        num = 1
        data_len = len(os.listdir("../data/imgs_json/")) / 2
        #full_img = cv2.imread(self.img_paths[idx])
        while num <= data_len:
            full_img = cv2.imread(os.path.join("../data/imgs_json/", "{}.png".format(num)))
            #print(full_img.shape)
            #full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
            #full_img = scale_to_width(full_img, 80)
            #self.resize_image = torchvision.transforms.ComResize(size=(54, 90), interpolation=2)
            #full_img = self.resize_image(full_img)  
            #full_img = cv2.imread("datasets/UnityEyes/imgs/5.jpg")
            #with open(self.json_paths[idx]) as f:
            with open(os.path.join("../data/imgs_json/", "{}.json".format(num))) as f:
                json_data = json.load(f)
            #json_data = json.load("UnityEyes/imgs/1.json")
            eye_sample = preprocess_unityeyes_image(full_img, json_data)
            sample = {'full_img': full_img, 'json_data': json_data }
            sample.update(eye_sample)
            num = num + 1
            return sample