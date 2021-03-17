import os
import sys
import json
import fnmatch
import tarfile
import cv2
from PIL import Image
from glob import glob
from tqdm import tqdm
from six.moves import urllib

import numpy as np

#from utils import imread, imwrite

unityeyes_folder = ''
save_dir = ''

#file_num = list_num / 2
num = 1
max_num = 101624
save_num = 1
def process_json_list(json_list, img):
  ldmks = [eval(s) for s in json_list]
  return np.array([(x, img.shape[0]-y, z) for (x,y,z) in ldmks])

def scale_to_width(img, width):
    scale = width / img.shape[1]
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)


while num <= max_num:
    #for num in tqdm(range(max_num)):
    if(os.path.exists(os.path.join(unityeyes_folder, "{}.json".format(num)))):
        with open(os.path.join(unityeyes_folder, "{}.json".format(num))) as json_file:
            img = cv2.imread(os.path.join(unityeyes_folder, "{}.jpg".format(num)))
            j = json.loads(json_file.read())
            key = "interior_margin_2d"
            j[key] = process_json_list(j[key], img)
            #print(j[key])
            x_min, x_max = int(min(j[key][:,0])), int(max(j[key][:,0]))
            y_min, y_max = int(min(j[key][:,1])), int(max(j[key][:,1]))
            x_center, y_center = int((x_min + x_max)/2), int((y_min + y_max)/2)
            cropped_img = img[y_center-72: y_center+72, x_center-120:x_center+120]
            grayscale_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            resize_img = scale_to_width(grayscale_img, 160)
            #print(resize_img.shape)
            cv2.imwrite(os.path.join(save_dir, "{}.png".format(num)), resize_img)
            num = num + 1
            save_num = save_num + 1
    else:
        num = num + 1        
