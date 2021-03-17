import cv2
import os
import pandas as pd
from PIL import Image
import csv
import pprint
import numpy as np
import sys

def scale_to_width(img, width):
    scale = width / img.shape[1]
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)

csv_file_name = ''
image_file_name = ''
save_dir = ''

csv_file = pd.read_csv(csv_file_name,encoding="cp932")
line_num = 1
image_num = 1
length = len(csv_file)
print(length)
image_num_max = length
#save_num = 18882
save_num = 1
#gamma22LUT = np.array([pow(x/255.0 , 2.2) for x in range(256)], dtype='float32')
gamma22LUT  = [pow(x/255.0, 2.2)*255 for x in range(256)] * 3
gamma045LUT = [pow(x/255.0, 1.0/2.2)*255 for x in range(256)]

while int(image_num) <= int(image_num_max):    
    if(os.path.exists(os.path.join(image_file_name, "{}.png".format(image_num)))):
        im = cv2.imread(os.path.join(image_file_name, "{}.png".format(image_num)))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        line_num = image_num
        
        left_eye_left_point = csv_file.iat[line_num, 21]
        left_eye_left_point = int(left_eye_left_point)
        
        left_eye_right_point = csv_file.iat[line_num, 27]
        left_eye_right_point = int(left_eye_right_point)
        
        left_eye_top_point = csv_file.iat[line_num, 80]
        left_eye_top_point = int(left_eye_top_point)
        
        left_eye_bottom_point = csv_file.iat[line_num, 86]
        left_eye_bottom_point = int(left_eye_bottom_point)
        
        left_eye_x_cen = int((left_eye_right_point + left_eye_left_point) / 2)
        left_eye_y_cen = int((left_eye_bottom_point + left_eye_top_point) / 2)
        
        cropped_img = im[left_eye_y_cen-76:left_eye_y_cen+56, left_eye_x_cen-110:left_eye_x_cen+110]
        cropped_img = scale_to_width(cropped_img, 160)
        
        cv2.imwrite(os.path.join(save_dir, "{}.png".format(save_num)), cropped_img)
        
        image_num = image_num + 1
        save_num = save_num + 1
    else:
        image_num = image_num + 1    

