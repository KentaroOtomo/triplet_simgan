'''
	DataLoader over rides the torch.utils.data.dataset.Dataset class
	This should be changed dependent on the format of your data sets.
'''

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from PIL import Image, ImageOps, ImageStat
import os
import sys
import json 
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from tqdm import tqdm
from preprocess import preprocess_unityeyes_image
import pandas as pd

import torch

with open('config.yaml', 'r') as yml:
    config = yaml.load(yml)

fake_dir = config['synthetic_image_directory']['dirname']
fake_num = len(os.listdir(fake_dir))
eval_fake_dir = config['eval_synthetic_image_directory']['dirname']
eval_fake_num = len(os.listdir(eval_fake_dir))

def convert_gray(self):
		self.transform = transforms.Compose(
			[transforms.Grayscale(num_output_channels=1)
			]					    
		)

name_num = 1
sum_mean = 0 
sum_variance = 0

while name_num <= fake_num:
	#for name_num in tqdm(range(fake_num)):
	#print('name_num = ' + str(name_num))
	#img = Image.open(os.path.join("../data/imgs/", "{}.png".format(name_num)))
	#%img = Image.open(os.path.join("../data/nvgaze/", "{}.png".format(name_num)))
	img = Image.open(os.path.join(fake_dir, "{}.png".format(name_num)))
	#gray_img = convert_gray(img)
	gray_img = np.array(img.convert('L'))
	#print(gray_img)
	gray_img = gray_img / 255
	#print(gray_img)
	mean = np.mean(gray_img)
	#print(mean)
	var = np.var(gray_img)
	#print(var)
	#stat = ImageStat.Stat(gray_img)
	#pix_mean = stat.mean
	#pix_variance = stat.var
	sum_mean = sum_mean + mean
	sum_variance = sum_variance + var
	#print(name_num)
	name_num = name_num + 1    

fake_mean = sum_mean / fake_num
fake_variance = sum_variance / fake_num
print('fake_num:' + str(fake_num))
print('fake_mean:' + str(fake_mean))
print('fake_variance:' + str(fake_variance))

name_num = 1
sum_mean = 0 
sum_variance = 0

name_num = 1
eval_sum_mean = 0 
eval_sum_variance = 0

while name_num <= eval_fake_num:
	#%img = Image.open(os.path.join("../data/nvgaze_eval/", "{}.png".format(name_num)))
	img = Image.open(os.path.join(eval_fake_dir, "{}.png".format(name_num)))
	gray_img = np.array(img.convert('L'))
	gray_img = gray_img / 255
	mean = np.mean(gray_img)
	var = np.var(gray_img)
	
	eval_sum_mean = eval_sum_mean + mean
	eval_sum_variance = eval_sum_variance + var
	name_num = name_num + 1    

eval_fake_mean = eval_sum_mean / eval_fake_num
eval_fake_variance = eval_sum_variance / eval_fake_num
print('eval_fake_num:' + str(eval_fake_num))
print('eval_fake_mean:' + str(eval_fake_mean))
print('eval_fake_variance:' + str(eval_fake_variance))

class Real_Dataset(Dataset):
	def __init__(self):
		self.img_dir_name = config['real_image_directory']['dirname']
		self.dir_name = config['data_directory']['dirname']
		self.real_csv_name = config['real_csv']['csvname']
		self.df = pd.read_csv(self.dir_name + self.real_csv_name)
		self.img_path = self.img_dir_name
		self.imageFiles = sorted([img for img in os.listdir(self.img_path)])

		self.data_transform = transforms.Compose(
			[transforms.Grayscale(num_output_channels=1),
			 transforms.ToTensor()
			 ]					    
		)
		self.data_len = len(self.imageFiles)
		
	def __len__(self):
		return self.data_len

	def calc_mean(self, image):
		gray_img = np.array(image.convert('L'))
		gray_img = gray_img / 255
		mean = np.mean(gray_img)
		var = np.var(gray_img)
		return mean, var

	def __getitem__(self, index):
		file = self.df['id'][index]
		label_num = np.array(self.df['label'][index])
		label = torch.zeros(1, 5).type(torch.LongTensor)
		label[0, label_num] = 1
		
		image_file = (str(file) + ".png")
		image_files = Image.open(self.img_path + image_file)
		r_mean, r_variance = self.calc_mean(image_files)
		image_as_tensor = self.data_transform(image_files)
		
		return image_as_tensor, label

class Eval_Real_Dataset(Dataset):
	def __init__(self):
		self.img_dir_name = config['eval_real_image_directory']['dirname']
		self.img_path = self.img_dir_name
		self.imageFiles = sorted([img for img in os.listdir(self.img_path)])
		
		self.data_transform = transforms.Compose(
			[transforms.Grayscale(num_output_channels=1),
			 transforms.ToTensor(),
			]					    
		)
		self.data_len = len(self.imageFiles)

	def __len__(self):
		return self.data_len

	def calc_mean(self, image):
		gray_img = np.array(image.convert('L'))
		gray_img = gray_img / 255
		mean = np.mean(gray_img)
		var = np.var(gray_img)
		return mean, var

	def __getitem__(self, index):
		image_file = self.imageFiles[index]
		image_files = Image.open(self.img_path + image_file)
		r_mean, r_variance = self.calc_mean(image_files)
		image_as_tensor = self.data_transform(image_files)
		
		return image_as_tensor		

class Fake_Dataset(Dataset):
	def __init__(self):
		self.img_dir_name = config['synthetic_image_directory']['dirname']
		self.json_dir_name = config['synthetic_json_directory']['dirname']
		self.dir_name = config['data_directory']['dirname']
		self.synthetic_csv_name = config['synthetic_csv']['csvname']
		self.df = pd.read_csv(self.dir_name + self.synthetic_csv_name)
		self.img_path = self.img_dir_name
		self.json_path = self.json_dir_name
		self.imageFiles = sorted([img for img in os.listdir(self.img_path)])
		self.jsonFiles = sorted([json for json in os.listdir(self.json_path)])

		self.img_data_len = len(self.imageFiles)
		self.json_data_len = len(self.jsonFiles)
		self.data_len = len(self.img_dir_name)
		self.data_transform = transforms.Compose(
			[transforms.Grayscale(num_output_channels=1),
				transforms.ToTensor()
			]							    
		)
		self.data_len = self.img_data_len
		print('initialize finish')		
	
	def calc_mean(self, image):
		gray_img = np.array(image.convert('L'))
		gray_img = gray_img / 255
		mean = np.mean(gray_img)
		var = np.var(gray_img)
		return mean, var

	def __getitem__(self, index):
		label = torch.zeros(1, 5).type(torch.LongTensor)
		file = self.df['id'][index]
		label_num = np.array(self.df['label'][index])
		label[0, label_num] = 1
		image_file = (str(file) + ".png")
		json_file = (str(file) + ".json")

		image_files = Image.open(self.img_path + image_file)
		image_files = image_files.convert("L")
		with open(self.json_path + json_file) as f:
			json_data = json.load(f)
		
		heatmaps, landmarks, gaze = preprocess_unityeyes_image(image_files, json_data)
		f_mean, f_variance = self.calc_mean(image_files)
		image_as_tensor = self.data_transform(image_files)
		image_as_tensor = transforms.Normalize((f_mean,), (f_variance,))(image_as_tensor)
		image_as_tensor = transforms.Normalize((-1 * fake_mean / fake_variance), (1.0 / fake_variance))(image_as_tensor)
		image_as_tensor = torch.clamp(image_as_tensor, 0, 1)
		
		return image_as_tensor, heatmaps, landmarks, gaze, label

	def __len__(self):	
		return self.data_len

class Eval_Fake_Dataset(Dataset):
	def __init__(self):
		self.img_dir_name = config['eval_synthetic_image_directory']['dirname']
		self.json_dir_name = config['eval_synthetic_json_directory']['dirname']
		self.img_path = self.img_dir_name
		self.json_path = self.json_dir_name
		self.imageFiles = sorted([img for img in os.listdir(self.img_path)])
		self.jsonFiles = sorted([json for json in os.listdir(self.json_path)])
		
		self.img_data_len = len(self.imageFiles)
		self.json_data_len = len(self.jsonFiles)
		self.data_len = len(self.img_dir_name)

		self.data_transform = transforms.Compose(
			[transforms.Grayscale(num_output_channels=1),
				transforms.ToTensor()
			]							    
		)
		self.data_len = self.img_data_len
		print('initialize finish')		
	
	def calc_mean(self, image):
		gray_img = np.array(image.convert('L'))
		gray_img = gray_img / 255
		mean = np.mean(gray_img)
		var = np.var(gray_img)
		return mean, var

	def __getitem__(self, index):
		image_file = self.imageFiles[index]
		json_file = self.jsonFiles[index]
		
		image_files = Image.open(self.img_path + image_file)
		image_files = image_files.convert("L")
		with open(self.json_path + json_file) as f:
			json_data = json.load(f)
		
		heatmaps, landmarks, gaze = preprocess_unityeyes_image(image_files, json_data)
		f_mean, f_variance = self.calc_mean(image_files)
		image_as_tensor = self.data_transform(image_files)
		image_as_tensor = transforms.Normalize((f_mean,), (f_variance,))(image_as_tensor)
		image_as_tensor = transforms.Normalize((-1 * eval_fake_mean / eval_fake_variance), (1.0 / eval_fake_variance))(image_as_tensor)
		image_as_tensor = torch.clamp(image_as_tensor,0, 1)
		return image_as_tensor, heatmaps, landmarks, gaze
		
	def __len__(self):	
		return self.data_len