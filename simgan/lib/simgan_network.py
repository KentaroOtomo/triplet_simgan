'''
	This file hold the network architecture for SimGAN including the 
	Refiner and Discriminator
'''

#from torch import nn
import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import argparse
import yaml

from layers import Conv, Hourglass, Pool, Residual
from losses import HeatmapLoss
from softargmax import softargmax2d

with open('config.yaml', 'r') as yml:
	config = yaml.safe_load(yml)

nstack = config['nstack']['param']
nfeatures = config['nfeatures']['param']
nlandmarks = config['nlandmarks']['param']

class ResnetBlock(nn.Module):
	# Resnet block used for building the Refiner
	# Implements a skip connection
	# 64 output filters with 3x3 convolutions

	def __init__(self, input_features, nb_features=64, filter_size=3):
		super(ResnetBlock, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(input_features, nb_features, filter_size, 1, 1),
			nn.InstanceNorm2d(nb_features, affine=False),
			#nn.BatchNorm2d(nb_features, affine=False),
			#nn.LayerNorm(nb_features),
			nn.LeakyReLU(),
			nn.Conv2d(nb_features, nb_features, filter_size, 1, 1),
			nn.InstanceNorm2d(nb_features, affine=False),
			#nn.BatchNorm2d(nb_features, affine=False),
			#nn.LayerNorm(nb_features),
			nn.LeakyReLU(),
			nn.Conv2d(nb_features, nb_features, filter_size, 1, 1),
			#nn.LeakyReLU()
		)
		self.relu = nn.LeakyReLU()

	def forward(self, x):
		convs = self.conv(x)
		sum_ = convs + x
		output = self.relu(sum_)
		#print("refiner_output = " + str(output.shape))
		return output

class Refiner1(nn.Module):
	'''
	Refiner --- class used to refine inputed synthetic data

		Notes*
			1. input should be a batch of synthetic, grayscale images with shape equal to [1, 35, 55]

			2. Output would be a batch of refined imaged (more realistic)

	'''

	def __init__(self, block_num=8, nb_features=64):
		super(Refiner1, self).__init__()
		
		''' Image size is [1, 35, 55] '''
		self.conv_1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=nb_features, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm2d(nb_features, affine=False),
			#nn.BatchNorm2d(nb_features, affine=False),
			#nn.LayerNorm(nb_features),
			nn.LeakyReLU()
		)

		blocks = []
		for i in range(block_num):
			blocks.append(ResnetBlock(nb_features, nb_features, filter_size=3))
			
				
		self.resnet_blocks = nn.Sequential(*blocks)
		self.conv_2 = nn.Sequential(
			nn.Conv2d(nb_features, 1, kernel_size=1, stride=1, padding=0),
			#nn.Tanh()
		)
	# used externally 
	# used for forward prop
	def forward(self, x):
		#print(x.shape)
		conv_1 = self.conv_1(x)
		#uns(conv_1.shape)
		res_block = self.resnet_blocks(conv_1)
		#output = x - self.conv_2(res_block)
		output = self.conv_2(res_block)
		return output

class Refiner2(nn.Module):
	'''
	Refiner --- class used to refine inputed synthetic data

		Notes*
			1. input should be a batch of synthetic, grayscale images with shape equal to [1, 35, 55]

			2. Output would be a batch of refined imaged (more realistic)

	'''

	def __init__(self, block_num=8, nb_features=64):
		super(Refiner2, self).__init__()
		
		''' Image size is [1, 35, 55] '''
		self.conv_1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=nb_features, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm2d(nb_features, affine=False),
			#nn.BatchNorm2d(nb_features, affine=False),
			#nn.LayerNorm(nb_features),
			nn.LeakyReLU()
		)

		blocks = []
		for i in range(block_num):
			blocks.append(ResnetBlock(nb_features, nb_features, filter_size=3))
			
				
		self.resnet_blocks = nn.Sequential(*blocks)
		self.conv_2 = nn.Sequential(
			nn.Conv2d(nb_features, 1, kernel_size=1, stride=1, padding=0),
			#nn.Tanh()
		)
	# used externally 
	# used for forward prop
	def forward(self, x):
		conv_1 = self.conv_1(x)
		res_block = self.resnet_blocks(conv_1)
		output = self.conv_2(res_block)
		return output

class Global_Discriminator1(nn.Module):
	''' 
	Discriminator --- class used to discriminate between refined and real data
		
		Notes*
			1. Input is a set of refined or real grayscale images of shape == [1, 35, 55]
			2. Output is a 2D conv map which is a map of probabilities between refined or real
	'''

	def __init__(self):
		super(Global_Discriminator1, self).__init__()
		
		''' Image size is [1, 55, 35] '''
		self.convs = nn.Sequential(
			nn.Conv2d(2, 96, 3, 2, 1),
			nn.InstanceNorm2d(96, affine=False),
			#nn.BatchNorm2d(96, affine=False),
			#nn.LayerNorm(96),
			nn.LeakyReLU(),
			nn.Conv2d(96, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=False),
			#nn.BatchNorm2d(64, affine=False),
			#nn.LayerNorm(64),
			nn.LeakyReLU(),
			nn.Conv2d(64, 32, 3, 1, 1),
			nn.InstanceNorm2d(32, affine=False),
			#nn.BatchNorm2d(32, affine=False),
			#nn.LayerNorm(32),
			nn.LeakyReLU(),
			nn.Conv2d(32, 16, 3, 1, 1),
			nn.InstanceNorm2d(16, affine=False),
			#nn.BatchNorm2d(16, affine=False),
			#nn.LayerNorm(16),
			nn.LeakyReLU(),
			nn.Conv2d(16, 2, 3, 1, 1)		
		)
		self.linear = nn.Linear(7680, 1)
		self.sigmoid = nn.Sigmoid()
		self.sequential = nn.Sequential()
		#self.linear = nn.Linear(168, 1)
		
	def forward(self, x):
		convs = self.convs(x)
		output = convs.view(convs.shape[0], -1)
		return output

class Local_Discriminator1(nn.Module):
	''' 
	Discriminator --- class used to discriminate between refined and real data
		
		Notes*
			1. Input is a set of refined or real grayscale images of shape == [1, 35, 55]
			2. Output is a 2D conv map which is a map of probabilities between refined or real
	'''

	def __init__(self):
		super(Local_Discriminator1, self).__init__()
		
		''' Image size is [1, 55, 35] '''
		self.convs = nn.Sequential(
			nn.Conv2d(2, 96, 3, 2, 1),
			nn.InstanceNorm2d(96, affine=False),
			#nn.BatchNorm2d(96, affine=False),
			#nn.LayerNorm(96),
			nn.LeakyReLU(),
			nn.Conv2d(96, 64, 3, 2, 1),
			nn.InstanceNorm2d(64, affine=False),
			#nn.BatchNorm2d(64, affine=False),
			#nn.LayerNorm(64),
			nn.LeakyReLU(),
			nn.Conv2d(64, 64, 3, 2, 1),
			nn.InstanceNorm2d(64, affine=False),
			#nn.BatchNorm2d(64, affine=False),
			#nn.LayerNorm(64),
			nn.LeakyReLU(),
			nn.Conv2d(64, 64, 3, 2, 1),
			nn.InstanceNorm2d(64, affine=False),
			#nn.BatchNorm2d(64, affine=False),
			#nn.LayerNorm(64),
			nn.LeakyReLU(),
			#nn.MaxPool2d(3, 1, 1),
			#nn.MaxPool2d(2, 2, 0),
			nn.Conv2d(64, 32, 3, 1, 1),
			nn.InstanceNorm2d(32, affine=False),
			#nn.BatchNorm2d(32, affine=False),
			#nn.LayerNorm(32),
			nn.LeakyReLU(),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.InstanceNorm2d(32, affine=False),
			#nn.BatchNorm2d(32, affine=False),
			#nn.LayerNorm(32),
			nn.LeakyReLU(),
			#nn.Conv2d(32, 32, 3, 1, 1),
			#nn.LeakyReLU(),
			nn.Conv2d(32, 16, 3, 1, 1),
			nn.InstanceNorm2d(16, affine=False),
			#nn.BatchNorm2d(16, affine=False),
			#nn.LayerNorm(16),
			nn.LeakyReLU(),
			#nn.MaxPool2d(2, 2, 0),
			#nn.Conv2d(32, 32, 1, 1, 0),
			#nn.LeakyReLU(),
			nn.Conv2d(16, 2, 3, 1, 1)
		)
		self.linear = nn.Linear(120, 1)
		self.sigmoid = nn.Sigmoid()
		self.sequential = nn.Sequential()
		#self.linear = nn.Linear(168, 1)
		
	def forward(self, x):
		convs = self.convs(x)
		output = convs.view(convs.shape[0], -1)
		return output

class Global_Discriminator2(nn.Module):
	''' 
	Discriminator --- class used to discriminate between refined and real data
		
		Notes*
			1. Input is a set of refined or real grayscale images of shape == [1, 35, 55]
			2. Output is a 2D conv map which is a map of probabilities between refined or real
	'''

	def __init__(self):
		super(Global_Discriminator2, self).__init__()
		
		''' Image size is [1, 55, 35] '''
		self.convs = nn.Sequential(
			nn.Conv2d(2, 96, 3, 2, 1),
			nn.InstanceNorm2d(96, affine=False),
			#nn.BatchNorm2d(96, affine=False),
			#nn.LayerNorm(96),
			nn.LeakyReLU(),
			nn.Conv2d(96, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=False),
			#nn.BatchNorm2d(64, affine=False),
			#nn.LayerNorm(64),
			nn.LeakyReLU(),
			nn.Conv2d(64, 32, 3, 1, 1),
			nn.InstanceNorm2d(32, affine=False),
			#nn.BatchNorm2d(32, affine=False),
			#nn.LayerNorm(32),
			nn.LeakyReLU(),
			nn.Conv2d(32, 16, 3, 1, 1),
			nn.InstanceNorm2d(16, affine=False),
			#nn.BatchNorm2d(16, affine=False),
			#nn.LayerNorm(16),
			nn.LeakyReLU(),
			nn.Conv2d(16, 2, 3, 1, 1)	
		)
		self.linear = nn.Linear(7680, 1)
		self.sigmoid = nn.Sigmoid()
		self.sequential = nn.Sequential()
		#self.linear = nn.Linear(168, 1)
		
	def forward(self, x):
		convs = self.convs(x)
		output = convs.view(convs.shape[0], -1)
		return output

class Local_Discriminator2(nn.Module):
	''' 
	Discriminator --- class used to discriminate between refined and real data
		
		Notes*
			1. Input is a set of refined or real grayscale images of shape == [1, 35, 55]
			2. Output is a 2D conv map which is a map of probabilities between refined or real
	'''

	def __init__(self):
		super(Local_Discriminator2, self).__init__()
		
		''' Image size is [1, 55, 35] '''
		self.convs = nn.Sequential(
			nn.Conv2d(2, 96, 3, 2, 1),
			nn.InstanceNorm2d(96, affine=False),
			#nn.BatchNorm2d(96, affine=False),
			#nn.LayerNorm(96),
			nn.LeakyReLU(),
			nn.Conv2d(96, 64, 3, 2, 1),
			nn.InstanceNorm2d(64, affine=False),
			#nn.BatchNorm2d(64, affine=False),
			#nn.LayerNorm(64),
			nn.LeakyReLU(),
			nn.Conv2d(64, 64, 3, 2, 1),
			nn.InstanceNorm2d(64, affine=False),
			#nn.BatchNorm2d(64, affine=False),
			#nn.LayerNorm(64),
			nn.LeakyReLU(),
			nn.Conv2d(64, 64, 3, 2, 1),
			nn.InstanceNorm2d(64, affine=False),
			#nn.BatchNorm2d(64, affine=False),
			#nn.LayerNorm(64),
			nn.LeakyReLU(),
			#nn.MaxPool2d(3, 1, 1),
			#nn.MaxPool2d(2, 2, 0),
			nn.Conv2d(64, 32, 3, 1, 1),
			nn.InstanceNorm2d(32, affine=False),
			#nn.BatchNorm2d(32, affine=False),
			#nn.LayerNorm(32),
			nn.LeakyReLU(),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.InstanceNorm2d(32, affine=False),
			#nn.BatchNorm2d(32, affine=False),
			#nn.LayerNorm(32),
			nn.LeakyReLU(),
			#nn.Conv2d(32, 32, 3, 1, 1),
			#nn.LeakyReLU(),
			nn.Conv2d(32, 16, 3, 1, 1),
			nn.InstanceNorm2d(16, affine=False),
			#nn.BatchNorm2d(16, affine=False),
			#nn.LayerNorm(16),
			nn.LeakyReLU(),
			#nn.MaxPool2d(2, 2, 0),
			#nn.Conv2d(32, 32, 1, 1, 0),
			#nn.LeakyReLU(),
			nn.Conv2d(16, 2, 3, 1, 1)
		)
		self.linear = nn.Linear(120, 1)
		self.sigmoid = nn.Sigmoid()
		self.sequential = nn.Sequential()
		#self.linear = nn.Linear(168, 1)
		
	def forward(self, x):
		convs = self.convs(x)
		output = convs.view(convs.shape[0], -1)
		return output				


class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()

		self.c_convs = nn.Sequential(
			nn.Conv2d(1, 32, 5, 1, 2),
			nn.LeakyReLU(),
			nn.MaxPool2d(2),
			nn.Dropout(0.5),
			nn.Conv2d(32, 64, 3, 1, 1),
			nn.LeakyReLU(),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.LeakyReLU(),
			nn.MaxPool2d(2),
			nn.Dropout(0.5),
			nn.Conv2d(64, 128, 3, 1, 1),
			nn.LeakyReLU(),
			nn.Conv2d(128, 128, 3, 1, 1),
			nn.LeakyReLU(),
			#add
			#nn.MaxPool2d(2, 2, 0),
			#nn.Conv2d(128, 128, 3, 1, 1),
			#nn.LeakyReLU()
		)
		self.avgpool = nn.AvgPool2d(7)
		self.fc  = nn.Linear(1920, 5)
		#self.reshape = torch.reshape()
	
	def forward(self, x, gt_class):
		self.cross_entropy_loss = nn.CrossEntropyLoss()
		convs = self.c_convs(x)
		convs = self.avgpool(convs)
		convs = torch.reshape(convs, (len(convs), -1))
		output = self.fc(convs)
		probs = torch.softmax(output, dim=1)
		winners = probs.argmax(dim=1)
		gt_class_winners = gt_class.argmax(dim=1)
		corrects = (winners == gt_class_winners)
		accuracy = corrects.sum().float() / float(gt_class_winners.size(0))
		classifier_loss = self.cross_entropy_loss(output, torch.max(gt_class, 1)[1])

		return accuracy, classifier_loss

#Regressor Network(GazeML)
class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class Regressor(nn.Module):
	def __init__(self, nstack, nfeatures, nlandmarks, bn=False, increase=0, **kwargs):
		super(Regressor, self).__init__()

		self.img_w = 160
		self.img_h = 96	
		self.nstack = config['nstack']['param']
		self.nfeatures = config['nfeatures']['param']
		self.nlandmarks = config['nlandmarks']['param']

		self.heatmap_w = self.img_w
		self.heatmap_h = self.img_h

		self.pre = nn.Sequential(
			Conv(1, 64, 7, 1, bn=True, relu=True),
			Residual(64, 128),
			#Pool(3, 1),
			Residual(128, 128),
			Residual(128, nfeatures)
		)

		self.pre2 = nn.Sequential(
			Conv(nfeatures, 64, 7, 2, bn=True, relu=True),
			Residual(64, 128),
			Pool(2, 2),
			Residual(128, 128),
			Residual(128, nfeatures)
		)

		self.hgs = nn.ModuleList([
			nn.Sequential(
				Hourglass(4, nfeatures, bn, increase),
			) for i in range(nstack)])

		self.features = nn.ModuleList([
			nn.Sequential(
				Residual(nfeatures, nfeatures),
				Conv(nfeatures, nfeatures, 1, bn=True, relu=True)
			) for i in range(nstack)])

		self.outs = nn.ModuleList([Conv(nfeatures, nlandmarks, 1, relu=False, bn=False) for i in range(nstack)])
		self.merge_features = nn.ModuleList([Merge(nfeatures, nfeatures) for i in range(nstack - 1)])
		self.merge_preds = nn.ModuleList([Merge(nlandmarks, nfeatures) for i in range(nstack - 1)])
		
		self.gaze_fc1 = nn.Linear(in_features=int(nfeatures * self.img_w * self.img_h / 16 + nlandmarks*2), out_features=256)
		self.gaze_fc2 = nn.Linear(in_features=256, out_features=2)

		self.nstack = nstack
		self.heatmapLoss = HeatmapLoss()
		self.landmarks_loss = nn.MSELoss()
		self.gaze_loss = nn.MSELoss()

	def forward(self, imgs, gt_heatmap, gt_landmark):
		# imgs of size 1,ih,iw
		
		x = imgs.unsqueeze(1)
		x = self.pre(imgs)
		
		gaze_x = self.pre2(x)
		gaze_x = gaze_x.flatten(start_dim=1)

		combined_hm_preds = []
		for i in torch.arange(self.nstack):
			hg = self.hgs[i](x)
			feature = self.features[i](hg)
			preds = self.outs[i](feature)
			combined_hm_preds.append(preds)
			if i < self.nstack - 1:
				x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
				
		heatmaps_out = torch.stack(combined_hm_preds, 1)
		combined_loss = []
		for i in range(self.nstack):
			combined_loss.append(self.heatmapLoss(heatmaps_out[:, i, :], gt_heatmap))
		heatmap_loss = torch.stack(combined_loss, dim=1)	
		landmarks_out = softargmax2d(preds)  # N x nlandmarks x 2
		landmark_loss = self.landmarks_loss(landmarks_out, gt_landmark)

		return heatmaps_out, landmarks_out, heatmap_loss, landmark_loss

class GazeEstimator(nn.Module):
	def __init__(self, nstack, nfeatures, nlandmarks, bn=False, increase=0, **kwargs):
		super(GazeEstimator, self).__init__()

		self.img_w = 160
		self.img_h = 96	
		self.nstack = config['nstack']['param']
		self.nfeatures = config['nfeatures']['param']
		self.nlandmarks = config['nlandmarks']['param']

		self.heatmap_w = self.img_w
		self.heatmap_h = self.img_h

		self.nstack = nstack
		self.pre = nn.Sequential(
			Conv(1, 64, 7, 1, bn=True, relu=True),
			Residual(64, 128),
			#Pool(3, 1),
			Residual(128, 128),
			Residual(128, nfeatures)
		)

		self.pre2 = nn.Sequential(
			Conv(nfeatures, 64, 7, 2, bn=True, relu=True),
			Residual(64, 128),
			Pool(2, 2),
			Residual(128, 128),
			Residual(128, nfeatures)
		)

		self.features = nn.ModuleList([
			nn.Sequential(
				Residual(nfeatures, nfeatures),
				Conv(nfeatures, nfeatures, 1, bn=True, relu=True)
			) for i in range(nstack)])

		self.outs = nn.ModuleList([Conv(nfeatures, nlandmarks, 1, relu=False, bn=False) for i in range(nstack)])
		self.merge_features = nn.ModuleList([Merge(nfeatures, nfeatures) for i in range(nstack - 1)])
		self.merge_preds = nn.ModuleList([Merge(nlandmarks, nfeatures) for i in range(nstack - 1)])
		
		self.gaze_fc1 = nn.Linear(in_features=int(nfeatures * self.img_w * self.img_h / 16 + nlandmarks*2), out_features=256)
		self.gaze_fc2 = nn.Linear(in_features=256, out_features=2)

		self.nstack = nstack
		self.gaze_loss = nn.MSELoss()

	def forward(self, imgs, landmarks, gt_gaze):
		x = imgs.unsqueeze(1)
		x = self.pre(imgs)
		
		gaze_x = self.pre2(x)
		gaze_x = gaze_x.flatten(start_dim=1)
		
		# Gaze
		gaze = torch.cat((gaze_x, landmarks.flatten(start_dim=1)), dim=1)
		gaze = self.gaze_fc1(gaze)
		gaze = nn.functional.relu(gaze)
		gaze = self.gaze_fc2(gaze)
		gaze_loss = self.gaze_loss(gaze, gt_gaze)
		return gaze, gaze_loss

		