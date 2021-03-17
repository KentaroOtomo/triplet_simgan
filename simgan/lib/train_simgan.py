'''
	This file includes the class to train SimGAN
	
	TrainSimGAN inherits the functions from 
	SubSimGAN. 

	SubSimGAN has the functions for 
	weight loading, initializing 
	data loaders, and accuracy metrics
'''

import os

import torch
import torch.nn as nn
import torchvision
import cv2
import tensorboardX as tbx
import numpy as np
import argparse
import yaml
import torch.nn.functional as F
import torch.autograd as autograd
import copy
from torchvision import transforms,utils, models
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, median_filter
from sub_simgan import SubSimGAN
from show_images import *
from data_loader import Fake_Dataset, Real_Dataset, Eval_Fake_Dataset
from torchvision.utils import save_image
from tqdm import tqdm
from torch.autograd import Variable
from unity_eyes import UnityEyesDataset
from preprocess import preprocess_unityeyes_image
from losses import HeatmapLoss

with open('config.yaml', 'r') as yml:
	config = yaml.safe_load(yml)

fake_num = len(os.listdir(config['synthetic_image_directory']['dirname']))
real_num = len(os.listdir(config['real_image_directory']['dirname']))


"""
name_num = 1
sum_mean = 0 
sum_variance = 0


while name_num <= fake_num:
	#for name_num in tqdm(range(fake_num)):
	#img = Image.open(os.path.join("../data/imgs/", "{}.png".format(name_num)))
	img = Image.open(os.path.join("../data/eval_imgs3_16/", "{}.png".format(name_num)))
	#%img = Image.open(os.path.join("../data/nvgaze_eval/", "{}.png".format(name_num)))
	gray_img = np.array(img.convert('L'))
	gray_img = gray_img / 255
	mean = np.mean(gray_img)
	var = np.var(gray_img)
	sum_mean = sum_mean + mean
	sum_variance = sum_variance + var
	name_num = name_num + 1

fake_mean = sum_mean / fake_num
fake_variance = sum_variance / fake_num
print("fake_mean = " + str(fake_mean))
print("fake_variance = " + str(fake_variance))


name_num = 1
sum_mean = 0 
sum_variance = 0

while name_num <= real_num:
	#for name_num in tqdm(range(real_num)):
	#img = Image.open(os.path.join("../data/real_yzk_reg/", "real_{}.png".format(name_num)))
	img = Image.open(os.path.join("../data/eval_new_real/", "{}.png".format(name_num)))
	#gray_img = convert_gray(img)
	gray_img = np.array(img.convert('L'))
	gray_img = gray_img / 255
	mean = np.mean(gray_img)
	#print(mean)
	var = np.var(gray_img)
	#stat = ImageStat.Stat(img)
	#pix_mean = stat.mean
	#pix_variance = stat.var
	sum_mean = sum_mean + mean
	sum_variance = sum_variance + var
	name_num = name_num + 1

real_mean = sum_mean / real_num
real_variance = sum_variance / real_num
"""

class TrainSimGAN(SubSimGAN):
	def __init__(self, cfg):
		SubSimGAN.__init__(self, cfg)
		self.writer = tbx.SummaryWriter(config['log_directory']['dirname'])
		self.recon_loss = None
		self.refiner_loss = None
		self.g_global_refined_adv_loss = None
		self.g_local_refined_adv_loss = None
		self.g_global_real_adv_loss = None
		self.g_local_real_adv_loss = None
		self.global_adv_loss = None
		self.local_adv_loss = None
		self.loss_real = None
		self.loss_refined = None
		self.classifier_cross_entropy_loss = None

		self.cfg = cfg
		

	def edge_detection(self,gray):
		dev = gray.device
		#k = 8
		k = config['edge_kernel_size']['size']
		edge_kernel = kernel = np.array([[k, k, k], [k, -k*8, k], [k, k, k]], np.float32)  # convolution filter  
		gaussian_kernel = np.array([[0.077847, 0.123317, 0.077847], [0.123317, 0.195346, 0.123317], [0.077847, 0.123317, 0.077847]], np.float32)
		sharpen_kernel = np.array([[-2, -2, -2], [-2, 31, -2], [-2, -2, -2]], np.float32) / 8.0
		edge_k = torch.as_tensor(edge_kernel.reshape(1, 1, 3, 3)).to(dev)
		gaussian_k =  torch.as_tensor(gaussian_kernel.reshape(1, 1, 3, 3)).to(dev)
		sharpen_k =  torch.as_tensor(sharpen_kernel.reshape(1, 1, 3, 3)).to(dev)
		edge_image = F.conv2d(gray, sharpen_k, padding=1)
		edge_image = F.conv2d(edge_image, edge_k, padding=1)
		edge_image = F.conv2d(edge_image,gaussian_k, padding=1)
		return edge_image


	#%def calc_loss(self, combined_hm_preds, heatmaps, landmarks_pred, landmarks, gaze_pred, gaze):
	def calc_loss(self, combined_hm_preds, heatmaps, landmarks_pred, landmarks):
		combined_loss = []
		self.heatmapLoss = HeatmapLoss().cuda()
		self.landmarks_loss = nn.MSELoss().cuda()
		self.gaze_loss = nn.MSELoss().cuda()
		for i in range(self.nstack):
			combined_loss.append(self.heatmapLoss(combined_hm_preds[:, i, :], heatmaps))

		heatmap_loss = torch.stack(combined_loss, dim=1)
		landmarks_loss = self.landmarks_loss(landmarks_pred, landmarks)
		gaze_loss = self.gaze_loss(gaze_pred, gaze)
		#%return torch.sum(heatmap_loss), landmarks_loss, 1000 * gaze_loss
		return torch.sum(heatmap_loss), landmarks_loss

	

	def update_refiner(self, pretrain=False):
		''' Get batch of synthetic images '''
		synthetic_images, gt_heatmaps, gt_landmarks, gt_gaze, gt_label = next(self.synthetic_data_iter)
		synthetic_images = synthetic_images.cuda()
		edge_synthetic_images = self.edge_detection(synthetic_images)
		gt_heatmaps = gt_heatmaps.cuda()
		gt_landmarks = gt_landmarks.cuda()
		#%gt_gaze = gt_gaze.cuda()
		real_images, label = next(self.real_data_iter)
		real_images = real_images.cuda()
		edge_real_images = self.edge_detection(real_images)
		
		''' Refine synthetic images '''
		cg_to_real_images = self.R1(synthetic_images)
		edge_cg_to_real_images = self.edge_detection(cg_to_real_images)
		self.recon_loss1 = (self.feature_loss(synthetic_images, cg_to_real_images) + self.feature_loss(edge_synthetic_images, edge_cg_to_real_images))/2
		
		return_real_to_cg_images = self.R2(cg_to_real_images)
		edge_return_real_to_cg_images = self.edge_detection(return_real_to_cg_images)
		self.cycle_fake_recon_loss = self.feature_loss(synthetic_images, return_real_to_cg_images) + self.feature_loss(edge_synthetic_images, edge_return_real_to_cg_images)
		
		"""refiner 2"""
		real_to_cg_images = self.R2(real_images)
		edge_real_to_cg_images = self.edge_detection(real_to_cg_images)
		return_cg_to_real_images = self.R1(real_to_cg_images)
		edge_return_cg_to_real_images = self.edge_detection(return_cg_to_real_images)
		self.cycle_real_recon_loss = self.feature_loss(real_images, return_cg_to_real_images) + self.feature_loss(edge_real_images, edge_return_cg_to_real_images)

		"""classifier"""
		synthetic_label = gt_label.cuda()
		synthetic_label = torch.squeeze(synthetic_label, 1)
		self.refiner_refined_accuracy, self.refiner_refined_classifier_loss = self.C(cg_to_real_images, synthetic_label)

		if not pretrain:
			cat_refined_data = torch.cat((cg_to_real_images, edge_cg_to_real_images.cuda()), dim=1)
			#adversarial loss
			g_global_refined_predictions = self.G_D1(cat_refined_data)
			g_local_refined_predictions = self.L_D1(cat_refined_data)
			self.g_global_refined_adv_loss = -torch.mean(g_global_refined_predictions)
			self.g_local_refined_adv_loss = -torch.mean(g_local_refined_predictions)
			cat_real_data = torch.cat((real_to_cg_images, edge_real_to_cg_images.cuda()), dim=1)
			g_global_real_predictions = self.G_D2(cat_real_data)
			g_local_real_predictions = self.L_D2(cat_real_data)
			self.g_global_real_adv_loss = -torch.mean(g_global_real_predictions)
			self.g_local_real_adv_loss = -torch.mean(g_local_real_predictions)
			#heatmap loss
			self.g_refined_heatmaps_pred, self.g_refined_landmarks_pred, self.g_refined_heatmap_loss, self.g_refined_landmarks_loss = self.Reg(cg_to_real_images, gt_heatmaps, gt_landmarks)
			self.g_refined_heatmap_loss2 = (self.g_refined_heatmap_loss * 0.001).mean()

			self.refiner1_optimizer.zero_grad()
			self.refiner2_optimizer.zero_grad()
			#adversarial loss
			self.g_global_refined_adv_loss.backward(retain_graph=True)
			self.g_local_refined_adv_loss.backward(retain_graph=True)
			self.g_global_real_adv_loss.backward(retain_graph=True)
			self.g_local_real_adv_loss.backward(retain_graph=True)
			#heatmap loss
			self.g_refined_heatmap_loss2.backward(retain_graph=True)
			#cycle consistency loss
			self.cycle_fake_recon_loss.backward(retain_graph=True)
			self.cycle_real_recon_loss.backward(retain_graph=True)
			#classifier loss
			self.refiner_refined_classifier_loss.backward()
			self.refiner1_optimizer.step()	
			self.refiner2_optimizer.step()
		else:
			self.refiner1_optimizer.zero_grad()
			self.refiner2_optimizer.zero_grad()
			self.cycle_fake_recon_loss.backward()
			self.cycle_real_recon_loss.backward()
			self.refiner1_optimizer.step()
			self.refiner2_optimizer.step()
	# Used to update the discriminator
	
	def update_discriminator(self, pretrain=False):
		''' get batch of real images '''
		
		real_images, label = next(self.real_data_iter)
		real_images = real_images.cuda()
		edge_real_images = self.edge_detection(real_images)
		cat_real = torch.cat((real_images, edge_real_images.cuda()), dim=1).cuda()
		
		global_real_predictions = self.G_D1(cat_real).cuda()
		local_real_predictions = self.L_D1(cat_real).cuda()

		''' get batch of synthetic images '''
		#%synthetic_images, gt_heatmaps, gt_landmarks, gt_gaze = next(self.synthetic_data_iter)
		synthetic_images, gt_heatmaps, gt_landmarks, gt_gaze, gt_label = next(self.synthetic_data_iter)
		synthetic_images = synthetic_images.cuda()
		edge_synthetic_images = self.edge_detection(synthetic_images)
		cat_synthetic = torch.cat((synthetic_images, edge_synthetic_images.cuda()), dim=1)

		cg_to_real_images = self.R1(synthetic_images)
		edge_cg_to_real_images = self.edge_detection(cg_to_real_images)
		cat_refined = torch.cat((cg_to_real_images, edge_cg_to_real_images.cuda()), dim=1).cuda()

		global_refined_predictions = self.G_D1(cat_refined).cuda()
		local_refined_predictions = self.L_D1(cat_refined).cuda()

		global_synthetic_predictions = self.G_D1(cat_synthetic).cuda()
		local_synthetic_predictions = self.L_D1(cat_synthetic).cuda()
		train_global_gradient_penalty = self.global_compute_gradient_penalty(self.G_D1, cat_real, cat_refined).cuda()
		train_local_gradient_penalty = self.local_compute_gradient_penalty(self.L_D1, cat_real, cat_refined).cuda()
		pretrain_global_gradient_penalty = self.global_compute_gradient_penalty(self.G_D1, cat_real, cat_synthetic).cuda()
		pretrain_local_gradient_penalty = self.local_compute_gradient_penalty(self.L_D1, cat_real, cat_synthetic).cuda()
		if not pretrain:
			self.global_discriminator1_optimizer.zero_grad()
			self.global_discriminator1_loss = -torch.mean(global_real_predictions) + torch.mean(global_refined_predictions) + (10 * train_global_gradient_penalty)
			self.global_discriminator1_loss.backward(retain_graph=True)
			self.global_discriminator1_optimizer.step()

			self.local_discriminator1_optimizer.zero_grad()
			self.local_discriminator1_loss = -torch.mean(local_real_predictions) + torch.mean(local_refined_predictions) + (10 * train_local_gradient_penalty)
			self.local_discriminator1_loss.backward(retain_graph=True)
			self.local_discriminator1_optimizer.step()
		else:
			self.global_discriminator1_optimizer.zero_grad()
			self.global_discriminator1_loss = -torch.mean(global_real_predictions) + torch.mean(global_synthetic_predictions) + (10 * pretrain_global_gradient_penalty)
			self.global_discriminator1_loss.backward(retain_graph=True)
			self.global_discriminator1_optimizer.step()

			self.local_discriminator1_optimizer.zero_grad()
			self.local_discriminator1_loss = -torch.mean(local_real_predictions) + torch.mean(local_synthetic_predictions) + (10 * pretrain_local_gradient_penalty)
			self.local_discriminator1_loss.backward(retain_graph=True)
			self.local_discriminator1_optimizer.step()
			
		#real_to_cg_images = self.R2(real_images)
		real_to_cg_images = self.R1(real_images)
		edge_real_to_cg_images = self.edge_detection(real_to_cg_images)
		cat_real_to_cg = torch.cat((real_to_cg_images, edge_real_to_cg_images), dim=1).cuda()

		global_fake_predictions2 = self.G_D2(cat_real_to_cg).cuda()
		local_fake_predictions2 = self.L_D2(cat_real_to_cg).cuda()		

		global_synthetic_predictions2 = self.G_D2(cat_synthetic).cuda()
		local_synthetic_predictions2 = self.L_D2(cat_synthetic).cuda()

		train_global_gradient_penalty2 = self.global_compute_gradient_penalty(self.G_D2, cat_synthetic, cat_real_to_cg).cuda()
		train_local_gradient_penalty2 = self.local_compute_gradient_penalty(self.L_D2, cat_synthetic, cat_real_to_cg).cuda()

		global_real_predictions2 = self.G_D2(cat_real).cuda()
		local_real_predictions2 = self.L_D2(cat_real).cuda()

		pretrain_global_gradient_penalty2 = self.global_compute_gradient_penalty(self.G_D2, cat_synthetic, cat_real).cuda()
		pretrain_local_gradient_penalty2 = self.local_compute_gradient_penalty(self.L_D2, cat_synthetic, cat_real).cuda()

		if not pretrain:
			self.global_discriminator2_optimizer.zero_grad()
			self.global_discriminator2_loss = -torch.mean(global_synthetic_predictions2) + torch.mean(global_fake_predictions2) + (10 * train_global_gradient_penalty2)
			self.global_discriminator2_loss.backward(retain_graph=True)
			self.global_discriminator2_optimizer.step()

			self.local_discriminator2_optimizer.zero_grad()
			self.local_discriminator2_loss = -torch.mean(local_synthetic_predictions2) + torch.mean(local_fake_predictions2) + (10 * train_local_gradient_penalty2)
			self.local_discriminator2_loss.backward(retain_graph=True)
			self.local_discriminator2_optimizer.step()
		else:
			self.global_discriminator2_optimizer.zero_grad()
			self.global_discriminator2_loss = -torch.mean(global_synthetic_predictions2) + torch.mean(global_real_predictions2) + (10 * pretrain_global_gradient_penalty2)
			self.global_discriminator2_loss.backward(retain_graph=True)
			self.global_discriminator2_optimizer.step()

			self.local_discriminator2_optimizer.zero_grad()
			self.local_discriminator2_loss = -torch.mean(local_synthetic_predictions2) + torch.mean(local_real_predictions2) + (10 * pretrain_local_gradient_penalty2)
			self.local_discriminator2_loss.backward(retain_graph=True)
			self.local_discriminator2_optimizer.step()

		#updata classifier
	def update_classifier(self, pretrain=False):
		real_images, label = next(self.real_data_iter)
		real_images = real_images.cuda() 
		label = label.cuda()
		label = torch.squeeze(label, 1)
		synthetic_images, gt_heatmaps, landmark, gaze, synthetic_label = next(self.synthetic_data_iter)
		synthetic_images = synthetic_images.cuda()
		synthetic_label = synthetic_label.cuda()
		synthetic_label = torch.squeeze(synthetic_label, 1)

		refined_images = self.R1(synthetic_images)
		self.real_accuracy, self.real_classifier_loss = self.C(real_images, label)
		self.refined_accuracy, self.refined_classifier_loss = self.C(refined_images, synthetic_label)

		if not pretrain:	
			self.classifier_optimizer.zero_grad()
			self.real_classifier_loss.backward()
			self.refined_classifier_loss.backward()
			self.classifier_optimizer.step()
		else:
			self.classifier_optimizer.zero_grad()
			self.real_classifier_loss.backward()
			self.classifier_optimizer.step()

	def update_regressor(self, pretrain=False):
		''' Get batch of synthetic images '''
		#%synthetic_images, gt_heatmaps, gt_landmarks, gt_gaze = next(self.synthetic_data_iter)
		synthetic_images, gt_heatmaps, gt_landmarks, gt_gaze, gt_label = next(self.synthetic_data_iter)
		synthetic_images = synthetic_images.cuda()
		gt_heatmaps = gt_heatmaps.cuda()
		gt_landmarks = gt_landmarks.cuda()
		#%gt_gaze = gt_gaze.cuda()
		''' Refine synthetic images '''
		refined_images = self.R1(synthetic_images)		
		if not pretrain:
			self.regressor_optimizer.zero_grad()
			self.reg_refined_heatmaps_pred, self.reg_refined_landmarks_pred, self.reg_refined_heatmap_loss, self.reg_refined_landmarks_loss = self.Reg(refined_images.cuda(), gt_heatmaps, gt_landmarks)
			self.reg_refined_heatmap_loss2 = self.reg_refined_heatmap_loss * 1000
			self.reg_refined_all_loss = self.reg_refined_heatmap_loss2.mean() + self.reg_refined_landmarks_loss.mean()
			self.reg_refined_all_loss.backward()
			
			self.reg_synthetic_heatmaps_pred, self.reg_synthetic_landmarks_pred, self.reg_synthetic_heatmap_loss, self.reg_synthetic_landmarks_loss = self.Reg(synthetic_images, gt_heatmaps, gt_landmarks)
			self.synthetic_heatmap_loss2 = self.reg_synthetic_heatmap_loss * 1000
			self.reg_synthetic_all_loss = self.synthetic_heatmap_loss2.mean() + self.reg_synthetic_landmarks_loss.mean()
			self.reg_synthetic_all_loss.backward()
			self.regressor_optimizer.step()
		else:
			self.regressor_optimizer.zero_grad()
			self.pre_reg_synthetic_heatmaps_pred, self.pre_reg_synthetic_landmarks_pred, self.pre_reg_synthetic_heatmap_loss, self.pre_reg_synthetic_landmarks_loss = self.Reg(synthetic_images, gt_heatmaps, gt_landmarks)
			self.pre_synthetic_heatmap_loss2 = self.pre_reg_synthetic_heatmap_loss * 1000
			self.pre_reg_synthetic_all_loss = self.pre_synthetic_heatmap_loss2.mean() + self.pre_reg_synthetic_landmarks_loss.mean()
			self.pre_reg_synthetic_all_loss.backward()
			self.regressor_optimizer.step()

	def update_gaze_estimator(self, pretrain=False):
		''' Get batch of synthetic images '''
		synthetic_images, gt_heatmaps, gt_landmarks, gt_gaze, gt_label = next(self.synthetic_data_iter)
		self.synthetic_images = synthetic_images.cuda()
		self.gt_heatmaps = gt_heatmaps.cuda()
		self.gt_landmarks = gt_landmarks.cuda()
		self.gt_gaze = gt_gaze.cuda()

		''' Refine synthetic images '''
		self.refined_images = self.R1(self.synthetic_images)		
		#edge_refined_images = self.edge_detection(refined_images)
		#edge_synthetic_images = self.edge_detection(synthetic_images)
		if not pretrain:
			self.reg_refined_heatmaps_pred, self.reg_refined_landmarks_pred, self.reg_refined_heatmap_loss, self.reg_refined_landmarks_loss = self.Reg(self.refined_images, self.gt_heatmaps, self.gt_landmarks)
			self.reg_refined_gaze, self.reg_refined_gaze_loss = self.Gaze(self.refined_images, self.reg_refined_landmarks_pred, self.gt_gaze)
			self.gaze_estimator_optimizer.zero_grad()
			self.reg_return_refined_gaze_loss = self.reg_refined_gaze_loss.mean() * 1000
			self.reg_return_refined_gaze_loss.backward(retain_graph=True)
			
			self.reg_synthetic_heatmaps_pred, self.reg_synthetic_landmarks_pred, self.reg_synthetic_heatmap_loss, self.reg_synthetic_landmarks_loss = self.Reg(self.synthetic_images, self.gt_heatmaps, self.gt_landmarks)
			self.reg_synthetic_gaze, self.reg_synthetic_gaze_loss = self.Gaze(self.synthetic_images, self.gt_landmarks, self.gt_gaze)
			self.reg_return_synthetic_gaze_loss = self.reg_synthetic_gaze_loss.mean() * 1000
			self.reg_return_synthetic_gaze_loss.backward(retain_graph=True)
			self.gaze_estimator_optimizer.step()
		else:
			self.gaze_estimator_optimizer.zero_grad()
			self.pre_reg_synthetic_heatmaps_pred, self.pre_reg_synthetic_landmarks_pred, self.pre_reg_synthetic_heatmap_loss, self.pre_reg_synthetic_landmarks_loss = self.Reg(self.synthetic_images, self.gt_heatmaps, self.gt_landmarks)
			self.pre_reg_synthetic_gaze, self.pre_reg_synthetic_gaze_loss = self.Gaze(self.synthetic_images, self.gt_landmarks, self.gt_gaze)
			self.pre_return_gaze_loss = self.pre_reg_synthetic_gaze_loss.mean() * 1000
			self.pre_return_gaze_loss.backward(retain_graph=True)
			self.gaze_estimator_optimizer.step()
			
	''' Used to pretrain the refiner if no previous
		weights are found '''
	def pretrain_refiner(self):
		# This method pretrains the generator if called
		print('Pre-training the refiner network {} times'.format(config['refiner_pretrain_iteration']['num']))
		''' Set the refiner gradients parameters to True 
			Set the discriminators gradients params to False'''

		#self.R.train_mode(True)
		self.R1.train()
		for param in self.R1.parameters():
			param.requires_grad = True
		self.R2.train()		
		for param in self.R2.parameters():
			param.requires_grad = True		
		
		''' Begin pre-training the refiner '''
		for step in range(config['refiner_pretrain_iteration']['num']):
			self.update_refiner(pretrain=True)
			if step % config['print_interval']['num'] == 0 or (step == config['refiner_pretrain_iteration']['num'] - 1):
				self.print_refiner_info(step, pretrain=True)	
		print('Done pre-training the refiner')

	''' Used to pretrain the discriminator if no previous
		weights are found '''
	def pretrain_discriminator(self):
		print('Pre-training the discriminator network {} times'.format(config['discriminator_pretrain_iteration']['num']))

		''' Set the Discriminators gradient parameters to True
			Set the Refiners gradient parameters to False '''
		#self.D.train_mode(True)
		self.G_D1.train()
		for param in self.G_D1.parameters():
			param.requires_grad = True
		self.G_D2.train()
		for param in self.G_D2.parameters():
			param.requires_grad = True
		self.L_D1.train()
		for param in self.L_D1.parameters():
			param.requires_grad = True
		self.L_D2.train()
		for param in self.L_D2.parameters():
			param.requires_grad = True			
		self.R1.eval()
		for param in self.R1.parameters():
			param.requires_grad = False
		self.R2.eval()
		for param in self.R2.parameters():
			param.requires_grad = False
		
		''' Begin pretraining the discriminator '''
		for step in range(config['discriminator_pretrain_iteration']['num']):
			''' update discriminator and return some important info for printing '''
			self.update_discriminator(pretrain=True)
			if step % config['print_interval']['num'] == 0 or (step == config['discriminator_pretrain_iteration']['num'] - 1):
				self.print_discriminator_info(step, pretrain=True)

		print('Done pre-training the discriminator')

	def pretrain_classifier(self):
		print('Pre-training the classifier network {} times'.format(config['classifier_pretrain_iteration']['num']))

		''' Set the Classifiers gradient parameters to True
			Set the Refiners gradient parameters to False '''
		self.C.train()
		for param in self.C.parameters():
			param.requires_grad = True
		self.R1.eval()
		for param in self.R1.parameters():
			param.requires_grad = False
		''' Begin pretraining the discriminator '''
		for step in range(config['classifier_pretrain_iteration']['num']):
			''' update discriminator and return some important info for printing '''
			self.update_classifier(pretrain=True)
			if step % config['print_interval']['num'] == 0 or (step == config['classifier_pretrain_iteration']['num'] - 1):
				self.print_classifier_info(step, pretrain=True)
		print('Done pre-training the classifier')
		
	def pretrain_regressor(self):
		print('Pre-training the regressor network {} times'.format(config['regressor_pretrain_iteration']['num']))

		''' Set the Classifiers gradient parameters to True
			Set the Refiners gradient parameters to False '''
		self.Reg.train()
		for param in self.Reg.parameters():
			param.requires_grad = True
	
		self.R1.eval()
		for param in self.R1.parameters():
			param.requires_grad = False

		''' Begin pretraining the discriminator '''
		for step in range(config['regressor_pretrain_iteration']['num']):	
			''' update discriminator and return some important info for printing '''
			self.update_regressor(pretrain=True)
			if step % config['print_interval']['num'] == 0 or (step == config['regressor_pretrain_iteration']['num'] - 1):
				self.print_regressor_info(step, pretrain=True)
		print('Done pre-training the regressor')
	''' main function called externally
		used to train the entire network '''
	
	def pretrain_gaze_estimator(self):
		print('Pre-training the regressor network {} times'.format(config['gaze_estimator_pretrain_iteration']['num']))

		''' Set the Classifiers gradient parameters to True
			Set the Refiners gradient parameters to False '''
		self.Gaze.train()
		for param in self.Gaze.parameters():
			param.requires_grad = True        
		self.Reg.eval()
		for param in self.Reg.parameters():
			param.requires_grad = False
		self.R1.eval()
		for param in self.R1.parameters():
			param.requires_grad = False

		''' Begin pretraining the discriminator '''
		for step in range(config['gaze_estimator_pretrain_iteration']['num']):	
			''' update discriminator and return some important info for printing '''
			self.update_gaze_estimator(pretrain=True)
			if step % config['print_interval']['num'] == 0 or (step == config['gaze_estimator_pretrain_iteration']['num'] - 1):
				self.print_gaze_estimator_info(step, pretrain=True)
		print('Done pre-training the gaze estimator')
	''' main function called externally
		used to train the entire network '''

	def train(self):
		self.build_network()	
		self.get_data_loaders()
		''' If no saved weights are found,
			pretrain the refiner / discriminator '''
		if not self.weights_loaded:
			self.pretrain_refiner()
			self.pretrain_discriminator()
			self.pretrain_classifier()
			self.pretrain_regressor()
			self.pretrain_gaze_estimator()
			print('save refiner state')
			torch.save(self.R1.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_refiner1_path']['pathname']))
			torch.save(self.R2.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_refiner2_path']['pathname']))
			print('save discriminator state')
			torch.save(self.G_D1.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_global_discriminator1_path']['pathname']))
			torch.save(self.L_D1.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_local_discriminator1_path']['pathname']))
			torch.save(self.G_D2.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_global_discriminator2_path']['pathname']))
			torch.save(self.L_D2.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_local_discriminator2_path']['pathname']))
			print('save regressor state')
			torch.save(self.Reg.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_regressor_path']['pathname']))
			print('save gaze estimator state')
			torch.save(self.Gaze.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_gaze_estimator_path']['pathname']))
			print('save classifier state')
			torch.save(self.C.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['pretrain_classifier_path']['pathname']))

			state = {
					'step': 0,
					'optG_D_1' : self.global_discriminator1_optimizer.state_dict(),
					'optG_D_2' : self.global_discriminator2_optimizer.state_dict(),
					'optL_D_1' : self.local_discriminator1_optimizer.state_dict(),
					'optL_D_2' : self.local_discriminator2_optimizer.state_dict(),
					'optR1' : self.refiner1_optimizer.state_dict(),
					'optR2' : self.refiner2_optimizer.state_dict(),
					'optC' : self.classifier_optimizer.state_dict(),
					'optReg' : self.regressor_optimizer.state_dict(),
					'optGaze' : self.gaze_estimator_optimizer.state_dict()
				}
				
			torch.save(state, os.path.join(config['checkpoint_path']['pathname'], config['optimizer_path']['pathname']))
			
		assert self.current_step < config['train_iteration']['num'], 'Target step is smaller than current step'
		for step in range((self.current_step + 1), config['train_iteration']['num']):
			self.current_step = step			
			''' Train Refiner '''
			self.R1.train()
			for param in self.R1.parameters():
				param.requires_grad = True
			self.R2.train()
			for param in self.R2.parameters():
				param.requires_grad = True	
			self.Reg.eval()
			for param in self.Reg.parameters():
				param.requires_grad = False
			self.G_D1.eval()
			for param in self.G_D1.parameters():
				param.requires_grad = False
			self.G_D2.eval()
			for param in self.G_D2.parameters():
				param.requires_grad = False
			self.L_D1.eval()
			for param in self.L_D1.parameters():
				param.requires_grad = False
			self.L_D2.eval()
			for param in self.L_D2.parameters():
				param.requires_grad = False
			self.C.eval()
			for param in self.C.parameters():
				param.requires_grad = False					
			for idx in range(config['updates_refiner_per_step']['num']):								
				self.update_refiner(pretrain=False)

			'''Train Regressor'''
			self.Reg.train()
			for param in self.Reg.parameters():
				param.requires_grad = True
			self.R1.eval()
			for param in self.R1.parameters():
				param.requires_grad = False
			for idx in range(config['updates_regressor_per_step']['num']):
				self.update_regressor(pretrain=False)

			'''Train Gaze Estimator'''
			self.Gaze.train()
			for param in self.Gaze.parameters():
				param.requires_grad = True
			self.R1.eval()
			for param in self.R1.parameters():
				param.requires_grad = False
			for param in self.Reg.parameters():
				param.requires_grad = False    
			for idx in range(config['updates_gaze_estimator_per_step']['num']):
				self.update_gaze_estimator(pretrain=False)   	

			'''Train Discriminator'''
			self.G_D1.train()
			for param in self.G_D1.parameters():
				param.requires_grad = True
			self.G_D2.train()
			for param in self.G_D2.parameters():
				param.requires_grad = True	
			self.L_D1.train()
			for param in self.L_D1.parameters():
				param.requires_grad = True
			self.L_D2.train()
			for param in self.L_D2.parameters():
				param.requires_grad = True	
			self.R1.eval()
			for param in self.R1.parameters():
				param.requires_grad = False
			self.R2.eval()	
			for param in self.R2.parameters():
				param.requires_grad = False	
			#self.D.train_mode(True)					
			for idx in range(config['updates_discriminator_per_step']['num']):
				self.update_discriminator()

			'''Train Classifier'''
			self.C.train()
			for param in self.C.parameters():
				param.requires_grad = True
			self.R1.eval()
			for param in self.R1.parameters():
				param.requires_grad = False
			for idx in range(config['updates_classifier_per_step']['num']):
				self.update_classifier(pretrain=False)
					#self.C.eval()
					#self.C.train_mode(False)
					###self.Reg.eval()
					###for param in self.Reg.parameters():
					###	param.requires_grad = False
					
					#self.update_classifier(pretrain=False)
				
			
			if step % config['print_interval']['num'] == 0 and step > 0:
				self.print_refiner_info(step, pretrain=False)
				self.print_discriminator_info(step, pretrain=False)
				self.print_classifier_info(step, pretrain=False)
				self.print_regressor_info(step, pretrain=False)
			
			
			if config['log']['bool'] == True and (step % config['log_interval']['num'] == 0 or step == 0):
				self.writer.add_scalar("Refiner/cycle fake recon loss ", self.cycle_fake_recon_loss.item(), global_step = step)
				self.writer.add_scalar("Refiner/cycle real recon loss ", self.cycle_real_recon_loss.item(), global_step = step)
				self.writer.add_scalar("Refiner/Reconstruction Loss ", self.recon_loss1.item(), global_step = step)
				self.writer.add_scalar("Refiner/Global refined adversarial loss ", self.g_global_refined_adv_loss.item(), global_step = step)
				self.writer.add_scalar("Refiner/Local refined adversarial loss ", self.g_local_refined_adv_loss.item(), global_step = step)
				self.writer.add_scalar("Refiner/Global real adversarial loss ", self.g_global_real_adv_loss.item(), global_step = step)
				self.writer.add_scalar("Refiner/Local real adversarial loss ", self.g_local_real_adv_loss.item(), global_step = step)
				self.writer.add_scalar("Refiner/refined entropy loss", self.refiner_refined_classifier_loss.item(), global_step = step)
				self.writer.add_scalar("Refiner/refined accuracy", self.refiner_refined_accuracy.item(), global_step = step)
				#self.writer.add_scalar("Refiner/style loss ", self.style_losses.item(), global_step = step)
				self.writer.add_scalar("Discriminator/Global Discriminator1 Adversarial loss", self.global_discriminator1_loss.item(), global_step = step)
				self.writer.add_scalar("Discriminator/Local Discriminator1 Adversarial loss", self.local_discriminator1_loss.item(), global_step = step)
				self.writer.add_scalar("Discriminator/Global Discriminator2 Adversarial loss", self.global_discriminator2_loss.item(), global_step = step)
				self.writer.add_scalar("Discriminator/Local Discriminator2 Adversarial loss", self.local_discriminator2_loss.item(), global_step = step)
				
				self.writer.add_scalar("Regressor/refined heatmap loss", self.reg_refined_heatmap_loss.mean().item(), global_step = step)
				self.writer.add_scalar("Regressor/refined landmark loss", self.reg_refined_landmarks_loss.mean().item(), global_step = step)
				#%self.writer.add_scalar("Regressor/refined gaze loss", self.reg_refined_gaze_loss.item(), global_step = step)

				self.writer.add_scalar("Regressor/synthetic heatmap loss", self.reg_synthetic_heatmap_loss.mean().item(), global_step = step)
				self.writer.add_scalar("Regressor/synthetic landmark loss", self.reg_synthetic_landmarks_loss.mean().item(), global_step = step)
				#%self.writer.add_scalar("Regressor/synthetic gaze loss", self.reg_synthetic_gaze_loss.item(), global_step = step)
				
				self.writer.add_scalar("Classifier/real entropy loss", self.real_classifier_loss.item(), global_step = step)
				self.writer.add_scalar("Classifier/refined entropy loss", self.refined_classifier_loss.item(), global_step = step)
				self.writer.add_scalar("Classifier/real accuracy", self.real_accuracy.item(), global_step = step)
				self.writer.add_scalar("Classifier/refined accuracy", self.refined_accuracy.item(), global_step = step)
				self.writer.close()
			
			if step % config['save_interval']['num'] == 0:
				print('Saving checkpoints, Step : {}'.format(step))	
				torch.save(self.R1.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['refiner1_path']['pathname'] % step))
				torch.save(self.R2.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['refiner2_path']['pathname'] % step))
				torch.save(self.G_D1.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['global_discriminator1_path']['pathname'] % step))
				torch.save(self.G_D2.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['global_discriminator2_path']['pathname'] % step))
				torch.save(self.L_D1.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['local_discriminator1_path']['pathname'] % step))
				torch.save(self.L_D2.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['local_discriminator2_path']['pathname'] % step))
				torch.save(self.C.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['classifier_path']['pathname'] % step))
				torch.save(self.Reg.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['regressor_path']['pathname'] % step))
				torch.save(self.Gaze.state_dict(), os.path.join(config['checkpoint_path']['pathname'], config['gaze_estimator_path']['pathname'] % step))

				state = {
					'step': step,
					'optG_D_1' : self.global_discriminator1_optimizer.state_dict(),
					'optG_D_2' : self.global_discriminator2_optimizer.state_dict(),
					'optL_D_1' : self.local_discriminator1_optimizer.state_dict(),
					'optL_D_2' : self.local_discriminator2_optimizer.state_dict(),
					'optR1' : self.refiner1_optimizer.state_dict(),
					'optR2' : self.refiner2_optimizer.state_dict(),
					'optC' : self.classifier_optimizer.state_dict(),
					'optReg' : self.regressor_optimizer.state_dict(),
					'optGaze' : self.gaze_estimator_optimizer.state_dict()
				}
				
				torch.save(state, os.path.join(config['checkpoint_path']['pathname'], config['optimizer_path']['pathname']))
				#%clamp_synthetic_images, clamp_gt_heatmaps, clamp_gt_landmarks, clamp_gt_gaze = next(self.eval_data_iter)
				clamp_synthetic_images, clamp_gt_heatmaps, clamp_gt_landmarks, _ = next(self.eval_data_iter)
				eval_real_images = next(self.eval_real_data_iter)
				eval_real_images = eval_real_images.cuda()
				save_refined_images = self.R1(clamp_synthetic_images.cuda())
				save_refined_heatmaps_pred, save_refined_landmarks_pred, _, _ = self.Reg(save_refined_images, clamp_gt_heatmaps.cuda(), clamp_gt_landmarks.cuda())

				stack_real_images = torch.empty(0, 3, 96, 160)
				stack_synthetic_images = torch.empty(0, 3, 96, 160)
				stack_eye_blend_images = torch.empty(0, 3, 96, 160)
				stack_refined_images = torch.empty(0, 3, 96, 160)
				stack_refined_blend_images = torch.empty(0, 3, 96, 160)
				stack_all_images = torch.empty(5, 3, 96, 160)
				stack_eval_real_images = torch.empty(0, 3, 96, 160)
				i = 1
				for i in range(config['batch_size']['size']):
					save_synthetic_img = torch.empty(1, 96, 160)
					save_refined_img = torch.empty(1, 96, 160)
					save_real_img = torch.empty(1, 96, 160)
					
					transform_img = transforms.Compose([
						transforms.ToPILImage()
					])
					transform_tensor = transforms.Compose([
						transforms.ToTensor()
					])

					save_synthetic_img = clamp_synthetic_images[i].cuda()
					save_real_img = eval_real_images[i].cuda()
					#save_real_img = save_real_img.unsqueeze(0)
					gt_hm = clamp_gt_heatmaps[i].cuda()
					gt_hm = np.mean(gt_hm[0:32].cpu().detach().numpy(), axis=0)
					gt_hm = cv2.normalize(gt_hm, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
					cv2.imwrite('gt_heatmap.png', gt_hm*255)
					gt_hm_color = cv2.imread('gt_heatmap.png')
					gt_hm_color = cv2.applyColorMap(gt_hm_color, cv2.COLORMAP_JET)
					#cv2.imwrite('gt_heatmap_color255.png', gt_hm_color*255)
					#cv2.imwrite('heatmap_color.png', gt_hm_color)
					save_synthetic_img_3ch = torch.cat((save_synthetic_img, save_synthetic_img, save_synthetic_img), dim=0)
					utils.save_image(save_synthetic_img_3ch, 'synthetic_3ch_tensor.png')
					save_synthetic_img_3ch_pil = transforms.ToPILImage()(save_synthetic_img_3ch.cpu())
					save_synthetic_img_3ch_np = np.asarray(save_synthetic_img_3ch_pil, dtype=np.uint8)
					save_synthetic_img_3ch_np = cv2.cvtColor(save_synthetic_img_3ch_np, cv2.COLOR_RGB2BGR)
					#cv2.imwrite('synthetic_3ch.png', save_synthetic_img_3ch_np)
					alpha = 0.3
					blended_image = cv2.addWeighted(gt_hm_color, alpha, save_synthetic_img_3ch_np, 1-alpha, 0)
					#cv2.imwrite('gt_blend_cv2.png', blended_image)
					blended_image_cv2 = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
					blended_image_pil = Image.fromarray(blended_image_cv2)
					eye_blend_images = transform_tensor(blended_image_pil)
					#utils.save_image(eye_blend_images, 'gt_blend.png')
					eye_blend_images = eye_blend_images.unsqueeze(0)
					#print("eye_blend_images = ", eye_blend_images.shape)
					save_refined_img = save_refined_images[i].cuda()
					save_heatmaps_img = save_refined_heatmaps_pred[i].cuda()
					save_heatmaps_img = save_heatmaps_img.squeeze(0)
					hm_pred = np.mean(save_heatmaps_img[-1, 0:32].cpu().detach().numpy(), axis=0)
					hm_pred = cv2.normalize(hm_pred, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
					cv2.imwrite('pred.png', hm_pred*255)
					hm_pred_color = cv2.imread('pred.png')
					
					i = 0
					while i <= 95:
						j = 0
						while j <= 159:
							#print(j)
							if hm_pred_color[i][j][0] < 70 or hm_pred_color[i][j][1] < 70 or hm_pred_color[i][j][2] < 70:
								hm_pred_color[i][j][0]=0
								hm_pred_color[i][j][1]=0
								hm_pred_color[i][j][2]=0
								j = j + 1
							else:
								j = j + 1
						i = i + 1
					
					hm_pred_color = cv2.applyColorMap(hm_pred_color, cv2.COLORMAP_JET)
					#cv2.imwrite('pred_heatmap_color.png', hm_pred_color)
					#cv2.imwrite('pred_heatmap_color255.png', hm_pred_color*255)
					#cv2.imwrite('heatmap_pred_color.png', hm_pred_color)
					save_refined_img_3ch = torch.cat((save_refined_img, save_refined_img, save_refined_img), dim=0)
					save_refined_img_pil = transforms.ToPILImage()(save_refined_img_3ch.cpu())
					save_refined_img_np = np.asarray(save_refined_img_pil, dtype=np.uint8)
					save_refined_img_np = cv2.cvtColor(save_refined_img_np, cv2.COLOR_RGB2BGR)
					alpha = 0.3
					refined_blended_image = cv2.addWeighted(hm_pred_color, alpha, save_refined_img_np, 1-alpha, 0)
					refined_blended_image_cv2 = cv2.cvtColor(refined_blended_image, cv2.COLOR_BGR2RGB)
					#print(refined_blended_image_cv2.shape)
					refined_blended_image_pil = Image.fromarray(refined_blended_image_cv2)
					refined_blend_images = transforms.ToTensor()(refined_blended_image_pil)
					#cv2.imwrite('refined_blend_cv2.png', refined_blended_image)
					#refined_blend_images = transform_tensor(refined_blended_image)
					#utils.save_image(refined_blend_images, 'refined_blend.png')
					#print("refined_blend_images = ", refined_blend_images.shape)
					refined_blend_images = refined_blend_images.unsqueeze(0)
					#print("refined_blend_images = ", refined_blend_images.shape)

					save_synthetic_img_3ch = save_synthetic_img_3ch.unsqueeze(0)
					#print("save_synthetic_img_3ch:", save_synthetic_img_3ch.shape)
					
					save_real_images = torch.cat((save_real_img, save_real_img, save_real_img), dim=0)
					save_real_img_3ch = save_real_images.unsqueeze(0)
					#print("save_real_img_3ch:", save_real_img_3ch.shape)
					#stack_real_images = torch.cat((stack_real_images.cuda(), save_real_img_3ch), dim=0)
					
					save_refined_img_3ch = save_refined_img_3ch.unsqueeze(0)
					#print("save_refined_img_3ch:", save_refined_img_3ch.shape)
					#stack_refined_images = torch.cat((stack_refined_images.cuda(), save_refined_img_3ch), dim=0)
					#print("stack_all_images:", stack_all_images.shape)
					#stack_eye_blend_images = torch.cat((stack_eye_blend_images, eye_blend_images), dim=0)
					#stack_refined_blend_images = torch.cat((stack_refined_blend_images, refined_blend_images), dim=0)
					stack_all_images = torch.cat((stack_all_images.cuda(), save_synthetic_img_3ch.cuda(), eye_blend_images.cuda(), save_refined_img_3ch.cuda(), refined_blend_images.cuda(), save_real_img_3ch.cuda()), dim = 0)
					#print("stack_all_images:", stack_all_images.shape)
					i = i + 1
				make_grid_all_images = utils.make_grid(stack_all_images, 5)
				utils.save_image(make_grid_all_images, config['save_image_directory']['dirname'] + 'all_' + str(step) + '.png')
