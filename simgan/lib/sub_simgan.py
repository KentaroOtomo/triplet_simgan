
'''
	This file hold a class 'SubSimGAN' that has some basic
	functionality we can inherit from when building the
	TrainSimGAN or TestSimGAN classes. Most of it isn't 
	terribly important thats why I hide it in this sub class.

	Things such as accuracy metrics, data loaders, 
	weight loading, etc
'''

import os

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import yaml
import torch.autograd as autograd
from torchvision import transforms, models
from time import sleep
from torch.autograd import Variable

import torch.nn.functional as F
from simgan_network import Refiner1, Refiner2, Global_Discriminator1, Local_Discriminator1, Global_Discriminator2, Local_Discriminator2, Classifier, Regressor, GazeEstimator
#from data_loader import Real_DataLoader, Fake_DataLoader, Fake_Dataset, Real_Dataset
from data_loader import Fake_Dataset, Real_Dataset, Eval_Fake_Dataset, Eval_Real_Dataset
import numpy as np


with open('config.yaml', 'r') as yml:
	config = yaml.safe_load(yml)

class SubSimGAN:
	''' 
		SubSimGAN : a class that can be inherited by TestSimGAN or TrainSimGAN.

			Notes*
				1. This class is meant to hide anything that isn't directly used to 
				train or test the simgan network.
				2. Input is the training / testing config file
				3. No output, methods and variables are inherited
	'''
	
	def __init__(self, cfg):
		
		self.cfg = cfg
		
		# initializing variables to None, used later
		self.R1 = None
		self.R2 = None
		self.G_D1 = None
		self.L_D1 = None
		self.G_D2 = None
		self.L_D2 = None
		self.C = None
		self.Gaze = None
		self.Reg = None
		self.vgg = None
		self.local_discriminator2 = None
		self.classifier = None
		self.regressor = None
		self.gaze_estimator = None

		self.refiner1_optimizer = None
		self.refiner2_optimizer = None
		self.global_discriminator_optimizer1 = None
		self.local_discriminator_optimizer1 = None
		self.global_discriminator_optimizer2 = None
		self.local_discriminator_optimizer2 = None
		self.classifier_optimizer = None
		self.regressor_optimizer = None
		
		self.feature_loss = None #Usually L1 norm or content loss
		self.local_adversarial_loss = None #CrossEntropyLoss
		self.data_loader = None
		self.current_step = None

		self.synthetic_data_loader = None
		self.real_data_loader = None
		self.synthetic_data_iter = None
		self.test_real_data_loader = None
		self.real_data_iter = None
		self.weights_loaded = None
		self.current_step = 0

		if not config['train']['bool']:
			self.testing_done = False

	# Set up cmdline args

	# used internally
	# checks for saved weights in the checkpoint path
	# return True if weights are loaded
	# return False if no weights are found
	print("sub simgan initialize finish.")
	def load_weights(self):
		
		print("Checking for Saved Weights")

		# If checkpoint path doesn't exist, create it
		if not os.path.isdir(config['checkpoint_path']['pathname']):
			os.mkdir(config['checkpoint_path']['pathname'])
		
		# get list of checkpoints from checkpoint path
		checkpoints = os.listdir(config['checkpoint_path']['pathname'])
		#print(checkpoints)
		# Only load weights that start with 'R_' or 'D_'
		refiner1_checkpoints = [ckpt for ckpt in checkpoints if 'R1_' == ckpt[:3]]
		refiner2_checkpoints = [ckpt for ckpt in checkpoints if 'R2_' == ckpt[:3]]
		global_discriminator1_checkpoints = [ckpt for ckpt in checkpoints if 'G_D1_' == ckpt[:5]]
		local_discriminator1_checkpoints = [ckpt for ckpt in checkpoints if 'L_D1_' == ckpt[:5]]
		global_discriminator2_checkpoints = [ckpt for ckpt in checkpoints if 'G_D2_' == ckpt[:5]]
		local_discriminator2_checkpoints = [ckpt for ckpt in checkpoints if 'L_D2_' == ckpt[:5]]
		classifier_checkpoints = [ckpt for ckpt in checkpoints if 'C_' == ckpt[:2]]
		regressor_checkpoints = [ckpt for ckpt in checkpoints if 'Reg_' == ckpt[:4]]
		gaze_estimator_checkpoints = [ckpt for ckpt in checkpoints if 'Gaze_' == ckpt[:5]]

		refiner1_checkpoints.sort(key=lambda x: int(x[3:-4]), reverse=True)
		refiner2_checkpoints.sort(key=lambda x: int(x[3:-4]), reverse=True)
		global_discriminator1_checkpoints.sort(key=lambda x: int(x[5:-4]), reverse=True)
		local_discriminator1_checkpoints.sort(key=lambda x: int(x[5:-4]), reverse=True)
		global_discriminator2_checkpoints.sort(key=lambda x: int(x[5:-4]), reverse=True)
		local_discriminator2_checkpoints.sort(key=lambda x: int(x[5:-4]), reverse=True)
		classifier_checkpoints.sort(key=lambda x: int(x[2:-4]), reverse=True)
		regressor_checkpoints.sort(key=lambda x: int(x[4:-4]), reverse=True)
		gaze_estimator_checkpoints.sort(key=lambda x: int(x[5:-4]), reverse=True)

		if len(refiner1_checkpoints) == 0 or not os.path.isfile(	
										os.path.join(config['checkpoint_path']['pathname'], config['optimizer_path']['pathname'])):
			print("No Previous Weights Found. Building and Initializing new Model")
			self.current_step = 0
			return False

		print("Found Saved Weights, Loading...")		

		if config['train']['bool']:
			# load optimizer information / estimator weigths
			optimizer_status = torch.load(os.path.join(config['checkpoint_path']['pathname'], config['optimizer_path']['pathname']))
			self.refiner1_optimizer.load_state_dict(optimizer_status['optR1'])
			self.refiner2_optimizer.load_state_dict(optimizer_status['optR2'])
			self.global_discriminator1_optimizer.load_state_dict(optimizer_status['optG_D_1'])
			self.local_discriminator1_optimizer.load_state_dict(optimizer_status['optL_D_1'])
			self.global_discriminator2_optimizer.load_state_dict(optimizer_status['optG_D_2'])
			self.local_discriminator2_optimizer.load_state_dict(optimizer_status['optL_D_2'])
			self.classifier_optimizer.load_state_dict(optimizer_status['optC'])
			self.regressor_optimizer.load_state_dict(optimizer_status['optReg'])
			self.gaze_estimator_optimizer.load_state_dict(optimizer_status['optGaze'])
			self.current_step = optimizer_status['step']

			self.G_D1.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], global_discriminator1_checkpoints[0])))
			self.L_D1.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], local_discriminator1_checkpoints[0])))
			self.G_D2.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], global_discriminator2_checkpoints[0])))
			self.L_D2.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], local_discriminator2_checkpoints[0])))
		
		self.R1.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], refiner1_checkpoints[0])))
		self.R2.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], refiner2_checkpoints[0])))
		self.C.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], classifier_checkpoints[0])))
		self.Reg.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], regressor_checkpoints[0])))
		self.Gaze.load_state_dict(torch.load(os.path.join(config['checkpoint_path']['pathname'], gaze_estimator_checkpoints[0])))
		
		return True

	def build_network(self):
		print("Building SimGAN Network")
		
		# init the network and load weights



		#device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.R1 = Refiner1().cuda()
		self.R2 = Refiner2().cuda()

		self.G_D1 = Global_Discriminator1().cuda()
		self.L_D1 = Local_Discriminator1().cuda()
		self.G_D2 = Global_Discriminator2().cuda()
		self.L_D2 = Local_Discriminator2().cuda()
		self.C = Classifier().cuda()
		#self.Reg = Regressor(3, 32, 34).cuda()
		self.Reg = Regressor(config['nstack']['param'], config['nfeatures']['param'], config['nlandmarks']['param']).cuda()
		self.Gaze = GazeEstimator(config['nstack']['param'], config['nfeatures']['param'], config['nlandmarks']['param']).cuda()
		self.vgg = models.vgg19(pretrained=True).features.cuda().eval()
		# If we are using cuda, place the models on the GPU
		
		if torch.cuda.device_count() > 1:
			print("Let's use", torch.cuda.device_count(), "GPUs!")

			#device_id = [0,1,2,3]
			device_id = config['gpu_num']['num']
			self.R1 = torch.nn.DataParallel(self.R1, device_ids=device_id)
			self.G_D1 = torch.nn.DataParallel(self.G_D1, device_ids=device_id)
			self.L_D1 = torch.nn.DataParallel(self.L_D1, device_ids=device_id)
			self.R2 = torch.nn.DataParallel(self.R2, device_ids=device_id)
			self.G_D2 = torch.nn.DataParallel(self.G_D2, device_ids=device_id)
			self.L_D2 = torch.nn.DataParallel(self.L_D2, device_ids=device_id)
			self.Reg = torch.nn.DataParallel(self.Reg, device_ids=device_id)
			self.Gaze = torch.nn.DataParallel(self.Gaze, device_ids=device_id)
			self.C = torch.nn.DataParallel(self.C, device_ids=device_id)
			#self.vgg = torch.nn.DataParallel(self.vgg, device_ids=[0,1])
			#for param in self.R.parameters():
			#	print(param.device)
		
		if config['train']['bool']:
			# Set optimizers
			self.refiner1_optimizer = torch.optim.Adam(self.R1.parameters(), lr=config['refiner_lr']['param'], betas=(0.5, 0.999))
			self.refiner2_optimizer = torch.optim.Adam(self.R2.parameters(), lr=config['refiner_lr']['param'], betas=(0.5, 0.999))

			self.global_discriminator1_optimizer = torch.optim.Adam(self.G_D1.parameters(), lr=config['discriminator_lr']['param'], betas=(0.5, 0.9))
			self.local_discriminator1_optimizer = torch.optim.Adam(self.L_D1.parameters(), lr=config['discriminator_lr']['param'], betas=(0.5, 0.9))

			self.global_discriminator2_optimizer = torch.optim.Adam(self.G_D2.parameters(), lr=config['discriminator_lr']['param'], betas=(0.5, 0.9))
			self.local_discriminator2_optimizer = torch.optim.Adam(self.L_D2.parameters(), lr=config['discriminator_lr']['param'], betas=(0.5, 0.9))
			
			self.classifier_optimizer = torch.optim.Adam(self.C.parameters(), lr=config['classifier_lr']['param'], betas=(0.5, 0.999))
			self.regressor_optimizer = torch.optim.Adam(self.Reg.parameters(), lr=config['regressor_lr']['param'])
			self.gaze_estimator_optimizer = torch.optim.Adam(self.Gaze.parameters(), lr=config['gaze_estimator_lr']['param'])


		self.weights_loaded = self.load_weights()
		# Set loss functions
		self.feature_loss = nn.L1Loss().cuda()
		#self.feature_loss = nn.MSELoss()
		self.cross_entropy_loss = nn.CrossEntropyLoss()
		print('Done building')

	'''Gradient Penalty for WGAN Loss'''
	def global_compute_gradient_penalty(self, D, real_samples, fake_samples):
		"""Calculates the gradient penalty loss for WGAN GP"""
    	# Random weight term for interpolation between real and fake samples
		Tensor = torch.cuda.FloatTensor
		alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
    	# Get random interpolation between real and fake samples
		interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
		d_interpolates = D(interpolates)
		fake = Variable(Tensor(real_samples.shape[0], 7680).fill_(1.0), requires_grad=False).cuda()
		# Get gradient w.r.t. interpolates
		gradients = autograd.grad(
			outputs=d_interpolates,
			inputs=interpolates,
			grad_outputs=fake,
			create_graph=True,
			retain_graph=True,
			only_inputs=True,
		)[0]
		gradients = gradients.view(gradients.size(0), -1)
		global_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
		return global_gradient_penalty

	def local_compute_gradient_penalty(self, D, real_samples, fake_samples):
		"""Calculates the gradient penalty loss for WGAN GP"""
    	# Random weight term for interpolation between real and fake samples
		Tensor = torch.cuda.FloatTensor
		alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
		
    	# Get random interpolation between real and fake samples
		interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
		d_interpolates = D(interpolates)
		fake = Variable(Tensor(real_samples.shape[0], 120).fill_(1.0), requires_grad=False).cuda()
		# Get gradient w.r.t. interpolates
		gradients = autograd.grad(
			outputs=d_interpolates,
			inputs=interpolates,
			grad_outputs=fake,
			create_graph=True,
			retain_graph=True,
			only_inputs=True,
		)[0]
		gradients = gradients.view(gradients.size(0), -1)
		local_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
		return local_gradient_penalty	


	# iterator for the data loader..
	# not really important
	def loop_iter(self, dataloader):
		while True:
			for data in iter(dataloader):
				yield data
			
			if not config['train']['bool']:
				print('Finished one epoch, done testing')
				self.testing_done = True

	# init data loader stuff
	def get_data_loaders(self):
		synthetic_data = Fake_Dataset()
		eval_synthetic_data = Eval_Fake_Dataset()
		
		#num = 24
		num = config['num_worker']['num']
		batchsize = config['batch_size']['size']
		self.synthetic_data_loader = Data.DataLoader(synthetic_data, batch_size=batchsize, shuffle=True, pin_memory=True, drop_last=True, num_workers=num)
		self.eval_data_loader = Data.DataLoader(eval_synthetic_data, batch_size=batchsize, shuffle=False, pin_memory=True, drop_last=True, num_workers=num)
		self.synthetic_data_iter = self.loop_iter(self.synthetic_data_loader)
		self.eval_data_iter = self.loop_iter(self.eval_data_loader)
		eval_real_data			 	  = Eval_Real_Dataset()
		real_data 				 	  = Real_Dataset()
		self.real_data_loader 	 	  = Data.DataLoader(real_data, batch_size=batchsize, shuffle=True, pin_memory=True, drop_last=True, num_workers=num)
		self.real_data_iter           = self.loop_iter(self.real_data_loader)
		self.eval_real_data_loader 	  = Data.DataLoader(eval_real_data, batch_size=batchsize, shuffle=False, pin_memory=True, drop_last=True, num_workers=num)
		self.eval_real_data_iter      = self.loop_iter(self.eval_real_data_loader)

	def print_refiner_info(self, step, pretrain=False):

		if not pretrain:
			#for step in tqdm(range(int(self.cfg.train_steps))):
			#	sleep(1)
			#	pass

			print('Step: {}'.format(step))
			print('Refiner... Cycle Fake Recon Loss: %.4f, Cycle Real Recon Loss: %.4f,Global refined adversarial loss: %.4f, Local refined adversarial loss: %.4f, Global real adversarial loss: %.4f, Local real adversarial loss: %.4f\n' % (self.cycle_fake_recon_loss.item(), self.cycle_real_recon_loss.item(), self.g_global_refined_adv_loss.item(), self.g_local_refined_adv_loss.item(), self.g_global_real_adv_loss.item(), self.g_local_real_adv_loss.item()))
			
		else:
			print('Step: {} / {}'.format(step, config['refiner_pretrain_iteration']['num']))
			print('Refiner... Cycle Fake Recon Loss: %.4f, Cycle Real Recon Loss: %.4f\n' % (self.cycle_fake_recon_loss.item(), self.cycle_real_recon_loss.item()))
	''' function to print some info on
		the discriminator during training '''
	def print_discriminator_info(self, step, pretrain=False):
		if not pretrain:
			print('Step: {}'.format(step))

		else:
			print('step: {} / {}'.format(step, config['discriminator_pretrain_iteration']['num']))
		
		print('Discriminator1... Global Discriminator Adversarial loss: %.4f, Local Discriminator Adversarial loss: %.4f\n' % (self.global_discriminator1_loss.item(), self.local_discriminator1_loss.item()))
		print('Discriminator2... Global Discriminator Adversarial loss: %.4f, Local Discriminator Adversarial loss: %.4f\n' % (self.global_discriminator2_loss.item(), self.local_discriminator2_loss.item()))

	def print_regressor_info(self, step, pretrain=False):
		if not pretrain:
			print('Step: {}'.format(step))
			print('Regressor... Synthetic Heatmap loss: %.4f, Synthetic Landmark loss: %.4f\n' %   (self.reg_synthetic_heatmap_loss.mean().item(), self.reg_synthetic_landmarks_loss.mean().item()))
			print('Regressor... Refined Heatmap loss: %.4f, Refined Landmark loss: %.4f\n' %   (self.reg_refined_heatmap_loss.mean().item(), self.reg_refined_landmarks_loss.mean().item()))

		else:
			print('step: {} / {}'.format(step, config['regressor_pretrain_iteration']['num']))
			print('Regressor... Synthetic Heatmap loss: %.4f, Synthetic Landmark loss: %.4f\n' %   (self.pre_reg_synthetic_heatmap_loss.mean().item(), self.pre_reg_synthetic_landmarks_loss.mean().item()))
				
	def print_classifier_info(self, step, pretrain=False):
		if not pretrain:
			print('Step: {}'.format(step))
			print('Classifier... Real Entropy loss: %.4f, Refined Entropy loss: %.4f, Real Accuracy:%.4f, Refined Accuracy:%.4f\n' %   (self.real_classifier_loss.item(), self.refined_classifier_loss.item(), self.real_accuracy.item(), self.refined_accuracy.item()))

		else:
			print('step: {} / {}'.format(step, config['classifier_pretrain_iteration']['num']))
			print('Classifier... Real Entropy loss: %.4f, Real_Accuracy:%.4f\n' %   (self.real_classifier_loss.item(), self.real_accuracy.item()))
	
	def print_gaze_estimator_info(self, step, pretrain=False):
		if not pretrain:
			print('Step: {}'.format(step))
			#print('Regressor... Refined Heatmap loss: %.4f, Refined Landmark loss: %.4f, Refined Gaze Loss:%.4f\n' %   (self.reg_refined_heatmap_loss.item(), self.reg_refined_landmarks_loss.item(), self.reg_refined_gaze_loss.item()))
			print('Gaze Estimator... Synthetic Gaze Loss:%.4f\n' %   (self.reg_return_synthetic_gaze_loss.item()))
			print('Gaze Estimator... Refined Gaze Loss:%.4f\n' %   (self.reg_return_refined_gaze_loss.item()))
			
		else:
			print('step: {} / {}'.format(step, config['gaze_estimator_pretrain_iteration']['num']))
			print('Gaze Estimator... Synthetic Gaze Loss:%.4f\n' %   (self.pre_return_gaze_loss.item()))
			