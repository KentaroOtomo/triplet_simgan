''' 
	Config file for training the refiner / discriminator networks 

'''
import torch

''' ---- Everything below is pretty much hyperparameters to change ----- '''

# path to the synthetic data set (a folder with all the synthetic images)
#synthetic_path  = '../data/nvgaze/'
#synthetic_path = "../data/imgs_ue3"
#synthetic_path  = '../data/classifier_image/'
#synthetic_path  = '/home/silvias15/simgan-torch/data/alldata/'

# path to the real data set (a folder with all the real images)
#real_path	= '../data/imgs_yzk_left/'
#real_path	= '../data/new_real/'	

#save_refined_path 	= '/media/silvias15/40e69c69-43ae-4816-b6c5-c46f45bf6469/simgan-torch/data/save/'
# If we want to log the model's generated outputs, set to true.
log 		= True																							# If log == True, a display of refined images will be shown during training

# Save the refiners output every log_interval steps (batches)
log_interval 	= 2000																					# log_interval == # of steps before changing the display

# Which cuda device would you like to use?
cuda_nums 	= [5, 6, 7, 8, 9]																	# what nvidia-gpu to use

# regularization parameter on the reconstruction loss (commonly called lambda)
delta 		= 0.7

# number of steps (batches) to train for
train_steps	= 150000

# size of the buffer aka image pool, see paper for more details
buffer_size	= 6

# momentum value to use for optimization (stochastic gradient descent)
momentum 	= .9

# which batch size to use
batch_size	= 16

# which learning rate to use
lr  		= 0.001			

# number of steps to pretrain the refiner with
r_pretrain	= 5000
#r_pretrain	= 100

# number of steps to pretrain the discriminator, using only MSE loss
d_pretrain	= 2000
#d_pretrain = 100

# number of steps to pretrain the classifier
#c_pretrain = 1000
#c_pretrain = 10
# number of steps to pretrain the regressor
reg_pretrain = 15000
#reg_pretrain = 100
gaze_pretrain = 15000
#gaze_pretrain = 100
#reg_pretrain = 100
# number of updates to the refiner per step
k_r		= 1	

# number of updates to the discriminator per step
k_d		= 2

# number of updates to the classifier per step
#k_c     = 1

# number of updates to the regressor per step
k_reg   = 1

k_gaze  = 1

# interval to print model status such as loss, measured by step (batch) number
#print_interval  = 1000
print_interval  = 100	

# interval to save model checkpoints (weights), measured by step (batch) number
#save_interval	= 2000

save_interval = 5000
# path to save checkpoints to
#checkpoint_path = checkpoint_root + 'checkpoint_lr0001_bs512_deltaP5_unity/'
#checkpoint_path = '../' +  'reg_test_without_heatmaploss/'
#checkpoint_path = '../' +  'reg_test_with_heatmaploss/'
#checkpoint_path = '../' +  'instancenorm_24/'
#checkpoint_path = '../' +  'new_realdata_cyclegan_withclassifier/'
#checkpoint_path = '/home/silvias15-local/' +  'old_realdata_cyclegan_withclassifier_noreconloss/'
checkpoint_path = '../' +  'test_mpiigaze_lr0.00005/'
#checkpoint_path = '../' +  'old_realdata_cyclegan_noclassifier/'
#checkpoint_path = '../' +  'new_realdata_cyclegan/'
#checkpoint_path = '../' + 'train_regressor/'
''' ----- No need to change anything below ----- '''
# name to save discriminator checkpoint as, don't change this or you have to change load functions
G_D1_path		= 'G_D1_%d.pkl'		
L_D1_path		= 'L_D1_%d.pkl'
pretrain_G_D1_path		= 'G_D1_0.pkl'		
pretrain_L_D1_path		= 'L_D1_0.pkl'
G_D2_path		= 'G_D2_%d.pkl'		
L_D2_path		= 'L_D2_%d.pkl'
pretrain_G_D2_path		= 'G_D2_0.pkl'		
pretrain_L_D2_path		= 'L_D2_0.pkl'
# name to save refiner checkpoint as, don't change this or you have to change load functions
R1_path		= 'R1_%d.pkl'
pretrain_R1_path		= 'R1_0.pkl'
R2_path		= 'R2_%d.pkl'
pretrain_R2_path		= 'R2_0.pkl'
#C_path	= 'C_%d.pkl'
#pretrain_C_path	= 'C_0.pkl'

# name to save classifier checkpoint as, don't change this or you have to change load functions
#%C_path		= 'C_%d.pkl'

# name to save regressor checkpoint as, don't change this or you have to change load functions
Reg_path	= 'Reg_%d.pkl'
pretrain_Reg_path	= 'Reg_0.pkl'

Gaze_path   = 'Gaze_%d.pkl'
pretrain_Gaze_path	= 'Gaze_0.pkl'
# name to save optimizer checkpoint (current step, etc), don't change this or you have to change load functions
optimizer_path	= 'optimizer_status.pkl'

# if gpu is available, true.. else false 
cuda_use 	= torch.cuda.is_available()		

# train = True because this is the train_config file!...
train 		= True		
