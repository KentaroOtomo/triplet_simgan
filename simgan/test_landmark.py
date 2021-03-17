'''
	This file includes the class to train SimGAN
'''

import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import cv2
import pandas as pd
import csv
import time

from sub_simgan import SubSimGAN

class TestLandmarkSimGAN(SubSimGAN):
    def __init__(self, cfg):
        SubSimGAN.__init__(self, cfg)
        
        self.feature_loss = None
        self.adv_loss = None
        self.loss_real = None
        self.loss_refined = None

        self.cfg = cfg	

    def refine(self):
        self.build_network()	
        self.get_data_loaders()

        ''' If no saved weights are found,
            pretrain the refiner / discriminator '''
        if not self.weights_loaded:
            print('Returning because no weights were found!!')
            return

        #for images, imageFiles in self.test_data_loader:
        #    #self.G_D.eval()
        for param in self.G_D1.parameters():
            param.requires_grad = False
        for param in self.G_D2.parameters():
            param.requires_grad = False    

        #self.L_D.eval()
        for param in self.L_D1.parameters():
            param.requires_grad = False
        for param in self.L_D2.parameters():
            param.requires_grad = False        	
                    
        #self.R.eval()
        for param in self.R1.parameters():
            param.requires_grad = False
        for param in self.R2.parameters():
            param.requires_grad = False    	

        #self.Reg.Eval()
        for param in self.Reg.parameters():
            param.requires_grad = False
        
        img_dir = config['real_image_directory']['dirname']
        t1 = time.time()
        img_num = 1
        img_max = len(img_dir)
        print(img_max)
        filename = config['test_output_csv']['csv_name']
        save_dir = config['test_output_directory']['dirname']
        with open(filename, 'w', newline="") as f:
            fieldnames = ["num", "landmark_center_x", "landmark_center_y", "gaze_x", "gaze_y"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            while img_num <= img_max:
                #print(img_num)
                #img = Image.open(os.path.join("../data/imgs_yzk_left/", "real_{}.png".format(img_num)))
                #print(img_num)
                
                img = Image.open(os.path.join(img_dir, "{}.png".format(img_num)))
                #print(img)
                """
                cg_img = Image.open(os.path.join("/home/silvias15-local/data/eval_imgs_500/", "{}.png".format(img_num)))
                cgimg_tensor = transforms.Grayscale(num_output_channels=1)(cg_img)
                cgimg_tensor = transforms.ToTensor()(cg_img)
                cgimg_tensor = torch.clamp(cgimg_tensor,0, 255)
                
                cgimg_tensor = cgimg_tensor.unsqueeze(0).cuda()
                refined_img = self.R1(cgimg_tensor)
                save_refined_img = refined_img.squeeze(0)
                refined_img2 = torch.clamp(refined_img,0, 255)
                refined_img2 = refined_img.squeeze(0).cpu()
                refined_img3 = transforms.ToPILImage()(refined_img2)
                refined_img3.save(os.path.join(save_dir, "refined_{}.png".format(img_num)))
                """
                img_tensor = transforms.ToTensor()(img)
                img_tensor = img_tensor.unsqueeze(0).cuda()
                #print(img_tensor)
                heatmaps_pred, landmarks_pred_default = self.Reg(img_tensor)
                cv2_img = np.array(img, dtype=np.uint8)
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
                landmarks_pred = landmarks_pred_default.cpu().detach().numpy()[0, :]
                for (y, x) in landmarks_pred[0:15]:
                    cv2.circle(cv2_img, (int(x), int(y)), 1, (255, 0, 0), -1)
                for (y, x) in landmarks_pred[16:31]:
                    cv2.circle(cv2_img, (int(x), int(y)), 1, (0, 255, 0), -1)    
                for (y, x) in landmarks_pred[32:33]:
                    cv2.circle(cv2_img, (int(x), int(y)), 1, (0, 0, 255), -1)    
                #print('image_num = ', img_num)
                #print(landmarks_pred.shape)
                #print('iris center = ', landmarks_pred[32][0])    
                gaze_pred = self.Gaze(img_tensor, landmarks_pred_default)
                row = {"num": img_num, "landmark_center_x": landmarks_pred[32][0], "landmark_center_y": landmarks_pred[32][1], "gaze_x": float(gaze_pred[0][0]), "gaze_y": float(gaze_pred[0][1])}
                writer.writerow(row)
                
                cv2.imwrite(os.path.join(save_dir, "{}.png".format(img_num)), cv2_img)
                hm_pred2 = heatmaps_pred.cuda()
                hm_pred = hm_pred2.squeeze(0)
                hm_pred = np.mean(hm_pred[-1, 0:34].cpu().detach().numpy(), axis=0)
                hm_pred = cv2.normalize(hm_pred, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cv2.imwrite('pred.png', hm_pred*255)
                hm_pred_color = cv2.imread('pred.png')
                
                i = 0
                while i <= 35:
                    j = 0
                    while j <= 59:
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
                #save_refined_image = refined_img.squeeze(0)
                img_tensor2 = img_tensor.squeeze(0)
                save_refined_img_3ch = torch.cat((img_tensor2, img_tensor2, img_tensor2), dim=0)
                save_refined_img_pil = transforms.ToPILImage()(save_refined_img_3ch.cpu())
                save_refined_img_np = np.asarray(save_refined_img_pil, dtype=np.uint8)
                save_refined_img_np = cv2.cvtColor(save_refined_img_np, cv2.COLOR_RGB2BGR)
                alpha = 0.3
                refined_blended_image = cv2.addWeighted(hm_pred_color, alpha, save_refined_img_np, 1-alpha, 0)
                refined_blended_image_cv2 = cv2.cvtColor(refined_blended_image, cv2.COLOR_BGR2RGB)
                #print(refined_blended_image_cv2.shape)
                refined_blended_image_pil = Image.fromarray(refined_blended_image_cv2)
                cv2.imwrite(os.path.join(save_dir, "heatmap_{}.png".format(img_num)), refined_blended_image)
                #refined_blended_image.save(os.path.join("/disks/dl310/silvias15/old_realdata_cyclegan_withclassifier_noreconloss_output/", "heatmap_{}.png".format(img_num)), quality=95)
                                    
                img_num = img_num + 1
            else:
                img_num = img_num + 1   
            t2 = time.time()
        print("elapsed time:", t2-t1)