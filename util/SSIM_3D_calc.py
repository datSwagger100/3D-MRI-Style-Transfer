

import sys
from torchvision import transforms
import torch.nn.functional as F 
import numpy as np
import math
from PIL import Image
import os
sys.path.insert(0, '/home/home/bran_stu/3D-MRI-style-transfer/SSIM_3D_calc.py')

from pytorch_ssim_3D_master import pytorch_ssim
import torch
from torch.autograd import Variable
from torch import optim
import numpy as np
import nibabel as nib


def read_image_path_from_folder(folder):
    """
    Input: one folder path that contains all images
    Output: An list of png file names
    """

    filelist=os.listdir(folder)
    for file in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(file.endswith(".nii.gz")):
            filelist.remove(file)

    return filelist

def read_image(path):
    # img = np.moveaxis(np.asarray(Image.open(path)), -1, 0)
    label = nib.load(path)#!Image.open(os.path.join(self.seg_dir,label_name))
    #change to numpy
    nib_array = np.array(label.dataobj)
    return nib_array

def calc_SSIM_3D(dir_path):
    fakes_path = dir_path + "/images/fake_B"
    reals_path = dir_path + "/images/real_B"

    fakes_list = read_image_path_from_folder(fakes_path)
    fakes_list.sort()
    reals_list = read_image_path_from_folder(reals_path)
    reals_list.sort()

    score_list = []
    for i in range(len(fakes_list)):
        if torch.cuda.is_available():
            convert_img = transforms.ToTensor()
            img1_4D = convert_img(read_image(os.path.join(fakes_path, fakes_list[i]))).unsqueeze(0) # Torch.conv2d only accepts (sample size, channels, h, w), need to add 1 dimension in the first place.
            img2_4D = convert_img(read_image(os.path.join(reals_path, reals_list[i]))).unsqueeze(0)

            img1 = img1_4D.unsqueeze(0) # Torch.conv2d only accepts (sample size, channels, h, w), need to add 1 dimension in the first place.
            img2 = img2_4D.unsqueeze(0)
            tgt_fake = img1.cuda()
            tgt_real = img2.cuda()

            ssim_loss_3D = pytorch_ssim.SSIM3D(window_size = 11)
            score = ssim_loss_3D.forward(tgt_fake, tgt_real)
            print(fakes_list[i],',', reals_list[i],',', score.item())
            score_list.append(score.item())

    avg = sum(score_list) / len(score_list) 
    print('-------------------------------------------------')  
    print('final average of SSIM 3D across the test sets: ', avg)  

# Enter the path of the folder contains 2 subfolder: real_B and fake_B
path = "/home/home/bran_stu/results/BraTS_Ingenia_SIT_GAN+L1/test_latest"
calc_SSIM_3D(path)
