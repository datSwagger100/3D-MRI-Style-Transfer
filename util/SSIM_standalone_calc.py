import torch  
from torchvision import transforms
import torch.nn.functional as F 
import numpy as np
import math
from PIL import Image
import os
import cv2
from skimage import metrics as skimetric




def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = windo+w_size
    """    
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):

    # Generate an 1D tensor containing values sampled from a gaussian distribution
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
    
    # Converting to 2D  
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
     
    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window


def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    convert_img = transforms.ToTensor()
    img1 = convert_img(img1).unsqueeze(1) # Torch.conv2d only accepts (sample size, channels, h, w), need to add 1 dimension in the first place.
    img2 = convert_img(img2).unsqueeze(1)

    L = val_range # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2

    channels = 1
    
    try:
        _,  channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None: 
        real_size = min(window_size, height, width) # window should be atleast 11x11 
        window = create_window(real_size, channel=channels).to(img1.device)

    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability 
    C1 = (0.01 ) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03 ) ** 2 

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1  
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1 
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean() 
    else: 
        ret = ssim_score.mean(1).mean(1).mean(1)
    
    if full:
        return ret, contrast_metric
    
    return ret

# Helper: function to load images
load_images = lambda x: np.asarray(Image.open(x))

# Helper: functions to convert to Tensors
tensorify = lambda x: torch.Tensor(x.transpose((2, 0, 1))).unsqueeze(0).float()#.div(255.0)

# display imgs 
# def display_imgs(x, transpose=True, resize=True):
#   if resize:
#     x=cv2.resize(x, (400, 400))
#   if transpose:
#     cv2_imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
#   else:
#     cv2_imshow(x)

# Helper: Read all image paths in a folder into an array

def read_image_path_from_folder(folder):
    """
    Input: one folder path that contains all images
    Output: An list of png file names
    """

    filelist=os.listdir(folder)
    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".png")):
            filelist.remove(fichier)

    return filelist

def read_image(path):
    # img = np.moveaxis(np.asarray(Image.open(path)), -1, 0)
    pil = Image.open(path)
    return pil
def read_image_cv2(path):
    cv_img = cv2.imread(path)
    return cv_img

def psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR

def calc_PSNR(dir_path):
    fakes_path = dir_path + "/images/fake_B"
    reals_path = dir_path + "/images/real_B"

    fakes_list = read_image_path_from_folder(fakes_path)
    fakes_list.sort()
    reals_list = read_image_path_from_folder(reals_path)
    reals_list.sort()
    
    score_list = []
    for i in range(len(fakes_list)):
        psnr_score = psnr(read_image_cv2(os.path.join(fakes_path, fakes_list[i])), read_image_cv2(os.path.join(reals_path, reals_list[i])))
        print(fakes_list[i], ',',reals_list[i], ',', psnr_score)
        score_list.append(psnr_score)
    avg = sum(score_list) / len(score_list) 
    print('-------------------------------------------------')  
    print(' PSNR across the test sets: ',',', avg)  
    return

def calc_pytorch_SSIM(dir_path):
    fakes_path = dir_path + "/images/fake_B"
    reals_path = dir_path + "/images/real_B"

    fakes_list = read_image_path_from_folder(fakes_path)
    fakes_list.sort()
    reals_list = read_image_path_from_folder(reals_path)
    reals_list.sort()

    score_list = []
    for i in range(len(fakes_list)):

        real = Image.open(os.path.join(reals_path, reals_list[i])).convert('L')
        impaired = Image.open(os.path.join(fakes_path, fakes_list[i])).convert('L')
        score = ssim(real, impaired, val_range = 255, window_size=11, window=None, size_average=True, full=False)
        print(fakes_list[i], reals_list[i], score)
        score_list.append(score)
    avg = sum(score_list) / len(score_list) 
    print('-------------------------------------------------')  
    print('final average of PyTorch SSIM across the test sets: ', avg)  
    return

def calc_skimage_SSIM(dir_path):
    fakes_path = dir_path + "/images/fake_B"
    reals_path = dir_path + "/images/real_B"

    fakes_list = read_image_path_from_folder(fakes_path)
    fakes_list.sort()
    reals_list = read_image_path_from_folder(reals_path)
    reals_list.sort()

    score_list = []
    for i in range(len(fakes_list)):
        real = np.asfarray(read_image(os.path.join(reals_path, reals_list[i])).convert('L'))
        impaired = np.asfarray(read_image(os.path.join(fakes_path, fakes_list[i])).convert('L'))
        score = skimetric.structural_similarity(real, impaired, multichannel=False, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
        #score = ssim(read_image(os.path.join(fakes_path, fakes_list[i])), read_image(os.path.join(reals_path, reals_list[i])), val_range = 255, window_size=11, window=None, size_average=True, full=False)
        print(fakes_list[i], reals_list[i], score)
        score_list.append(score)
    avg = sum(score_list) / len(score_list) 
    print('-------------------------------------------------')  
    print('final average of Skimage SSIM across the test sets: ', avg)  
    return

def calc_cv2_SSIM(dir_path):

    fakes_path = dir_path + "/images/fake_B"
    reals_path = dir_path + "/images/real_B"

    fakes_list = read_image_path_from_folder(fakes_path)
    fakes_list.sort()
    reals_list = read_image_path_from_folder(reals_path)
    reals_list.sort()

    score_list = []
    for i in range(len(fakes_list)):
        
        first = cv2.imread(os.path.join(reals_path, reals_list[i]))
        second = cv2.imread(os.path.join(fakes_path, fakes_list[i]))

        # Convert images to grayscale
        first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

        # Compute SSIM between two images
        score, diff = skimetric.structural_similarity(first_gray, second_gray, full=True)
        print(fakes_list[i], reals_list[i], "Similarity Score: {:.3f}%".format(score * 100))

        # The diff image contains the actual image differences between the two images
        # and is represented as a floating point data type so we must convert the array 
        # to 8-bit unsigned integers in the range [0,255] before we can use it with OpenCV
        diff = (diff * 255).astype("uint8")

        # Threshold the difference image, followed by finding contours to
        # obtain the regions that differ between the two images
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        # # Highlight differences
        # mask = np.zeros(first.shape, dtype='uint8')
        # filled = second.copy()

        # for c in contours:
        #     area = cv2.contourArea(c)
        #     if area > 100:
        #         x,y,w,h = cv2.boundingRect(c)
        #         cv2.rectangle(first, (x, y), (x + w, y + h), (36,255,12), 2)
        #         cv2.rectangle(second, (x, y), (x + w, y + h), (36,255,12), 2)
        #         cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        #         cv2.drawContours(filled, [c], 0, (0,255,0), -1)

        # cv2.imshow('first', first)
        # cv2.imshow('second', second)
        # cv2.imshow('diff', diff)
        # cv2.imshow('mask', mask)
        # cv2.imshow('filled', filled)
        # cv2.waitKey()

        print(fakes_list[i], reals_list[i], score)
        score_list.append(score)

    avg = sum(score_list) / len(score_list) 
    print('-------------------------------------------------')  
    print('final average of cv2 SSIM across the test sets: ', avg)  

    return
# Enter the path of the folder contains 2 subfolder: real_B and fake_B
path = "/home/home/bran_stu/results/BraTS_Ingenia_UNET_GAN+L1/test_latest"

# calc_pytorch_SSIM(path)
# calc_skimage_SSIM(path)
# calc_cv2_SSIM(path)
calc_PSNR(path)

#print(skimetric.structural_similarity(ref_image, impaired_image, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))

# Compute SSIM between two images


    















