import glob
import PIL
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
from os import path
from pylab import rcParams
import torch
#import PIL
#print(PIL.PILLOW_VERSION)
import glob
import socket
import csv
import pandas as pd
import keras
from sklearn.utils import shuffle

#3. Illu
# Cau truc thu muc cua Pre_process va data_path la:
#illu_in va fl_origin_illu la original folder, sau do la cac Emotional folders, cuoi cung la imgs.
#illu_in la folder nguon, fl_origin_illu la facial_illumination

import logging
import numpy as np

# Homomorphic filter class
class HomomorphicFilter:
    def __init__(self, a = 0.5, b = 1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H = None):
        #  Validating image
        if len(I.shape) is not 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain 
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter=='butterworth':
            H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='gaussian':
            H = self.__gaussian_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='external':
            print('external')
            if len(H.shape) is not 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')
        
        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I = I_fft, H = H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return np.uint8(I)
# End of class HomomorphicFilter
if __name__ == "__main__":
  img_illu_out = []
  fl_illu_imgout = []
  img_data_list=[]
  n=0
  illu_in = '../data/train'
  fl_illu_in = os.listdir(illu_in)
  
  #Create folder output ill:
  fl_origin_illu = '../data_af_preprocessing/Illu_data/train'
  

  #Run illumination
  for fd_crop in fl_illu_in:
    if not os.path.isdir(fl_origin_illu + '/'+ fd_crop):
      fl_illu_imgout = fl_origin_illu + '/'+ fd_crop
      os.mkdir(fl_illu_imgout)
      print("Creating" + fd_crop )
    crop_list = os.listdir(illu_in + '/' + fd_crop)
    for crop_img in crop_list:
      n=n+1
      img_path_in = illu_in + '/'+ fd_crop + '/'+crop_img
      #img_illu_out = os.path.join(fl_illu_imgout , crop_img)
      img_path_out  = fl_origin_illu+'/' + fd_crop+ '/'+ crop_img

      img = cv2.imread(img_path_in)[:, :, 0]
      homo_filter = HomomorphicFilter(a = 0.75, b = 1.25)
      img_filtered = homo_filter.filter(I=img, filter_params=[30,2])
      cv2.imwrite(img_path_out, img_filtered)
      print("Done iluminate ", crop_img)
