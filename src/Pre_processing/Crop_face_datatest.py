import dlib
import glob
import PIL
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
from os import path
from pylab import rcParams
from __future__ import print_function
import torch
#import PIL
#print(PIL.PILLOW_VERSION)
import glob
import socket
import csv
import pandas as pd
import keras
from sklearn.utils import shuffle

#2.Crop FACE

# Cau truc thu muc cua Pre_process va data_path la:
#data_directory va result_directory la original folder, sau do la cac Emotional folders, cuoi cung la imgs.
#data_directory la folder nguon imgs, result_directory la facial_crop_img

#crop face step
## DIRECTORY of the images
data_directory = "../data/test"
data_dir_list = os.listdir(data_directory)

#Tao folder result crop face
result_directory = '../data_af_preprocessing/Cropped_facial_data/test'


## directory where the images to be saved:
#f_directory = "/content/gdrive/My Drive/ML Backup/jaffe"
def facecrop(image_path, result_img_path):
    ## Crops the face of a person from any image!

    ## OpenCV XML FILE for Frontal Facial Detection using HAAR CASCADES.
    face_model = "../Pre_processing/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(face_model)
    #cv2.CascadeClassifier.load(facedata)
    ## Reading the given Image with OpenCV
    img = cv2.imread(image_path)

    try:
        ## Some downloaded images are of unsupported type and should be ignored while raising Exception, so for that
        ## I'm using the try/except functions.

        faces = cascade.detectMultiScale(img)
        for (x,y,w,h) in faces:
            
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            sub_face = img[y:y+h, x:x+w]
            cv2.imwrite(result_img_path, sub_face)

            print("writing: ", result_img_path)
            #f_name = image.split('/')
    except Exception as e:
        print("exception: ", e)
if __name__ == '__main__':
    for dataset in data_dir_list:
        print("emotion: ", dataset)
        emotion_folder = os.path.join(data_directory, dataset)        
        if not os.path.exists(result_directory +"/"+ dataset):
          os.mkdir(result_directory +"/"+ dataset)
          result_emotion_folder = result_directory +"/"+ dataset
          print("Creating folder: ", result_emotion_folder)
        img_list=next(os.walk(emotion_folder))[2]     
        for img_name in img_list:
          file_path = os.path.join(emotion_folder, img_name)
          result_path = os.path.join(result_emotion_folder, img_name)
          print("processing: ", file_path)
          facecrop(file_path, result_path)