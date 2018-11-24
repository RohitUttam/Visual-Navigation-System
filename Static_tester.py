#librerias usadas
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import glob
import copy
import numpy as np
import scipy.misc as scpm
import PIL
from PIL import Image
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

#carga del modelo
model = load_model(dir_path+'/modelo2.h5')


width, height=[192,108]

def load_images(Nodo_path):
    filenames= sorted(glob.glob(Nodo_path))
    images =np.asarray([scpm.imread(file) for file in filenames])
    images_res =np.asarray([cv2.resize(scpm.imread(file, mode= 'L'),(width,height)) for file in filenames])
    return images, images_res

def test():
    test_X, test_XX= load_images('C:/Users/Rohit/Desktop/Master/Robots Au/Práctica_II/Robots-autonomos-P2-master/test/*.jpg')
    test_XX = test_XX.reshape(-1, height,width, 1)
    test_XX = test_XX.astype('float32')
    test_XX = test_XX / 255.
    labels = model.predict_classes(test_XX)+1
    return test_X, labels

def result(numero):
    test_X,labels=test()
    print('Nodo:', labels[numero])
    plt.imshow(test_X[numero])
    plt.show()

result(int(input('Enter file number: '))-1)
