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
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

def load_videos_filename(directory, filePattern):
    return glob.glob(directory + filePattern)

def load_video(file):
    return cv2.VideoCapture(file)

def test(gray):
    test_XX = gray.reshape(-1, height,width, 1)
    test_XX = test_XX.astype('float32')
    test_XX = test_XX / 255.
    labels = model.predict_classes(test_XX)+1
    return labels


TEST_VIDEO = dir_path +'/Defensa.MOV'

model = load_model(dir_path+'/modelo2.h5')


width, height=[192,108]

cap = load_video(TEST_VIDEO)
actual_node = 1
next_node = actual_node + 1
node_detected = -1
times_detected = 0


while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == False:
        break
    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(width,height))
    node_detected=test(gray)

    actual_node_name='Node: ' + str(node_detected)

    cv2.putText(frame, actual_node_name, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 153, 255), 13, cv2.LINE_AA)
    cv2.namedWindow('Test Dynamic', cv2.WINDOW_FULLSCREEN)
    #cv2.resizeWindow('Test Dynamic', 300, 300)
    cv2.imshow('Test Dynamic', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()