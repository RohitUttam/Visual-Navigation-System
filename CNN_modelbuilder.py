import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import glob
import copy
import numpy as np
import scipy.misc as scpm

import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))
path=dir_path+'Frames/'

#Image size chosen for processing
width, height=[192,108]

#Gray scale images loader
def load_images(Nodo_path):
    filenames= glob.glob(Nodo_path)
    images =np.asarray([cv2.resize(scpm.imread(file, mode= 'L'),(width,height)) for file in filenames])
    return images

#Node_X loads images assigned to Node X, eX assigns labels
labels=[]
nodes=[]

for x in range(1, 2):
    globals()['Nodo_%s' % x] = load_images(path+'Nodo_'+str(x)+'/*.jpg')
    globals()['e%s' % x] =[x]*len(globals()['Nodo_%s' % x])
    labels+=(globals()['e%s' % x])

nodes=Nodo_1
for i in [Nodo_2, Nodo_3, Nodo_4, Nodo_5, Nodo_6]:
    nodes= np.append(nodes,i, axis=0)

c=np.array(labels)

#Split 80%/20% for training and test.
tr=round(0.8*len(c))
te=len(c)-tr
training_idx = np.random.randint(nodes.shape[0], size=tr)
test_idx = np.random.randint(nodes.shape[0], size=te)
train_X, test_X = nodes[training_idx,:], nodes[test_idx,:]
train_Y, test_Y = c[training_idx], c[test_idx]
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

#Categorical data must start from 0
train_Y=train_Y-1
test_Y=test_Y-1

# Find the unique numbers from the train labels
classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

##data preprocessing
train_X = train_X.reshape(-1, height,width, 1)
test_X = test_X.reshape(-1, height,width, 1)
print(train_X.shape, test_X.shape)


train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

#Split training data into train and validation
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

#Choose batch size, epochs and number of classes
batch_size = 40
epochs = 10
num_classes = 6

#Build model
f_model = Sequential()
f_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(height,width,1),padding='same'))
f_model.add(LeakyReLU(alpha=0.1))
f_model.add(MaxPooling2D((2, 2),padding='same'))
f_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
f_model.add(LeakyReLU(alpha=0.1))
f_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
f_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
f_model.add(LeakyReLU(alpha=0.1))                  
f_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
f_model.add(Flatten())
f_model.add(Dense(128, activation='linear'))
f_model.add(LeakyReLU(alpha=0.1))                  
f_model.add(Dense(num_classes, activation='softmax'))

f_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

#Summary of model
print(f_model.summary())

#Train model
f_train = f_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

#Test model
test_eval = f_model.evaluate(test_X, test_Y_one_hot, verbose=0)


print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

#Save model
f_model.save(path+'modelo2.h5')

accuracy = f_train.history['acc']
val_accuracy = f_train.history['val_acc']
loss = f_train.history['loss']
val_loss = f_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()