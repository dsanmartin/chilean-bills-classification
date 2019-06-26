# Import necessary components to build LeNet
import os
import cv2
import json
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
from keras import losses
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD
from datetime import datetime
import pathlib
#from preprocessing import createDataset
from sklearn.model_selection import train_test_split
#%%

#Model and X and y test directory
folder = datetime.today().strftime('%Y%m%d%H%M%S')
pathlib.Path('data/output/models/' + str(folder)).mkdir(parents=True, exist_ok=True)
pathlib.Path('data/input').mkdir(parents=True, exist_ok=True)

DIR_BASE = "data/"
# Define canvas size 
WIDTH = 150
HEIGHT = 150

# CV color option
READ_COLOR = 1
TRAN_COLOR = cv2.COLOR_BGR2RGB # cv2.COLOR_BGR2GRAY

seed = 666
#%%
def alexnet_model(img_shape=(HEIGHT, WIDTH, 1), n_classes=5, l2_reg=0.,
	weights=None):

	# Initialize model
	alexnet = Sequential()

	# Layer 1
	alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
		padding='same', kernel_regularizer=l2(l2_reg)))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 2
	alexnet.add(Conv2D(256, (5, 5), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 3
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(512, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

#	# Layer 4
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
#
#	# Layer 5
#	alexnet.add(ZeroPadding2D((1, 1)))
#	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
#	alexnet.add(BatchNormalization())
#	alexnet.add(Activation('relu'))
#	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 6
	alexnet.add(Flatten())
	alexnet.add(Dense(32))#(3072))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 7
	alexnet.add(Dense(64))#(4096))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 8
	alexnet.add(Dense(n_classes))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('softmax'))

	if weights is not None:
		alexnet.load_weights(weights)

	return alexnet

#%% LOAD DATA
a1k = cv2.cvtColor(cv2.imread(DIR_BASE + "1000/anverso.jpg", READ_COLOR), TRAN_COLOR)
a2k = cv2.cvtColor(cv2.imread(DIR_BASE + "2000/anverso.jpg", READ_COLOR), TRAN_COLOR)
a5k = cv2.cvtColor(cv2.imread(DIR_BASE + "5000/anverso.jpg", READ_COLOR), TRAN_COLOR)
a10k = cv2.cvtColor(cv2.imread(DIR_BASE + "10000/anverso.jpg", READ_COLOR), TRAN_COLOR)
a20k = cv2.cvtColor(cv2.imread(DIR_BASE + "20000/anverso.jpg", READ_COLOR), TRAN_COLOR)

r1k = cv2.cvtColor(cv2.imread(DIR_BASE + "1000/reverso.jpg", READ_COLOR), TRAN_COLOR)
r2k = cv2.cvtColor(cv2.imread(DIR_BASE + "2000/reverso.jpg", READ_COLOR), TRAN_COLOR)
r5k = cv2.cvtColor(cv2.imread(DIR_BASE + "5000/reverso.jpg", READ_COLOR), TRAN_COLOR)
r10k = cv2.cvtColor(cv2.imread(DIR_BASE + "10000/reverso.jpg", READ_COLOR), TRAN_COLOR)
r20k = cv2.cvtColor(cv2.imread(DIR_BASE + "20000/reverso.jpg", READ_COLOR), TRAN_COLOR)
#%%
pa = [.5] * 5 # Proporcion Anversos
th = [0.5] * 5 # Umbrales
bc = [5] * 5 # Numero de billetes por clase
data_anv = [a1k, a2k, a5k, a10k, a20k]
data_rev = [r1k, r2k, r5k, r10k, r20k]
X, y = createDataset(data_anv, data_rev, pa, th, bc, HEIGHT, WIDTH)
np.save('data/input/X', X)
np.save('data/input/y', y)
X = np.load('data/input/X.npy')
y = np.load('data/input/y.npy')
#%%
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.33, random_state=seed, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.33, random_state=seed, stratify=y_tr)
#%%
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_val = np_utils.to_categorical(y_val)
#%%
np.save('data/output/models/' + str(folder) + '/X_test', X_test)
np.save('data/output/models/' + str(folder) + '/y_test', y_test)
#%%
del X, data_anv, data_rev, a1k, a2k, a5k, a10k, a20k, r1k, r2k, r5k, r10k, r20k
#%%
lr_ = .5
epochs_ = 1
batch_size_ = 50
sgd = SGD(lr=lr_)
alexnet = alexnet_model(img_shape=X_train[0].shape)
alexnet.compile(optimizer=sgd, loss=losses.categorical_crossentropy)
#%%
hist = alexnet.fit(X_train, y_train, epochs=epochs_, batch_size=batch_size_, verbose=1, 
                 validation_data=(X_val, y_val))
#%%
alexnet.save('data/output/models/' + str(folder)+ '/model.h5')
#%%
json.dump(hist.history, open('data/output/models/' + str(folder) + '/history.json', 'w'))