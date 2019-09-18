# Import necessary components to build LeNet
import os
import json
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
from keras import losses
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

SEED = 666

def dataPre(X, y, test_size=0.33, path=None):
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=test_size, random_state=SEED, stratify=y_tr)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    y_val = np_utils.to_categorical(y_val)
    
    if path is not None:
        np.save(path + 'X_test', X_test)
        np.save(path + 'y_test', y_test)
    
        return X_train, y_train, X_val, y_val
    
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test
        

class Alexnet:
    
    def __init__(self, input_shape, output_dir, n_classes=5, l2_reg=0., weights=None):
        self.input_shape = input_shape
        self.output_dir = output_dir
        self.n_classes = n_classes
        self.l2_reg = l2_reg
        self.weights = weights
        self.model = self.createModel()
        
    
    def createModel(self):
    
    	# Initialize model
    	alexnet = Sequential()
    
    	# Layer 1
    	alexnet.add(Conv2D(96, (11, 11), input_shape=self.input_shape,
    		padding='same', kernel_regularizer=l2(self.l2_reg)))
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
    
    	# Layer 4
    	alexnet.add(ZeroPadding2D((1, 1)))
    	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    	alexnet.add(BatchNormalization())
    	alexnet.add(Activation('relu'))
    
    	# Layer 5
    	alexnet.add(ZeroPadding2D((1, 1)))
    	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    	alexnet.add(BatchNormalization())
    	alexnet.add(Activation('relu'))
    	alexnet.add(MaxPooling2D(pool_size=(2, 2)))
    
    	# Layer 6
    	alexnet.add(Flatten())
    	alexnet.add(Dense(3072))#(32))
    	alexnet.add(BatchNormalization())
    	alexnet.add(Activation('relu'))
    	alexnet.add(Dropout(0.5))
    
    	# Layer 7
    	alexnet.add(Dense(4096))#(64))
    	alexnet.add(BatchNormalization())
    	alexnet.add(Activation('relu'))
    	alexnet.add(Dropout(0.5))
    
    	# Layer 8
    	alexnet.add(Dense(self.n_classes))
    	alexnet.add(BatchNormalization())
    	alexnet.add(Activation('softmax'))
    
    	if self.weights is not None:
    		alexnet.load_weights(self.weights)
    
    	return alexnet


    def compile(self, lr=.5):
        sgd = SGD(lr=lr)
        self.model.compile(optimizer=sgd, loss=losses.categorical_crossentropy)
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=1, batch_size=1, verbose=1):
        hist = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, 
                     validation_data=(X_val, y_val))
        self.model.save(self.output_dir + 'model.h5')
        json.dump(hist.history, open(self.output_dir + 'history.json', 'w'))
    
    



