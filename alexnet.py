import json
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
from keras import losses
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD

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


    def compile(self, lr=0.1, momentum=0.0, decay=0.0, nesterov=False):
        sgd = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)
        self.model.compile(optimizer=sgd, loss=losses.categorical_crossentropy)
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=10, verbose=1):
        hist = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, 
                     validation_data=(X_val, y_val))
        self.model.save(self.output_dir + 'model.h5')
        json.dump(hist.history, open(self.output_dir + 'history.json', 'w'))