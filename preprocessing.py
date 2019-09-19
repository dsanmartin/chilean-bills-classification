import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

SEED = 666

def rescale(X):
    return X / 255.0

def dataSplit(X, y, test_size=0.33, path='./'):
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=test_size, random_state=SEED, stratify=y_tr)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    y_val = np_utils.to_categorical(y_val)
    
    np.savez_compressed(path + 'test', X=X_test, y=y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

