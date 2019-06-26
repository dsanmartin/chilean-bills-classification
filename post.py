import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import json
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
#%%
def plotLoss(loss, val_loss):
    plt.plot(np.arange(len(loss)), loss, 'b-*', label='Loss')
    plt.plot(np.arange(len(val_loss)), val_loss, 'r-x', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    return None

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show() 
    return ax

#%% Show loss function
    
#Console variable
import sys
print (sys.argv [1])

with open('data/output/models/' + sys.argv [1] + '/history.json') as json_file:
    data = json.load(json_file)
    loss = np.array(data['loss'])
    val_loss = np.array(data['val_loss'])
plotLoss(loss, val_loss)
#%%
X_test = np.load('data/output/models/' + sys.argv [1] + '/X_test.npy')
y_test = np.load('data/output/models/'+ sys.argv [1] + '/y_test.npy')
model = load_model('data/output/models/' + sys.argv [1] + '/model.h5')
#%%
pred = model.predict(X_test)
#%%
y_pred = np.argmax(pred, axis=1, out=None)
y_real = np.argmax(y_test, axis=1, out=None)
#%%
print(accuracy_score(y_real, y_pred))
print(f1_score(y_real, y_pred, average='macro'))
plot_confusion_matrix(y_real, y_pred, np.arange(5))