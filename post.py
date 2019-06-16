import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
#%%
#history = np.array([1.9217076301574707, 1.6614644527435303, 1.658648058772087, 1.5037682801485062, 1.4885833710432053, 1.5394011437892914, 1.516641989350319, 1.4970482289791107, 1.4865447729825974, 1.4755478352308273]) 
#val_loss = np.array([9.288053512573242, 2.060764789581299, 2.106947422027588, 1.7529735565185547, 1.5461255311965942, 1.5523161888122559, 1.4646443128585815, 1.4526764154434204, 1.4493248462677002, 1.4392611980438232])
#plt.plot(history)
#plt.show()
#%%
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
model = load_model('my_model.h5')
#%%
pred = model.predict(X_test)
#%%
pred_ = np.argmax(pred, axis=1, out=None)
y_ = np.argmax(y_test, axis=1, out=None)

#%%
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels

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
    return ax
#%%
#cm = confusion_matrix(y_, pred_)
print(accuracy_score(y_, pred_))
print(f1_score(y_, pred_, average='macro'))
#plt.imshow(cm)
#plt.show()
plot_confusion_matrix(y_, pred_, np.arange(5))