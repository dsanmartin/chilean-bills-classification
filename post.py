#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
#from keras.models import load_model
from sklearn.metrics import confusion_matrix #accuracy_score, f1_score,
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels

def plotAcc(acc, val_acc, path=None):
    fig = plt.figure()
    plt.plot(np.arange(len(acc)), acc, 'b-*', label='Acc')
    plt.plot(np.arange(len(val_acc)), val_acc, 'r-x', label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    if path is None: 
        plt.show()
    else: 
        plt.savefig(path + 'acc_plot.png')
        plt.close(fig)
    return None

def plotLoss(loss, val_loss, path=None):
    fig = plt.figure()
    plt.plot(np.arange(len(loss)), loss, 'b-*', label='Loss')
    plt.plot(np.arange(len(val_loss)), val_loss, 'r-x', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    if path is None: 
        plt.show()
    else: 
        plt.savefig(path + 'loss_plot.png')
        plt.close(fig)
    return None

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          path=None):
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
    
    if path is not None:
        #np.save(path + 'cm.npy', cm)
        np.savetxt(path + 'cm.txt', cm, delimiter=',', fmt='%d')
    print(cm)
    
    fig = plt.figure()
    #fig, ax = plt.subplots()
    ax = fig.add_subplot(1,1,1)
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
    #fig.tight_layout()
    if path is None: 
        plt.show()
    else: 
        plt.savefig(path + 'confusion_matrix.png')
        plt.close(fig)
    return None

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', help="Input directory.", type=str, required=True) 
    args = parser.parse_args()
        
    input_dir = args.dir
    
    print("Loading history...")
    
    # Show loss function
    with open(input_dir + 'history.json') as json_file:
        data = json.load(json_file)
        loss = np.array(data['loss'])
        val_loss = np.array(data['val_loss'])
        acc = np.array(data['acc'])
        val_acc = np.array(data['val_acc'])
        
    plotLoss(loss, val_loss, input_dir)
    plotAcc(acc, val_acc, input_dir)
    
    
    print("Loading testing data...")
    # Load test data
    test = np.load(input_dir + 'test.npz')
    #X_test = test['X']
    y_test = test['y']
    
    # Load model
    #model = load_model(input_dir  + 'model.h5')
    
    print("Loading predition...")
    # Prediction
    #pred = model.predict(X_test)
    pred = np.load(input_dir + 'prediction.npz')
    
    # Softmax output to categorical
    y_pred = np.argmax(pred['y_pred'], axis=1, out=None)
    y_real = np.argmax(y_test, axis=1, out=None)
    
    print("Evaluation")
    # Show results
    #print('Accuracy score: %f' % accuracy_score(y_real, y_pred))
    #print('F1 score: %f' % f1_score(y_real, y_pred, average='macro'))
    names_ = ['1000', '2000', '5000', '10000', '20000']
    
    report = classification_report(y_real, y_pred, target_names=names_, digits=6, output_dict=True)
    json.dump(report, open(input_dir + 'class_report.json', 'w'))
    print(classification_report(y_real, y_pred, target_names=names_, digits=6))
    plot_confusion_matrix(y_real, y_pred, np.arange(5), path=input_dir)

if __name__ == "__main__":
    main()