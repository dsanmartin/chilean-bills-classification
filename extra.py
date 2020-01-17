import numpy as np
import matplotlib.pyplot as plt
from preprocessing import rescale

data = np.load('data/input/arrays/20190919131056/data.npz')
X = data['X']
y = data['y']
print(X.shape, y.shape)
X = rescale(X[:10])
print(np.min(X), np.max(X))

#%%
for i in range(len(X)):
    plt.imshow(X[i,:,:,0])
    plt.show()

#%%
import numpy as np

m = np.arange(25)
M = m.reshape(5, 5)
np.savetxt('teto.txt', M, delimiter=',', fmt='%d')

#%%
import json

BILLS = ['1000', '2000', '5000', '10000', '20000']

with open('class_report.json') as json_file:
    data = json.load(json_file)
    for b in BILLS:
        pre = data[b]['precision']
        rec = data[b]['recall']
        f1s = data[b]['f1-score']
        
        print("\$ $%s$ & $%.3f$ & $%.3f$ & $%.3f$ \\\\"%(b, pre, rec, f1s))
    
    micro = data['micro avg']
    macro = data['macro avg']
    weigh = data['weighted avg']
    print("& & & \\\\")
    print("Micro Average & $%.3f$ & $%.3f$ & $%.3f$ \\\\" % (micro['precision'], micro['recall'], micro['f1-score']))
    print("Macro Average & $%.3f$ & $%.3f$ &  $%.3f$ \\\\" % (macro['precision'], macro['recall'], macro['f1-score']))
    print("Weighted Average & $%.3f$ & $%.3f$ & $%.3f$ \\\\" % (weigh['precision'], weigh['recall'], weigh['f1-score']))
    