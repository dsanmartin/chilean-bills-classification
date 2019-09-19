import argparse
import sys
import numpy as np
from datetime import datetime
from alexnet import Alexnet
from preprocessing import rescale, dataSplit

if (sys.version_info > (3, 0)):
    import pathlib
else:
    import pathlib2 as pathlib

# Folders ##
DIR_BASE = "data/"
DIR_OUT = DIR_BASE + "output/experiments/"

# Default SGD parameters #
LR = 0.1
MOM = 0.0
DEC = 0.0
NES = False

# Fit default parameters #
EPOCHS = 3#100
BATCH = 1#0
VERBOSE = 1

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', help="Input directory.", type=str, required=True) 
    args = parser.parse_args()
        
    input_dir = args.dir
    
    # Experiment ID
    folder = datetime.today().strftime('%Y%m%d%H%M%S')
    
    # Output folder
    output_dir = DIR_OUT + folder + '/' 
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    data = np.load(input_dir + 'data.npz')
    X = data['X']
    y = data['y']
    
    Xr = rescale(X)
    
    # Model
    model = Alexnet(X[0].shape, output_dir)
    model.compile(LR, MOM, DEC, NES)
    
    X_train, y_train, X_val, y_val, X_test, y_test = dataSplit(Xr, y, path=output_dir)
    
    hist = model.fit(X_train, y_train, X_val, y_val, EPOCHS, BATCH, VERBOSE)
    
    y_pred = model.predict(X_test)
    
    eva = model.evaluate(X_test, y_test)
    
    model.save()
    
    model.saveParameters({ 'id': folder, 'lr': LR, 'mom': MOM, 'dec': DEC, 'nes': NES, 'epochs': EPOCHS, 'batch': BATCH })
    
    print(folder)

if __name__ == "__main__":
    main()
    