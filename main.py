import argparse
import sys
import numpy as np
from datetime import datetime
from alexnet import Alexnet, dataPre

if (sys.version_info > (3, 0)):
    import pathlib
else:
    import pathlib2 as pathlib

DIR_BASE = "data/"
DIR_OUT = DIR_BASE + "output/experiments/"

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
    X = np.load(input_dir + 'X.npy')
    y = np.load(input_dir + 'y.npy')
    
    # Model
    model = Alexnet(X[0].shape, output_dir)
    model.compile()
    
    X_train, y_train, X_val, y_val = dataPre(X, y, path=output_dir)
    
    model.fit(X_train, y_train, X_val, y_val)
    
    print(folder)

if __name__ == "__main__":
    main()
    

