import argparse
import pathlib2 as pathlib
from datetime import datetime
from preprocessing import loadBills, createDataset, saveDataset

# Data directory
DIR_BASE = "data/input/"
BILLS_DIR = DIR_BASE + "bills/"
ARRAY_DIR = DIR_BASE + "arrays/"

EXT = '.jpg'

# Define canvas size 
WIDTH = 150
HEIGHT = 150

# CV color option
COLOR = 0

BILLS = ['1000', '2000', '5000', '10000', '20000']

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', '-w', help="Image width.", type=int, default=WIDTH)
    parser.add_argument('--height', '-t', help="Image height.", type= int, default=HEIGHT)
    parser.add_argument('--color', '-c', help="Image color. 0: grayscale, 1: color.", type= int, default=COLOR)    
    args = parser.parse_args()
    
    folder = datetime.today().strftime('%Y%m%d%H%M%S')
    output = ARRAY_DIR + folder + '/'
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    
    # Load bills
    data_anv, data_rev = loadBills(BILLS, BILLS_DIR, EXT, args.color)
    
    # Create dataset
    pa = [.5] * 5 # Number of heads
    th = [.0] * 5 # Threshold
    bc = [5] * 5 # Number of bills per class
    X, y = createDataset(data_anv, data_rev, pa, th, bc, args.height, args.width, args.color)
    
    # Save
    saveDataset(X, y, output)

    print(folder)
    

if __name__ == "__main__":
    main()
    
##%%
#import numpy as np
#import matplotlib.pyplot as plt
#
#X = np.load('data/input/arrays/20190918011746/X.npy')
#
#print(X.shape)
#plt.imshow(X[0,:,:,0])
