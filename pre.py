import sys
import argparse
from datetime import datetime
from dataset import loadBills, createDataset, saveDataset

if (sys.version_info > (3, 0)):
    import pathlib
else:
    import pathlib2 as pathlib

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
    th = [.1] * 5 # Threshold
    bc = [5] * 5 # Number of bills per class
    X, y = createDataset(data_anv, data_rev, pa, th, bc, args.height, args.width, args.color)
    
    # Save
    saveDataset(X, y, output)

    print(folder)
    
    info = open(output + 'info.txt', 'w')
    info.write("ID: {0}\n".format(folder))
    info.write("Width: {0}\n".format(args.width))
    info.write("Heigth: {0}\n".format(args.height))
    info.write("Color: {0}\n".format(args.color))
    info.write("Bills per class: {0}\n".format(", ".join([str(i) for i in bc])))
    info.write("Heads %: {0}\n".format(", ".join([str(i) for i in pa])))
    info.write("Thresholds %: {0}".format(", ".join([str(i) for i in th])))
    info.close()
    

if __name__ == "__main__":
    main()