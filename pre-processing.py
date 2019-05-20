import cv2
import numpy as np
import matplotlib.pyplot as plt
#%% OPTIONS
# Data directory
DIR_BASE = "data/"

# Define canvas size 
WIDTH = 500
HEIGHT = 500

# CV color option
READ_COLOR = 1
TRAN_COLOR = cv2.COLOR_BGR2RGB # cv2.COLOR_BGR2GRAY
#%%
def randomSample(image):
    
    # Random canvas
    rand_color = tuple(np.random.randint(0, 256, 3))
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    canvas[:] = rand_color
    
    # Resize image
    res = cv2.resize(image, None,fx=.5, fy=.5, interpolation=cv2.INTER_CUBIC)
    
    # Random position
    x_offset = np.random.randint(0, 100)
    y_offset = np.random.randint(0, 100)
    
    new = np.copy(canvas)
    new[y_offset:y_offset+res.shape[0], x_offset:x_offset+res.shape[1]] = res
    
    # Image size
    (h, w) = new.shape[:2]
    center = (w // 2, h // 2) # Center
    
    # Random angle and scale
    angle = np.random.randint(0, 360)
    scale = np.random.uniform(low=0.5, high=2.0)
    
    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # New random sample
    new = cv2.warpAffine(new, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=rand_color)
    #new = new.astype(np.float)
    
    """
    poner alguna figura random...
    rand_color_2 = tuple(np.random.randint(0, 256, 3))
    rand_pos = (np.random.randint(0, HEIGHT + 1), np.random.randint(0, WIDTH + 1))
    rand_rad = np.random.randint(0, 100)
    cv2.circle(new, rand_pos, rand_rad, rand_color_2, -1)
    """
    
    # Noise
    rand_n = new.copy()
    mean = np.random.randint(0, 1, 3)
    std = np.random.randint(0, 50, 3)
    cv2.randn(rand_n, tuple(mean), tuple(std))
    new += rand_n
    
    return new 
    
#%% LOAD Banknotes
a1k = cv2.cvtColor(cv2.imread(DIR_BASE + "1000/anverso.jpg", READ_COLOR), TRAN_COLOR)
a2k = cv2.cvtColor(cv2.imread(DIR_BASE + "2000/anverso.jpg", READ_COLOR), TRAN_COLOR)
a5k = cv2.cvtColor(cv2.imread(DIR_BASE + "5000/anverso.jpg", READ_COLOR), TRAN_COLOR)
a10k = cv2.cvtColor(cv2.imread(DIR_BASE + "10000/anverso.jpg", READ_COLOR), TRAN_COLOR)
a20k = cv2.cvtColor(cv2.imread(DIR_BASE + "20000/anverso.jpg", READ_COLOR), TRAN_COLOR)

r1k = cv2.cvtColor(cv2.imread(DIR_BASE + "1000/reverso.jpg", READ_COLOR), TRAN_COLOR)
r2k = cv2.cvtColor(cv2.imread(DIR_BASE + "2000/reverso.jpg", READ_COLOR), TRAN_COLOR)
r5k = cv2.cvtColor(cv2.imread(DIR_BASE + "5000/reverso.jpg", READ_COLOR), TRAN_COLOR)
r10k = cv2.cvtColor(cv2.imread(DIR_BASE + "10000/reverso.jpg", READ_COLOR), TRAN_COLOR)
r20k = cv2.cvtColor(cv2.imread(DIR_BASE + "20000/reverso.jpg", READ_COLOR), TRAN_COLOR)
#%%
for i in range(10):
    rs = randomSample(a10k)
    plt.imshow(rs)
    plt.show()

