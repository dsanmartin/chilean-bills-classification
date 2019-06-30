import cv2
import numpy as np
import matplotlib.pyplot as plt
#%% OPTIONS
# Data directory
DIR_BASE = "data/input/bills/"

# Define canvas size 
WIDTH = 150
HEIGHT = 150

CHANNEL = 1

# CV color option
READ_COLOR = 1
TRAN_COLOR = cv2.COLOR_BGR2RGB # cv2.COLOR_BGR2GRAY

MAX_SHAPES = 3
#%% Data Agumetation Functions
def add_lines(image):
    imshape = image.shape
    slant_extreme = 10
    slant= np.random.randint(-slant_extreme,slant_extreme) 
    drop_length = 20
    drop_width = 2
    drop_color = (200,200,200) 
    line_drops = random_lines(imshape,slant,drop_length)
    for line_drop in line_drops:
        cv2.line(image,(line_drop[0],line_drop[1]),(line_drop[0]+slant,line_drop[1]+drop_length),drop_color,drop_width)
    return image

def random_lines(imshape,slant,drop_length):
    drops=[]
    for i in range(1500): 
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
            y= np.random.randint(0,imshape[0]-drop_length)
            drops.append((x,y))
    return drops

def generate_shadow_coordinates(imshape, no_of_shadows=1):
    vertices_list=[]
    for index in range(no_of_shadows):
        vertex=[]
        for dimensions in range(np.random.randint(3,15)): ## Dimensionality of the shadow polygon
            vertex.append(( imshape[1]*np.random.uniform(),imshape[0]//3+imshape[0]*np.random.uniform()))
        vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices 
        vertices_list.append(vertices)
    return vertices_list ## List of shadow vertices


def add_shadow(image,no_of_shadows=1):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    mask = np.zeros_like(image) 
    imshape = image.shape
    vertices_list= generate_shadow_coordinates(imshape, no_of_shadows) #3 getting list of shadow vertices
    for vertices in vertices_list: 
        cv2.fillPoly(mask, vertices, 255) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
    image_HLS[:,:,1][mask[:,:,0]==255] = image_HLS[:,:,1][mask[:,:,0]==255]*0.5   ## if red channel is hot, image's "Lightness" channel's brightness is lowered 
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB

def add_noise(image):  
    rand_n = image.copy()
    mean = np.random.randint(0, 1, 3)
    std = np.random.randint(0, 50, 3)
    cv2.randn(rand_n, mean, std)
    image += rand_n
    return image

def add_blur(image):
    rand_n = image.copy()
    blurImg = cv2.blur(rand_n,(10,10))
    return blurImg

def add_inv(image):
    rand_n = image.copy()
    invImg = cv2.bitwise_not(rand_n)
    return invImg

def add_light_color(image, gamma=1.0):
    color = np.random.randint(0, 200)
    invGamma = 1.0 / gamma
    image = (color - image)
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    image=cv2.LUT(image, table)
    return image

def hue_image(image):
    saturation = np.random.randint(0, 200)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = image[:, :, 2]
    v = np.where(v <= 255 + saturation, v - saturation, 255)
    image[:, :, 2] = v
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

def image_laplacian(image):
    ddepth = cv2.CV_16S
    kernel_size=np.random.randint(0, 100)
    src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(src_gray, ddepth, kernel_size)
    abs_dst = cv2.convertScaleAbs(dst)
    return abs_dst


def draw_rectangles(image):
    line_type = cv2.LINE_AA
    number = np.random.randint(0, MAX_SHAPES)
    for i in range(number):
        rand_color = np.random.randint(0, 256, 3)
        pt1 = (np.random.randint(0, WIDTH + 1), np.random.randint(0, HEIGHT + 1))
        pt2 = (np.random.randint(0, WIDTH + 1), np.random.randint(0, HEIGHT + 1))
        cv2.rectangle(image, pt1, pt2,rand_color.tolist(),-1,line_type,0)
    return image

def draw_elipses(image):
    line_type = cv2.LINE_AA
    number = np.random.randint(0, MAX_SHAPES)
    for i in range(number):
        rand_color = np.random.randint(0, 256, 3)
        pt1 = (np.random.randint(0, WIDTH + 1), np.random.randint(0, HEIGHT + 1))
        sz =  (np.random.randint(0, 150), np.random.randint(0, 150))
        angle = np.random.randint(0, 1000) * 0.180
        cv2.ellipse(image, pt1, sz, angle, angle - 100, angle + 200,
                        rand_color.tolist(), -1,line_type, 0)
    return image

def draw_circles(image):
    line_type = cv2.LINE_AA
    number = np.random.randint(0, MAX_SHAPES)
    for i in range(number):
        rand_color = np.random.randint(0, 256, 3)
        pt1 =  (np.random.randint(0, WIDTH + 1), np.random.randint(0, HEIGHT + 1))
        cv2.circle(image, pt1, np.random.randint(30, 50), rand_color.tolist(), -1,line_type, 0)
    return image
#%%
def randomSample(image, thr=.5):
    
    # Random canvas
    rand_color = np.random.randint(0, 256, 3)
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    canvas[:] = rand_color
    
    # Resize image
    res = cv2.resize(image, None,fx=.15, fy=.15, interpolation=cv2.INTER_CUBIC)
    
    # Random position
    x_offset = np.random.randint(0, 10)#res.shape[1])
    y_offset = np.random.randint(0, 10)#res.shape[0])
    
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
    new = cv2.warpAffine(new, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=rand_color.tolist())
    #new = new.astype(np.float)
    
    # Lines
    #if np.random.uniform(0, 1) <= thr: new = add_lines(new)
    
    # Rectangles
    if np.random.uniform(0, 1) <= thr: new = draw_rectangles(new)
    
    # Elipses
    if np.random.uniform(0, 1) <= thr: new = draw_elipses(new)
    
    # Circles
    if np.random.uniform(0, 1) <= thr: new = draw_circles(new)
    
    # Noise
    if np.random.uniform(0, 1) <= thr: new = add_noise(new)
	
    # Blur
    if np.random.uniform(0, 1) <= thr: new = add_blur(new)
		
	# Inversion
    if np.random.uniform(0, 1) <= thr: new = add_inv(new)
		
	# Light
    if np.random.uniform(0, 1) <= thr: new = add_light_color(new)
		
	# Hue
    #if np.random.uniform(0, 1) <= thr: new = hue_image(new)
		
	# Laplacian
    #if np.random.uniform(0, 1) <= thr: new = image_laplacian(new)
    
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
#for i in range(10):
#    rs = randomSample(a20k, .5)
#    print(rs.shape)
#    plt.imshow(rs)
#    plt.show()

#%%
def createDataset(data_anv, data_rev, prop_anv, threshold, bills_per_class, height, width, grayscale=False):
    N = sum(bills_per_class)
    if grayscale:
        X = np.zeros((N, height, width, 1))
    else:
        X = np.zeros((N, height, width, 3), np.uint8)
    y = np.zeros(N, dtype=int)
    n = 0
    for i in range(len(bills_per_class)):
        n_anv = int(prop_anv[i] * bills_per_class[i])
        n_rev = (bills_per_class[i] - n_anv)
        for a in range(n_anv):
            if grayscale: 
                tmp = randomSample(data_anv[i], threshold[i])
                X[n] = (tmp[:,:,0] * 0.299 + tmp[:,:,1] * 0.587 + tmp[:,:,2] * 0.114).reshape(height, width, 1) #np.dot(X[n,:3], [0.299, 0.587, 0.114])
            else:
                X[n] = randomSample(data_anv[i], threshold[i])
                plt.imshow(X[n])
            y[n] = i
            n += 1
        for r in range(n_rev):
            if grayscale: 
                tmp = randomSample(data_rev[i], threshold[i]) 
                X[n] = (tmp[:,:,0] * 0.299 + tmp[:,:,1] * 0.587 + tmp[:,:,2] * 0.114).reshape(height, width, 1)
            else:
                X[n] = randomSample(data_rev[i], threshold[i])
                plt.imshow(X[n])
            y[n] = i
            n += 1
    return X, y
#%% TEST
#pa = [.5, .3, .4, .3, .1] # Proporcion Anversos
#th = [0] * 5 # Umbrales
#bc = [10] * 5 # Numero de billetes por clase
#data_anv = [a1k, a2k, a5k, a10k, a20k]
#data_rev = [r1k, r2k, r5k, r10k, r20k]
#X, y = createDataset(data_anv, data_rev, pa, th, bc, HEIGHT, WIDTH, True)
