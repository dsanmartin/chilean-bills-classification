import cv2
import numpy as np

# Data Augmentation Functions
def randomLines(imshape, slant, drop_length, N=1500):
    drops = []
    for i in range(N): 
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)
            y = np.random.randint(0, imshape[0] - drop_length)
            drops.append((x, y))
    return drops

def addLines(image, slant_extreme=10, drop_length=20, drop_width=2, drop_color=(200, 200, 200)):
    new_img = image.copy()
    imshape = new_img.shape
    slant = np.random.randint(-slant_extreme, slant_extreme)
    line_drops = randomLines(imshape, slant, drop_length)
    for line_drop in line_drops:
        cv2.line(
            new_img, 
            (line_drop[0], line_drop[1]), 
            (line_drop[0] + slant, line_drop[1] + drop_length),
            drop_color, drop_width
        )
    return new_img

def generateShadowCoordinates(imshape, no_of_shadows=1, poly_dim=10):
    vertices_list = []
    for index in range(no_of_shadows):
        vertex = []
        for dimensions in range(poly_dim): ## Dimensionality of the shadow polygon
            vertex.append((imshape[1] * np.random.uniform(), imshape[0] // 3 +imshape[0] * np.random.uniform()))
        vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices 
        vertices_list.append(vertices)
    return vertices_list ## List of shadow vertices

def addShadow(image, no_of_shadows=1):
    new_img = image.copy()
    if len(image.shape) == 2:
        new_img = cv2.cvtColor(new_img ,cv2.COLOR_GRAY2RGB)
    image_HLS = cv2.cvtColor(new_img, cv2.COLOR_RGB2HLS) ## Conversion to HLS
    mask = np.zeros_like(new_img) 
    imshape = image.shape
    vertices_list = generateShadowCoordinates(imshape, no_of_shadows, np.random.randint(3, 30)) #3 getting list of shadow vertices
    for vertices in vertices_list: 
        cv2.fillPoly(mask, vertices, 255) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
    image_HLS[:,:,1][mask[:,:,0]==255] = image_HLS[:,:,1][mask[:,:,0]==255]*0.5   ## if red channel is hot, image's "Lightness" channel's brightness is lowered 
    new_img = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB) ## Conversion to RGB
    if len(image.shape) == 2:
        new_img = cv2.cvtColor(image_HLS, cv2.COLOR_RGB2GRAY)
    return new_img

def addNoise(image, max_mean=100, max_std=100):  
    new_img = image.copy()
    rand_n = image.copy()
    chs = 3 if len(image.shape) == 3 else 1
    mean = np.random.randint(0, max_mean, chs)
    std = np.random.randint(0, max_std, chs)
    cv2.randn(rand_n, mean, std)
    new_img += rand_n
    return new_img

def addBlur(image, kind='avg', **kwargs):
    img_cp = image.copy()
    
    if kind == 'avg': # Average
        k_size = (10, 10)
        if 'k_size' in kwargs: 
            k_size = kwargs['k_size']
        new_img = cv2.blur(img_cp, k_size)
        
    elif kind == 'gauss': # Gaussian
        k_size, sigmaX, sigmaY = (5, 5), 0, 0 # k_size odd numbers
        if 'k_size' in kwargs: k_size = kwargs['k_size']
        if 'sigmaX' in kwargs: sigmaX = kwargs['sigmaX']
        if 'sigmaY' in kwargs: sigmaY = kwargs['sigmaY']
        new_img = cv2.GaussianBlur(img_cp, k_size, sigmaX, sigmaY)
        
    elif kind == 'med': # Median
        k_size = 1
        if 'k_size' in kwargs and kwargs['k_size'] in [1, 3, 4, 5]: k_size = kwargs['k_size'] # Only 1, 3, 4, 5
        new_img = cv2.medianBlur(img_cp, k_size)
        
    elif kind == 'bi': # Bilateral
        d, color, space = 5, 75, 75
        if 'd' in kwargs: d = kwargs['d']
        if 'color' in kwargs: color = kwargs['color']
        if 'space' in kwargs: space = kwargs['space']
        new_img = cv2.bilateralFilter(img_cp, d, color, space)
        
    return new_img

def addInv(image):
    img_cp = image.copy()
    new_img = cv2.bitwise_not(img_cp)
    return new_img

def addLightColor(image, gamma=1.0, color=100):
    img_cp = image.copy()
    inv_gamma = 1.0 / gamma
    img_cp = (color - img_cp)
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    new_img = cv2.LUT(img_cp, table)
    return new_img

def hueImage(image, saturation=100):
    new_img = image.copy()
    if len(image.shape) == 2:
        new_img = cv2.cvtColor(new_img ,cv2.COLOR_GRAY2RGB)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2HSV)
    v = new_img[:, :, 2]
    v = np.where(v <= 255 + saturation, v - saturation, 255)
    new_img[:, :, 2] = v
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)
    if len(image.shape) == 2:
        new_img = cv2.cvtColor(new_img ,cv2.COLOR_RGB2GRAY)
    return new_img

def drawRectangles(image, shapes=3):
    new_img = image.copy()
    shape = image.shape
    line_type = cv2.LINE_AA
    chs = 3 if len(image.shape) == 3 else 1
    for i in range(shapes):
        rand_color = np.random.randint(0, 256, chs)
        pt1 = (np.random.randint(0, shape[1] + 1), np.random.randint(0, shape[0] + 1))
        pt2 = (np.random.randint(0, shape[1] + 1), np.random.randint(0, shape[0] + 1))
        cv2.rectangle(new_img, pt1, pt2,rand_color.tolist(),-1,line_type,0)
    return new_img

def drawEllipses(image, shapes=3):
    new_img = image.copy()
    line_type = cv2.LINE_AA
    chs = 3 if len(image.shape) == 3 else 1
    for i in range(shapes):
        rand_color = np.random.randint(0, 256, chs)
        ct = (np.random.randint(0, new_img.shape[1] + 1), np.random.randint(0, new_img.shape[0] + 1))
        ax =  (np.random.randint(0, new_img.shape[1] // 2 + 1), np.random.randint(0, new_img.shape[0] // 2 + 1))
        angle = np.random.randint(0, 361)
        st_ang = np.random.randint(0, 361)
        en_ang = np.random.randint(0, 361)
        cv2.ellipse(new_img, ct, ax, angle, st_ang, en_ang,
                        rand_color.tolist(), -1,line_type, 0)
    return new_img

def drawCircles(image, shapes=3, max_rad=50):
    new_img = image.copy()
    line_type = cv2.LINE_AA
    chs = 3 if len(image.shape) == 3 else 1
    for i in range(shapes):
        rand_color = np.random.randint(0, 256, chs)
        pt1 = (np.random.randint(0, new_img.shape[1] + 1), np.random.randint(0, new_img.shape[0] + 1))
        rad = np.random.randint(1, max_rad)
        cv2.circle(new_img, pt1, rad, rand_color.tolist(), -1, line_type, 0)
    return new_img

def imageLaplacian(image, k_size=100):
    new_img = image.copy()
    ddepth = cv2.CV_8U
    #kernel_size = np.random.randint(0, max_k)
    #new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.Laplacian(new_img, ddepth, k_size)
    #new_img = cv2.convertScaleAbs(new_img)
    return new_img

def addCanvas(image, height, width):
    new = image.copy()
    rand_color = np.random.randint(0, 256, 3)
    shape = (height, width, 3)
    
    if len(image.shape) == 2: 
        shape = shape[:2]
        rand_color = np.array([np.sum(rand_color * np.array([0.2989, 0.5870, 0.1140]))])
        
    canvas = np.zeros(shape, np.uint8)
    canvas[:] = rand_color
    
    # Resize image
    res = cv2.resize(new, None, fx=.15, fy=.15, interpolation=cv2.INTER_CUBIC)
    
    # Random position
    x_offset = np.random.randint(0, 10)
    y_offset = np.random.randint(0, 10)
    
    new = np.copy(canvas)
    
    new[y_offset:y_offset+res.shape[0], x_offset:x_offset+res.shape[1]] = res
    
    # Image size
    (h, w) = new.shape[:2]
    center = (w // 2, h // 2) # Center
    
    # Random angle and scale
    angle = np.random.randint(0, 360)
    scale = np.random.uniform(low=0.25, high=2.0)
    
    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # New random sample
    new_img = cv2.warpAffine(new, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=rand_color.tolist())
    
    return new_img

def randomSample(image, height, width, max_fig=1, max_tran=1, thr=.1):
    new = image.copy()
    
    figs = [drawRectangles, drawEllipses, drawCircles]
    figs_c = np.random.choice(figs, max_fig, replace=False)
    
    for fig in figs_c:
        if np.random.uniform(0, 1) <= thr: new = fig(new)
    
    # Add Canvas
    new = addCanvas(new, height, width)
    
    trans = [addLines, addNoise, addBlur, addInv, addLightColor, hueImage, imageLaplacian]
    trans_c = np.random.choice(trans, max_tran, replace=False)
    
    for tran in trans_c:
        if np.random.uniform(0, 1) <= thr: new = tran(new)
    
    return new 

def createDataset(data_anv, data_rev, prop_anv, threshold, bills_per_class, height, width, color):
    N = sum(bills_per_class)
    if color: X = np.zeros((N, height, width, 3), np.uint8)  
    else: X = np.zeros((N, height, width))
    y = np.zeros(N, dtype=int)
    n = 0
    for i in range(len(bills_per_class)):
        n_anv = int(prop_anv[i] * bills_per_class[i])
        n_rev = (bills_per_class[i] - n_anv)
        for a in range(n_anv):
            X[n] = randomSample(data_anv[i], height, width, thr=threshold[i])
            y[n] = i
            n += 1
        for r in range(n_rev):
            X[n] = randomSample(data_rev[i], height, width, thr=threshold[i])
            y[n] = i
            n += 1
    if not color: X = X.reshape(N, height, width, 1) # For Keras Convolution
    return X, y

def saveDataset(X, y, path):
    np.savez_compressed(path + 'data', X=X, y=y)
    #np.save(path + 'X', X)
    #np.save(path + 'y', y)
    
def loadBills(bills, path, ext, color):
    data_anv = list()
    data_rev = list()
    
    TRAN_COLOR = cv2.COLOR_BGR2RGB if color else cv2.COLOR_BGR2GRAY
    
    # Load bills pictures
    for bill in bills:
        path_anv = path + bill + '/anverso' + ext 
        path_rev = path + bill + '/reverso' + ext 
        load_anv = cv2.cvtColor(cv2.imread(path_anv, 1), TRAN_COLOR)
        load_rev = cv2.cvtColor(cv2.imread(path_rev, 1), TRAN_COLOR)
        data_anv.append(load_anv)
        data_rev.append(load_rev)
    
    return data_anv, data_rev
    