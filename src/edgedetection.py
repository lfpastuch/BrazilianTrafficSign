import matplotlib.pyplot as plt
import os
import numpy
import cv2 as cv
import numpy as np

def contrast_enhance(img):
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    L, a, b = cv.split(img_lab)
    L = cv.equalizeHist(L)
    img_lab_merge = cv.merge((L, a, b))
    return cv.cvtColor(img_lab_merge, cv.COLOR_Lab2BGR)

def auto_canny(img, method, sigma=0.33):
    """
    Args:
    img: grayscale image
    method: Otsu, triangle, and median
    sigma: 0.33 (default)
    2 outputs:
    edge_detection output, the high threshold for Hough Transform"""
    if method=="median":
        Th = np.median(img)
        
    elif method=="triangle":
        Th, _ = cv.threshold(img, 0, 255, cv.THRESH_TRIANGLE)
        
    elif method=="otsu":
        Th, _ = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
        
    else:
        raise Exception("method specified not available!")
        
    lowTh = (1-sigma) * Th
    highTh = (1+sigma) * Th
    
    return cv.Canny(img, lowTh, highTh)

img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset/train/images/')) + "/"

files = np.random.choice(os.listdir(img_dir), size=10, replace=False)

plt.figure(figsize=(150, 120))
for i, img_file in enumerate(files):
    img = cv.imread(img_dir + img_file)
    
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    ce = contrast_enhance(img_rgb)
    
    ac = auto_canny(ce, "median")
    
    plt.subplot(5, 2, i+1)
    plt.imshow(ac)
    
plt.show()


