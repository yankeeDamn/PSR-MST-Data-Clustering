# -*- coding: utf-8 -*-
"""
@author:
    
Saglam, Ali, & Baykan, N. A. (2017).

"Sequential image segmentation based on min- imum spanning tree representation".

Pattern Recognition Letters, 87 , 155–162.

https://doi.org/10.1016/j.patrec.2016.06.001 .
"""

from sequential_segmentation import sequential_segmentation
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb

img = cv.imread("37073.jpg")

img = cv.GaussianBlur(img,(5,5),1.5)

###################################################

labels = sequential_segmentation(img, m = 3, l = "scale")

###################################################

colored_segments = label2rgb(labels)
colored_segments = np.uint8(colored_segments * 255)

plt.imshow(colored_segments)

  