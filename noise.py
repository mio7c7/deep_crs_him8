# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:23:13 2019

@author: RMC
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from pylab import imread,subplot,imshow,show
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import normalize

img = cv2.imread('b7.png',cv2.IMREAD_ANYDEPTH)

#img = np.load("D:/deep_crs_him8-master/b7_30.npy")[100]


#plt.imshow(b7,cmap='gray')
#plt.imsave("b7.png", b7)
"""
img = cv2.imread('b7.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.imread('b7.png', cv2.IMREAD_GRAYSCALE)
plt.imsave("gray.png", gray)"""


b7 = np.load("D:/deep_crs_him8-master/b7_30.npy")[7]
maxb = np.max(b7)
b7 = b7 / maxb
b7 = b7*255
b7 = b7.astype(np.uint8)
print(b7)
dst = cv2.fastNlMeansDenoising(b7,None,9,13)

plt.subplot(121),plt.imshow(b7)
plt.subplot(122),plt.imshow(dst)
plt.show()
kernel = np.ones((3,4),np.uint8)
#denoised = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
erosion = cv2.erode(dst,kernel,iterations = 1)
plt.imshow(erosion)
#img = cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB)
img = erosion
original_shape = img.shape

flat_image = img.reshape(-1, 1)
bandwidth2 = estimate_bandwidth(flat_image,
                                quantile=.2, n_samples=500)
print(bandwidth2)
ms = MeanShift(bandwidth2, bin_seeding=True, cluster_all=False)
ms.fit(flat_image)
labels = ms.labels_
segmented_image = np.reshape(labels, original_shape[:2])


plt.imshow(segmented_image)


