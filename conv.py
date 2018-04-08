import cv2
import numpy as np
from PIL import Image, ImageFilter
from numpy import array
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.cm as cm
import operator

def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[(-size):size+1, (-size_y):size_y+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size_y)))
    return g / g.sum()


img = Image.open('7.png')
img_grey = img.convert('L')  
image = np.array(img_grey)

#image = cv2.imread('octocat.jpg', cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
# A kernel of all ones is called a box blur and is simply averaging all neighbors (sum all, optionally divide by count).
#kernel = (np.array([[1, 1, 1],
#                    [1, 1, 1],
#                    [1, 1, 1]]))

kernel = (np.array([[1, 2, 1],
                    [0, 0, 0],
                   [1, 2, 1]]))
#kernel = gaussian_kernel(5)

# the weighed pixels have to be in range 0..1, so we divide by the sum of all kernel
# values afterwards
kernel_sum = kernel.sum()

# fetch the dimensions for iteration over the pixels and weights
i_width, i_height = image.shape[0], image.shape[1]
k_width, k_height = kernel.shape[0], kernel.shape[1]

# prepare the output array
filtered = np.zeros_like(image)

# Iterate over each (x, y) pixel in the image ...
for y in range(i_height):
    for x in range(i_width):
        weighted_pixel_sum = 0
        for ky in range(int(-(k_height / 2)), k_height - 1):
            for kx in range(int(-(k_width / 2)), k_width - 1):
                pixel = 0
                pixel_y = y - ky
                pixel_x = x - kx
                # boundary check: all values outside the image are treated as zero.
                # This is a definition and implementation dependent, it's not a property of the convolution itself.
                if (pixel_y >= 0) and (pixel_y < i_height) and (pixel_x >= 0) and (pixel_x < i_width):
                    pixel = image[pixel_x, pixel_y]
                # get the weight at the current kernel position
                # (also un-shift the kernel coordinates into the valid range for the array.)
                weight = kernel[int(ky ), int(kx)]
                # weigh the pixel value and sum
                weighted_pixel_sum += pixel * weight
        # finally, the pixel at location (x,y) is the sum of the weighed neighborhood
        filtered[x,y] = weighted_pixel_sum / kernel_sum
cv2.imshow('DIY convolution', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()