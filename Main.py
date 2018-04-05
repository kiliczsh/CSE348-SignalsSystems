
from PIL import Image
from numpy import array
import numpy
import cv2
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.cm as cm
import numpy as np


#pl.imshow(im, origin='lower')
#pl.show()

# ayar
numpy.set_printoptions(threshold=numpy.nan)
# image 
img = Image.open('octocat.jpg')

# convert the image to *greyscale*
img_grey = img.convert('L')       

# convert to numpy array
img_array = np.array(img_grey)

# black white image kaydet
Image.fromarray(img_array).save("octocat_bw.png")
# show image
pl.imshow(img_array, cmap=cm.Greys_r)
pl.show() 
# image info and print array 
print(img_grey)
print(np.array(img_grey))