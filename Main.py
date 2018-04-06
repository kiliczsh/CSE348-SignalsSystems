
from PIL import Image
from numpy import array
import cv2
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.cm as cm
import numpy as np
import operator


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector





def calculate(image,kernel):

    # değerler
    image_size = np.array(image).shape
    image_sutun = np.array(image).shape[0]
    image_satir = np.array(image).shape[1]
    kernel_size = np.array(kernel).shape
    kernel_sutun = np.array(kernel).shape[0]
    kernel_satir = np.array(kernel).shape[1]
    padVal = max(kernel_satir,kernel_sutun)
    print(kernel_size)
    print(image_size)
    image_pad = image
    image_pad = np.arange(2)
    image_pad = x.reshape((image_sutun,image_satir))
    result = np.pad(image_pad, padVal, pad_with,padder=0)
    img_pad_sutun =  np.array(result).shape[0]
    img_pad_satir =  np.array(result).shape[1]
    #print(image)
    print(result)
    print(kernel)

    curSatir = kernel_satir
    curSutun = kernel_sutun
    newConvSatir = 0
    newConvSutun = -1
    convImg = np.zeros((image_sutun+kernel_sutun,image_satir+kernel_satir))


    for i in range(img_pad_sutun):
        curSatir = kernel_satir
        newConvSutun += 1
        for j in range(img_pad_satir):
            if ((i+kernel_sutun == curSutun) and (j+kernel_satir == curSatir)):
                sum = 0
                for k in range(kernel_sutun):
                    for l in range(kernel_satir):
                        sum = sum + (result[kernel_sutun+k][kernel_satir+l]*kernel[k][l])
                print(sum)
                convImg[newConvSutun][newConvSatir] = sum
                newConvSatir += 1
            curSatir += 1
        curSutun += 1


    print(convImg)




                    



    return


def reverseMatrix( someArray ):
    numPyArray = np.array(someArray)
    reversed_numPyArr = np.fliplr(np.flipud(numPyArray))
    return reversed_numPyArr

#reverseMatrix([[1, 2, 3, 4, 5, 6],[7, 8, 9, 10, 11, 12]])
#print(reverseMatrix([[1, 2, 3, 4, 5, 6],[7, 8, 9, 10, 11, 12]]))
x = np.array([[1, 2, 3, 4, 5, 6],[7, 8, 9, 10, 11, 12],[11,9,7,5,3,1]])
y = np.array([[9, 8],[4, 5]])
calculate(x,y)



#pl.imshow(im, origin='lower')
#pl.show()
"""
# ayar
np.set_printoptions(threshold = np.nan)
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
#pl.show() 
# image info and print array 
#print(img_grey)
#print(np.array(img_grey))
sutun = np.array(img_grey).shape[0]
satir = np.array(img_grey).shape[1]

#img_grey_reverse = reverseMatrix(img_grey)
"""
