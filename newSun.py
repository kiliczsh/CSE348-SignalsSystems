from PIL import Image
from numpy import array
import cv2
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.cm as cm
import numpy as np
import operator

def calculate(image,kernel,paddingArgument):
    image_height,image_width = np.array(image).shape
    kernel_height,kernel_width = np.array(kernel).shape
    table = np.zeros(((image_height+kernel_height-1),(image_width+kernel_width-1)))
    table_height,table_width = np.array(table).shape
    img_height_iter = -1
    img_width_iter = 0
    # padding
    for index_height in range(table_height):
        img_height_iter += 1
        for index_width in range(table_width):
            bound_height = (index_height > (image_height - (kernel_height - 2 )))
            bound_width = (index_width > (image_width - (kernel_width - 2)))
            if (bound_height and bound_width):
                table[index_height][index_width] = image[index_height-1][index_width-1]
                img_width_iter += 1
            else:
                if(paddingArgument == 0): # padding with zero
                    table[index_height][index_width] = 0
                if(paddingArgument == 1): # padding with borders
                    inHeight = index_height <= kernel_height -1
                    inWidth = index_width <= kernel_width -1
                    if(( inHeight ) and (inWidth) ) :
                        table[index_height][index_width] = image[0][0]
                    if(not inHeight)and inWidth:
                        table[index_height][index_width] = image[index_height-kernel_height+1][0]
                    if(not inWidth)and inHeight:
                        table[index_height][index_width] = image[0][index_width-kernel_width+1]
                    if(not inHeight) and (not inWidth):
                        table[index_height][index_width] = image[index_height-kernel_height+1][index_width-kernel_width+1]
    print(table)
    img_height_iter = -1
    img_width_iter = 0
    # convolution
    for index_height in range(table_height):
        for index_width in range(table_width):
            img_height_iter += 1
            sumOfCell = 0
            if( (index_height >= kernel_height-1 ) and (index_width >= kernel_width-1) ):
                for index_kernel_height in range(kernel_height):
                    for index_kernel_width in range(kernel_width):
                        production = table[index_height-kernel_height+1+index_kernel_height][index_width-kernel_width+1+index_kernel_width]*kernel[index_kernel_height][index_kernel_width]
                        sumOfCell= sumOfCell + production 
            image[index_height-kernel_height+1][index_width-kernel_width+1] = sumOfCell
    print(image)
    return



x = np.array([[1, 2, 3, 4, 5, 6],[7, 8, 9, 10, 11, 12],[13,14,15,16,17,18]])
y = np.array([[1, -2,-3],[3,-3, 4]])
calculate(x,y,1)