import cv2 as cv
import argparse
import math
import os
from cv2 import imshow
import numpy as np

path_to_dataset = "l2-ip-images/test/corrupted"
path_to_output = "l2-ip-images/test/results"

#take filenames out from the folder and store it to a list
filenames = [img for img in os.listdir(path_to_dataset)]
filenames.sort() 
# print(filenames)

# print(type(filenames))

#read the images out one by one and store them into a list
img_list = []
for img in filenames:
    i= cv.imread(img)
    img_list.append(i)
    # print (img)
# print(len(img_list))

# for i in img_list:
test001 = img_list[0]

test001 = cv.GaussianBlur(test001,(5,5),0)
# test001 = cv.medianBlur(test001, 5)
cv.imwrite(os.path.join(path_to_output , 'test001.png'), test001)

cv.imshow('test001.png', test001)
cv.waitKey(10000)
cv.destroyAllWindows()


