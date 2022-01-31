import cv2 as cv
import os
import numpy as np

##############                         ##############
############## Step1: Data preparation ##############
##############                         ##############

#setting path for image getting and storing
path_to_dataset = "l2-ip-images/test/corrupted"
path_to_output = "l2-ip-images/test/results"

#take filenames out from the folder and store it to a list, this is a list of filename strings
filenames = [img for img in os.listdir(path_to_dataset)]
filenames.sort() 

# read the images out from path and store them into a list, this is a list of image numpy data
img_list = []
for img in filenames:
    i= cv.imread(os.path.join(path_to_dataset, img),cv.IMREAD_GRAYSCALE)
    img_list.append(i)


##############                          ##############
############## Step 2: Images Dewarping ##############
##############                          ##############

# set dewarping as the first task to do in case the black border affects image processing

# found the four cornor pixels of the graph by diagonally reverse from four edges line by line till find the first non-zero pixel.
def findUpLeftPixel(matrix):
    row = len(matrix)
    col = len(matrix[0])
    for j in range(col + row - 1):
        i = 0
        while ((j-1) >= 0) and ((i+1) <= (row - 1)):
            i = i + 1
            j = j - 1        
            if i >= row or j >= col:
                # skip out-of-bounds indexes
                continue
            item = matrix[i][j]
            if item != 0:
                return i,j
    return None
 

def findUpRightPixel(matrix):
    row = len(matrix)
    col = len(matrix[0])
    for j in range(col- 1, -1, -1):
        i = 0
        while ((j+1) <= (col - 1)) and ((i+1) <= (row - 1)):
            i = i + 1
            j = j + 1      
            item = matrix[i][j]
            if item != 0:
                return i,j
    return None


def findBottomLeftPixel(matrix):
    row = len(matrix)
    col = len(matrix[0])
    for j in range(col + row - 1):
        i = row - 1
        while ((j-1) >= 0) and ((i-1) >= 0):
            i = i - 1
            j = j - 1      
            item = matrix[i][j]
            if item != 0:
                return i,j
    return None

def findBottomRightPixel(matrix):
    row = len(matrix)
    col = len(matrix[0])
    for j in range(col - 1,-1,-1):
        i = row - 1
        while ((j+1) <= col - 1) and ((i-1) >= 0):
            i = i - 1
            j = j + 1      
            item = matrix[i][j]
            if item != 0:
                return i,j
    return None


def findCornerPixels(matrix):
    ulr, ulc = findUpLeftPixel(matrix) #ulr = up left row, ulc = up left column
    urr, urc = findUpRightPixel(matrix)
    blr, blc = findBottomLeftPixel(matrix)
    brr, brc = findBottomRightPixel(matrix)
    return np.float32([[ulc,ulr],[blc,blr],[brc,brr],[urc,urr]])

def makePerspectiveMatrix(img):
    rows,cols = img.shape
    old = findCornerPixels(img)
    new = np.float32([[0,0],[0,rows - 1],[cols - 1,rows - 1],[cols - 1,0]])

    # Compute the perspective transform M
    M = cv.getPerspectiveTransform(old,new)
    return M

def dewarp(img_list):
    subject = img_list[0]
    rows,cols = subject.shape
    M = makePerspectiveMatrix(subject)
    for n in range(len(img_list)):
        i = img_list[n]
        i = cv.warpPerspective(i,M,(cols,rows))
        cv.imwrite(os.path.join(path_to_output, filenames[n]), i)

dewarp(img_list)

# # store the dewarpped images data in a list for further transformation
warpped_img_list = []
for img in filenames:
    i= cv.imread(os.path.join(path_to_output, img),cv.IMREAD_GRAYSCALE)
    warpped_img_list.append(i)


##############                        ##############
############## Step 3: Noise Removal  ##############
##############                        ##############

def removeNoise(img_list):
    for i in range(len(img_list)):
        each = img_list[i]
        # sharpening the photos first before noise removel in case it'll remove too many details
        gaussian = cv.GaussianBlur(each,(5,5),0)
        sharpen1 = cv.addWeighted(each,1.5,gaussian,-0.5,1)

        # salt and pepper noise removal using medianBlur
        median = cv.medianBlur(sharpen1, 3)

        cv.imwrite(os.path.join(path_to_output, filenames[i]), median)
        img_list[i] = median
    cv.destroyAllWindows()

removeNoise(warpped_img_list)

##############                                             ##############
############## Step 4: Brightness and Contrast Adjustment  ##############
##############                                             ##############

# prepare gama correction function to adjust contrast and brightness
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv.LUT(src, table)

def adjustContrastandBrightness(img_list):

    for i in range(len(img_list)):
        each = img_list[i]

        # adjust strong contrast and brightness using gamma correction
        gamma = gammaCorrection(each, 2.0)

        # sharpening edges to make objects more recognisable
        gaussian = cv.GaussianBlur(gamma,(5,5),0)
        sharpen2 = cv.addWeighted(gamma,1.5,gaussian,-0.5,1)
        cv.imwrite(os.path.join(path_to_output, filenames[i]), sharpen2)
    cv.destroyAllWindows()

adjustContrastandBrightness(warpped_img_list)

# ##############               ##############
# ############## Video making  ##############
# ##############               ##############

# append the modified images to a list
final_list = []
for img in filenames:
    i= cv.imread(os.path.join(path_to_output, img),cv.IMREAD_GRAYSCALE)
    final_list.append(i)

#create the video:
height, width = final_list[0].shape

size = (width, height)

print('Creating the result video!')
out_video = cv.VideoWriter('outvid.avi', cv.VideoWriter_fourcc(*'DIVX'), 3, size, isColor=False)

for i in range(len(final_list)):
    out_video.write(final_list[i])
out_video.release()




##############                                                            ##############
############## This part of code is just for images making in the report  ##############
##############                                                            ##############


# test = warpped_img_list[57]
# raw = img_list[57]

# # noise removal without sharpening first
# median0 = cv.medianBlur(test, 5) 

# # noise removal with sharpening first
# gaussian = cv.GaussianBlur(test,(5,5),0)
# sharpen1 = cv.addWeighted(test,1.5,gaussian,-0.5,1)
# median1 = cv.medianBlur(sharpen1, 5) 
# minus = median0 - median1

# res = np.vstack((median0,median1,minus))
# cv.imshow('comparison',res)
# cv.imwrite(os.path.join("l2-ip-images", 'noise.png'), res)
# cv.waitKey(0)
# cv.destroyAllWindows()


# # contrast and brightness adjustment 
# clahe = cv.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
# clahe_out = clahe.apply(median1)

# gamma = gammaCorrection(median1, 2.0)

# add = cv.addWeighted(clahe_out,0.5,gamma,0.5,1)

# #sharpening after gamma correction
# gaussian = cv.GaussianBlur(gamma,(5,5),0)
# sharpen2 = cv.addWeighted(gamma,1.5,gaussian,-0.5,1)

# res = np.vstack((median1,gamma,sharpen2))
# cv.imshow('comparison',res)
# cv.imwrite(os.path.join("l2-ip-images", 'sharpen2.png'), res)
# cv.waitKey(0)
# cv.destroyAllWindows()


# python3 code/yolo.py --video_file outvid.avi -cl code/coco.names -cf code/yolov3.cfg -w code/yolov3.weights 