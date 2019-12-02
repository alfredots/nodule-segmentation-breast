# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

def aumentarContraste(img):
        newImage = np.zeros((img.shape[0],img.shape[1]), np.uint8)
        rows, columns= img.shape

        for i in range(rows):
            for j in range(columns):
                resultado = 15.9687*np.sqrt(img[i,j])
                if resultado > 255:
                    resultado = 255
                newImage[i,j] = resultado
        return newImage
        
if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    args = vars(ap.parse_args())

    # load the image and convert it to a floating point data type
    originalImage = cv2.imread(args["image"])
    #aplicar o negativo
    preImage = cv2.bitwise_not(originalImage)
    #aplicando contraste
    alpha = 2.2 # Simple contrast control
    beta = 50   # Simple brightness control
    new_image = np.zeros(preImage.shape, preImage.dtype)
    for y in range(preImage.shape[0]):
        for x in range(preImage.shape[1]):
            for c in range(preImage.shape[2]):
                new_image[y,x,c] = np.clip(alpha*preImage[y,x,c] + beta, 0, 255)

    #aplicando thresholding
    a,preImage = cv2.threshold(new_image,150, 255, cv2.THRESH_BINARY)

    image = img_as_float(preImage)

    # loop over the number of segments
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    segments = slic(image, n_segments = 200)

    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (25))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.axis("off")
    # show the plots
    plt.show()
    fig.savefig('test.png')    
    
    #
    img = cv2.imread('test.png',0)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    circles = cv2.HoughCircles(dilation,cv2.HOUGH_GRADIENT,1,10,
                                param1=50,param2=12,minRadius=0,maxRadius=20)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('detected circles',cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
   