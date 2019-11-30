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
    preImage = cv2.imread(args["image"])
    #aplicar o negativo
    preImage = cv2.bitwise_not(preImage)
    cv2.imshow('negaaaa', preImage)
    #aplicando contraste
    alpha = 2.2 # Simple contrast control
    beta = 50   # Simple brightness control
    new_image = np.zeros(preImage.shape, preImage.dtype)
    for y in range(preImage.shape[0]):
        for x in range(preImage.shape[1]):
            for c in range(preImage.shape[2]):
                new_image[y,x,c] = np.clip(alpha*preImage[y,x,c] + beta, 0, 255)

    cv2.imshow('aaaaaa', new_image)
    #aplicando thresholding
    a,preImage = cv2.threshold(new_image,150, 255, cv2.THRESH_BINARY)

    image = img_as_float(preImage)

    # loop over the number of segments
    for numSegments in (100, 200, 300):
        # apply SLIC and extract (approximately) the supplied number
        # of segments
        segments = slic(image, n_segments = numSegments, sigma = 5)

        # show the output of SLIC
        fig = plt.figure("Superpixels -- %d segments" % (numSegments))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments))
        plt.axis("off")

    # show the plots
    plt.show()