# import the necessary packages
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries
from skimage.future import graph
from skimage.util import img_as_float
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2126, 0.7152, 0.0722])
        
if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    args = vars(ap.parse_args())

    # load the image and convert it to a floating point data type
    originalImage = cv2.imread(args["image"])
    
    image = img_as_float(originalImage)
    #
    segments = 1 + slic(originalImage,  n_segments=100)
    print("Slic number of segments: %d" % len(np.unique(segments)))
    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (25))
    ax = fig.add_subplot(1, 1, 1)
    #

    regions = regionprops(segments, intensity_image=rgb2gray(image))
    intensities = []
    for props in regions:
        intensities.append(props.mean_intensity)

    sortList = sorted(intensities, key=float, reverse=True)

    media = (sortList[0] + sortList[1] + sortList[2]) / 3
    minrList = []
    mincList = []
    maxrList = []
    maxcList = []
    for props in regions:
        if props.mean_intensity >= media:
            cy, cx = props.centroid
            minr, minc, maxr, maxc = props.bbox
            minrList.append(minr)
            mincList.append(minc)
            maxrList.append(maxr)
            maxcList.append(maxc)
    #
    minrList = sorted(minrList, key=float)
    mincList = sorted(mincList, key=float)
    maxrList = sorted(maxrList, key=float, reverse=True)
    maxcList = sorted(maxcList, key=float, reverse=True)
    #
    rect = plt.Rectangle((mincList[0], minrList[0]), maxcList[0] - mincList[0], maxrList[0] - minrList[0], fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    #
    ax.imshow(mark_boundaries(originalImage, segments))
    plt.axis("off")
    # show the plots
    plt.show()
    #
    # loop over the unique segment values

    
   