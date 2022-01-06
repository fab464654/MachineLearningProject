

import numpy as np
import cv2


#function to extract features from an image (pixels or histograms)
def extractFeatures(image, size, typeOfFeature, bins):
    if typeOfFeature == "histograms_bw":
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #calculate frequency of pixels in range 0-255 (given the greyscale image)
        histogram_bw = cv2.calcHist(images=[image_bw], channels=[0], mask=None, histSize=[bins], ranges=[0, 256])

        return histogram_bw.flatten()

    elif typeOfFeature == "histograms_RGB":
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #convert the image from BGR color space (default from cv2) to RGB

        chans = cv2.split(image_RGB)
        histogram_RGB = cv2.calcHist(images=[chans[0]], channels=[0], mask=None, histSize=[bins], ranges=[0, 256])
        histogram_RGB = np.append(histogram_RGB, cv2.calcHist(images=[chans[1]], channels=[0], mask=None, histSize=[bins], ranges=[0, 256]))
        histogram_RGB = np.append(histogram_RGB, cv2.calcHist(images=[chans[2]], channels=[0], mask=None, histSize=[bins], ranges=[0, 256]))

        return histogram_RGB.flatten()

    elif typeOfFeature == "histograms_HSV":
        image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert the image from BGR color space (default from cv2) to HSV

        chans = cv2.split(image_HSV)
        histogram_HSV = cv2.calcHist(images=[chans[0]], channels=[0], mask=None, histSize=[bins], ranges=[0, 256])
        histogram_HSV = np.append(histogram_HSV, cv2.calcHist(images=[chans[1]], channels=[0], mask=None, histSize=[bins], ranges=[0, 256]))
        histogram_HSV = np.append(histogram_HSV, cv2.calcHist(images=[chans[2]], channels=[0], mask=None, histSize=[bins], ranges=[0, 256]))

        return histogram_HSV.flatten()

    elif typeOfFeature == "rawPixels_RGB":
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #convert the image from BGR color space (default from cv2) to RGB
        return cv2.resize(image_RGB, size).flatten()


    elif typeOfFeature == "rawPixels_bw":
        #return the flattened raw pixels vector, after resizing the image
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(image_bw, size).flatten()

    elif typeOfFeature == "meanPixels":
        resizedImage = cv2.resize(image, size)
        featureVector = np.empty(resizedImage.shape[0]*resizedImage.shape[1])
        for i in range(resizedImage.shape[0]):
            for j in range(resizedImage.shape[1]):
                featureVector[i*resizedImage.shape[0] + j] = np.mean(resizedImage[i, j, :])
        return featureVector
