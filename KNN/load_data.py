

from my_feature_extraction import extractFeatures
import cv2
import numpy as np
import os
import random
# ------------------TRAIN IMAGES------------------------------------------

def load_train(trainPath, imageSize, pixelType, histogramType):
    # initialize the raw pixel intensities matrix, the features matrix,
    # and labels list
    trainImages = []      #raw pixels features
    trainHistograms = []  #histograms features
    trainLabels = []      #given labels

    print("Reading training images...")
    # loop over the input images
    for (i, imagePath) in enumerate(trainPath):
        image = cv2.imread(imagePath)            #read the image
        label = imagePath.split(os.path.sep)[1]  #extract label from path("class"/0.jpg)

        #extract raw / mean pixel intensity features
        pixels = extractFeatures(image=image, size=(imageSize, imageSize), typeOfFeature=pixelType, bins=256)

        #extract histograms feature (color distribution throughout the image)
        histograms = extractFeatures(image=image, size=(imageSize, imageSize), typeOfFeature=histogramType, bins=256)

        trainImages.append(pixels)          #append images
        trainHistograms.append(histograms)  #append features
        trainLabels.append(label)           #append labels

        #show an update every 1000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(trainPath)))
    print("Finished reading training images.")


    trainImages = np.array(trainImages)
    trainHistograms = np.array(trainHistograms)
    trainLabels = np.array(trainLabels)

    return trainImages, trainHistograms, trainLabels


def load_test(testPath, imageSize, pixelType, histogramType):
    # ------------------TEST IMAGES------------------------------------------
    testImages = []
    testHistograms = []
    testLabels = []
    print("Reading test images...")

    for (i, imagePath) in enumerate(testPath):
        image = cv2.imread(imagePath)  # read the image
        label = imagePath.split(os.path.sep)[1]  # extract label from path("class"/0.jpg)

        # extract raw / mean pixel intensity features
        pixels = extractFeatures(image=image, size=(imageSize, imageSize), typeOfFeature=pixelType, bins=256)

        # extract histograms feature (color distribution throughout the image)
        histograms = extractFeatures(image=image, size=(imageSize, imageSize), typeOfFeature=histogramType, bins=256)

        testImages.append(pixels)  # append images
        testHistograms.append(histograms)  # append features
        testLabels.append(label)  # append labels

        # show an update every 1000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(testPath)))
    print("Finished test training images.")

    testImages = np.array(testImages)
    testHistograms = np.array(testHistograms)
    testLabels = np.array(testLabels)

    return testImages, testHistograms, testLabels

#Reduce the number of samples (train and test) have a faster tuning of the KNN
def get_random_subset(data, labels, numSamples):
    randomSamples = random.sample(range(data.shape[0]), numSamples)
    dataSubset = data[randomSamples]
    labelsSubset = labels[randomSamples]

    return dataSubset, labelsSubset

