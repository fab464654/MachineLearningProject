

import numpy as np
import random
import pandas as pd

# ------------------TRAIN IMAGES' features------------------------------------------

def extractFeatures(csvPath):
    # extract CNN features from the .csv file

    print("Getting images' features (extracted with a ResNet18)...")

    df = pd.read_csv(csvPath)
    labels = df.iloc[:, 0].to_numpy()     #retrieve the first column (label)
    features = df.iloc[:, 1:].to_numpy()  #retrieve all rows minus the label

    print("Finished reading the previously saved features (shape = " + str(features.shape) + ").\n")

    return features, labels


#Reduce the number of samples (train and test) have a faster tuning of the KNN
def get_random_subset(data, labels, numSamples):
    randomSamples = random.sample(range(data.shape[0]), numSamples)
    dataSubset = data[randomSamples]
    labelsSubset = labels[randomSamples]

    return dataSubset, labelsSubset

