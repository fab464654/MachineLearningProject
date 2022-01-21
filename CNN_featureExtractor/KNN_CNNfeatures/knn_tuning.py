
#import required libraries
import numpy as np
from scipy.spatial.distance import cdist
from scipy import stats
import pandas as pd

def knn_tuning(trainFeatures, testFeatures, trainLabels, testLabels, k_values, distance_metrics):

    #Apply the KNN algorithm for each K / distance metric / image resolution combination
    bestAccuracy = 0
    K_best, distance_best = None, None
    allAccuracies = []

    for k_value in k_values:
        fixedK_accuracy = []

        for distance_metric in distance_metrics:

            print("\n[KNN tuning] Considering " + distance_metric + " distance metric and K=" + str(k_value) + "...", end=" ")

            #Calculate the distance between all train objects and test objects
            dist = cdist(trainFeatures, testFeatures, metric=distance_metric)

            #For each test data, order the distances from the smallest to the largest and find the train indices of the closest ones
            neighbors = np.argsort(dist, axis=0)  #sorting distances in ascending order

            k_neighbors = neighbors[:k_value, :]  #keeping the first K values

            #Check the labels of these K points and find the most frequent one
            neighbors_labels = trainLabels[k_neighbors]  #get labels of the neighbour points
            prediction = stats.mode(neighbors_labels, axis=0)[0]  #mode = most common element in a given set

            #Compute and print he accuracy with the current settings
            accuracy = np.sum(prediction == testLabels) / len(testLabels)
            print('  | Achieved accuracy: ' + "{0:.2f}".format(accuracy * 100) + '% |', end="")

            #Save the parameters that got the highest accuracy
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                K_best, distance_best = k_value, distance_metric

            #Save the accuracies (depending on the 4 distances) of K = ...
            fixedK_accuracy.append(accuracy)

        #Add append all of them into a list of lists
        allAccuracies.append(fixedK_accuracy)
    cols = [1, 3, 5, 7]
    df = pd.DataFrame(allAccuracies, columns=cols)
    df.to_csv('csvLogs/K_accuracy.csv', index=False)

    return K_best, distance_best

