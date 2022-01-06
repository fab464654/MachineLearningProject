#import required libraries
from sklearn.neighbors import KNeighborsClassifier
from imutils import paths
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from CNN.constants import natureClassNames
from scipy.spatial.distance import cdist
from scipy import stats

#Import my developed functions
from my_pca import my_pca, sklearn_pca, my_pca_tuning
from knn_tuning import knn_tuning
from load_data import load_train, load_test, get_random_subset


def knn_classfication_pipeline(histogramType, pixelType):
    #Load train features and labels
    trainImages, trainHistograms, trainLabels = load_train(trainPath, imageSize, pixelType, histogramType)

    #Load test features and labels
    testImages, testHistograms, testLabels = load_test(testPath, imageSize, pixelType, histogramType)

    #Call the my_pca function to reduce data dimensionality (for pixels)
    if usePCA:
        print("\nRunning Principal Component Analysis algorithm to reduce data dimensionality (on pixel features)...")

        if perform_pca_tuning:
            print("[PCA tuning] Perfoming PCA tuning of the goal variance hyperparameter....")
            bestGoalVariance, bestAccuracy, bestN = my_pca_tuning(increment_range=[0.6, 0.95], increment=0.05, imageSize=imageSize,
                                             trainImages=trainImages, trainLabels=trainLabels, testImages=testImages,
                                             testLabels=testLabels, show_updates=True, featureType=pixelType)
            print("[PCA tuning] The best goal variance found was " + str(np.round(bestGoalVariance, 2)) + " with an accuracy of " +
                    str(np.round(bestAccuracy, 2)) + "% (" + str(bestN), "components were considered)")





        else:  #SKIPPING THE PCA TUNING PROCESS
            #Imposing the goal variance to achieve
            bestGoalVariance = 0.7  #pixels best: 0.7 / 0.75 / 0.7

        #Train data:
        trainImages, N_from_training, accuracy = my_pca(trainImages, imageSize, imageSize, trainLabels, testImages,
                                              testLabels, training_phase=True, N=None, goal_variance=bestGoalVariance,
                                              show_plot=False, show_comparison=True)  #reducing "pixel" feature space



        print("Comparing (but not using) with sklearn PCA results...")
        sklearn_pca(trainImages, bestGoalVariance)
        #Test data:
        testImages = my_pca(testImages, imageSize, imageSize, trainLabels, testImages, testLabels,
                            training_phase=False, N=N_from_training, goal_variance=bestGoalVariance, show_plot=False,
                            show_comparison=True)  #reducing "pixel" feature space



    #Call the my_pca function to reduce data dimensionality (for histograms)
    if usePCA:
        print("\nRunning Principal Component Analysis algorithm to reduce data dimensionality (on histogram features)...")

        if perform_pca_tuning:
            print("[PCA tuning] Perfoming PCA tuning of the goal variance hyperparameter....")
            bestGoalVariance, bestAccuracy, bestN = my_pca_tuning(increment_range=[0.6, 0.95], increment=0.05, imageSize=imageSize,
                                             trainImages=trainHistograms, trainLabels=trainLabels, testImages=testHistograms,
                                             testLabels=testLabels, show_updates=True, featureType=histogramType)
            print("[PCA tuning] The best goal variance found was " + str(np.round(bestGoalVariance, 2)) + " with an accuracy of " +
                  str(np.round(bestAccuracy, 2)) + "% (" + str(bestN), "components were considered)")

        else:  #SKIPPING THE PCA TUNING PROCESS
            #Imposing the goal variance to achieve
            bestGoalVariance = 0.9  #histograms best: 0.9 / 0.8 / 0.9

        #Train data:
        trainHistograms, N_from_training, accuracy = my_pca(trainHistograms, imageSize, imageSize, trainLabels,
                                                            test_data=testHistograms, test_labels=testLabels, training_phase=True, N=None,
                                                            goal_variance=bestGoalVariance, show_plot=False, show_comparison=True)  #reducing "histograms" feature space

        print("Comparing (but not using) with sklearn PCA results...")
        sklearn_pca(trainHistograms, bestGoalVariance)

        #Test data:
        testHistograms = my_pca(testHistograms, imageSize, imageSize, train_labels=None, test_data=None, test_labels=None,
                                training_phase=False, N=N_from_training, goal_variance=bestGoalVariance, show_plot=False,
                                show_comparison=True)  #reducing "histograms" feature space

    #Call the KNN tuning function, to get the "best" hyperparameters
    if perform_KNN_tuning:
        k_values = [1, 3, 5, 7]
        distance_metrics = ['euclidean', 'cosine', 'jaccard', 'mahalanobis']

        #Reduce the number of samples (train and test) have a faster tuning of the KNN
        numSamples_train = 10000
        numSamples_test = 3000
        #numSamples_train = 600
        #numSamples_test = 300

        trainImagesSubset, trainLabelsSubset = get_random_subset(trainImages, trainLabels, numSamples=numSamples_train)
        testImagesSubset, testLabelsSubset   = get_random_subset(testImages, testLabels, numSamples=numSamples_test)

        trainHistSubset, trainHistLabelsSubset = get_random_subset(trainHistograms, trainLabels, numSamples=numSamples_train)
        testHistSubset, testHistLabelsSubset = get_random_subset(testHistograms, testLabels, numSamples=numSamples_test)

        print("\n------------------------------------------------------------")
        print("[KNN pixels tuning] Considering " + str(numSamples_train) + " samples for training and " + str(numSamples_test) + " for testing", end="")
        K_best_pixels, distance_best_pixels = knn_tuning(trainImagesSubset, testImagesSubset, trainLabelsSubset, testLabelsSubset, k_values, distance_metrics, pixelType)

        print("\n[KNN pixels tuning] The best hyperparameters found are: K=" + str(K_best_pixels) + "; metric distance=" + distance_best_pixels)
        print("------------------------------------------------------------")


        print("\n------------------------------------------------------------")
        print("[KNN histograms tuning] Considering " + str(numSamples_train) + " samples for training and " + str(numSamples_test) + " for testing", end="")
        K_best_hist, distance_best_hist = knn_tuning(trainHistSubset, testHistSubset, trainHistLabelsSubset, testHistLabelsSubset, k_values, distance_metrics, histogramType)

        print("\n[KNN histograms tuning] The best hyperparameters found are: K=" + str(K_best_hist) + "; metric distance=" + distance_best_hist)
        print("------------------------------------------------------------")

    else:  #SKIPPING THE KNN TUNING PROCESS
        #Imposing the KNN parameters
        K_best_pixels = 7  # 7 / 7 / 7
        distance_best_pixels = 'euclidean'  # euclidean / euclidean / euclidean

        K_best_hist = 7  #3 / 5 / 7
        distance_best_hist = 'cityblock'  #cityblock / euclidean / cityblock

        print("\n[Skip KNN pixels tuning] Skipping the KNN tuning process...")
        print("[Skip KNN pixels tuning] Using K=" + str(K_best_pixels) + " and metric distance=" + distance_best_pixels)
        print("[Skip KNN histograms tuning] Skipping the KNN tuning process...")
        print("[Skip KNN histograms tuning] Using K=" + str(K_best_hist) + " and metric distance=" + distance_best_hist)


    #-----------------------------------------------------------------------------#
    #Try with my implementation of KNN classifier with the "best" hyperparameters:
    #-----------------------------------------------------------------------------#

    print("\n[my KNN] Considering " + pixelType + " pixels features...", end="")

    #Calculate the distance between all train objects and test objects
    dist = cdist(trainImages, testImages, metric=distance_best_pixels)

    #For each test data, order the distances from the smallest to the largest and find the train indices of the closest ones
    neighbors = np.argsort(dist, axis=0)  #sorting distances in ascending order

    k_neighbors = neighbors[:K_best_pixels, :]        #keeping the first K values

    #Check the labels of these K points and find the most frequent one
    neighbors_labels = trainLabels[k_neighbors]           #get labels of the neighbour points
    prediction = stats.mode(neighbors_labels, axis=0)[0]  #mode = most common element in a given set

    #Compute and print the accuracy
    accuracy = np.sum(prediction == testLabels) / len(testLabels)
    print('   | Achieved accuracy: ' + "{0:.2f}".format(accuracy * 100) + '% |')

    #Compute and save the confusion matrix
    if showClassificationReport:
        print("[my KNN] Classification Report:\n", classification_report(testLabels, np.transpose(prediction)))
        precision, recall, fscore, support = precision_recall_fscore_support(testLabels, np.transpose(prediction), average="macro")
        print("precision = {:.3f}\nrecall = {:.3f}\nfscore = {:.3f}\nsupport = {}\n".format(precision, recall, fscore, support))

    confusionMatrix = confusion_matrix(testLabels, np.transpose(prediction))
    print(testLabels.shape, prediction.shape)

    df_cm = pd.DataFrame(confusionMatrix, index=[i for i in classNames], columns=[i for i in classNames])

    if showConfusionMatrix:
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.title("Confusion matrix: " + pixelType + "; K=" + str(K_best_pixels) + "; dist=" + distance_best_pixels, fontweight='bold')
        plt.tight_layout()
        plt.savefig(savingPath + "confusionMatrix_" + pixelType + "_KNN_best.jpg", dpi=300)
        plt.show()

    #-------------------------------------------------------------------------#
    print("[my KNN] Considering " + histogramType + " as features...", end="")

    #Calculate the distance between all train objects and test objects
    dist = cdist(trainHistograms, testHistograms, metric=distance_best_hist)

    #For each test data, order the distances from the smallest to the largest and find the train indices of the closest ones
    neighbors = np.argsort(dist, axis=0)  #sorting distances in ascending order
    k_neighbors = neighbors[:K_best_hist, :]        #keeping the first K values

    #Check the labels of these K points and find the most frequent one
    neighbors_labels = trainLabels[k_neighbors]           #get labels of the neighbour points
    prediction = stats.mode(neighbors_labels, axis=0)[0]  #mode = most common element in a given set

    #Compute and print the accuracy
    accuracy = np.sum(prediction == testLabels) / len(testLabels)
    print('   | Achieved accuracy: ' + "{0:.2f}".format(accuracy * 100) + '% |')

    #Compute and save the confusion matrix
    if showClassificationReport:
        print("[my KNN] Classification Report:\n", classification_report(testLabels, np.transpose(prediction)))
        precision, recall, fscore, support = precision_recall_fscore_support(testLabels, np.transpose(prediction),
                                                                             average="macro")
        print("precision = {:.3f}\nrecall = {:.3f}\nfscore = {:.3f}\nsupport = {}\n".format(precision, recall, fscore,
                                                                                            support))

    confusionMatrix = confusion_matrix(testLabels, np.transpose(prediction))
    df_cm = pd.DataFrame(confusionMatrix, index=[i for i in classNames], columns=[i for i in classNames])

    if showConfusionMatrix:
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.title("Confusion matrix: " + histogramType + "; K=" + str(K_best_hist) + "; dist=" + distance_best_hist, fontweight='bold')
        plt.tight_layout()
        plt.savefig(savingPath + "confusionMatrix_" + histogramType + "_KNN_best.jpg", dpi=300)
        plt.show()

    """
    #------------- Using the sklearn code to compare results ------------------#
    numNeighbors = K_best_pixels  #use the same number of neighbors to have a fair comparison

    #Train and evaluate a K-NN classifier on the raw pixel intensities
    print("\n[sklearn] Evaluating " + pixelType + " accuracy...   ", end="")
    #model = KNeighborsClassifier(n_neighbors=K_best_pixels, metric=distance_best_pixels, n_jobs=1)
    model = KNeighborsClassifier(n_neighbors=K_best_pixels, metric=distance_best_pixels, n_jobs=1)
    model.fit(trainImages, trainLabels)
    acc = model.score(testImages, testLabels)
    print("| Achieved accuracy: {:.2f}% |".format(acc * 100))

    #Test the model and compute the confusion matrix
    predictedLabels = model.predict(testImages)

    if showClassificationReport:
        print("Classification Report:\n", classification_report(testLabels, predictedLabels))

    confusionMatrix = confusion_matrix(testLabels, predictedLabels)
    df_cm = pd.DataFrame(confusionMatrix, index=[i for i in classNames], columns=[i for i in classNames])

    if showConfusionMatrix:
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.tight_layout()
        plt.savefig(savingPath + "confusionMatrix_" + pixelType + "_sklearn.jpg", dpi=300)


    #Train and evaluate a k-NN classifer on the histogram representations
    print("[sklearn] Evaluating " + histogramType + " accuracy...   ", end="")
    model = KNeighborsClassifier(n_neighbors=K_best_hist, n_jobs=1, metric=distance_best_hist)
    model.fit(trainHistograms, trainLabels)
    acc = model.score(testHistograms, testLabels)
    print("| Achieved accuracy: {:.2f}% |".format(acc * 100))

    #Test the model and compute the confusion matrix
    predictedLabels = model.predict(testHistograms)
    if showClassificationReport:
        print("Classification Report:\n", classification_report(testLabels, predictedLabels))
    confusionMatrix = confusion_matrix(testLabels, predictedLabels)
    df_cm = pd.DataFrame(confusionMatrix, index=[i for i in classNames], columns=[i for i in classNames])

    if showConfusionMatrix:
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.tight_layout()
        plt.savefig(savingPath + "confusionMatrix_" + histogramType + "_sklearn.jpg", dpi=300)
    """
#-------------------- end of knn_classfication_pipeline -----------------------------------------------------------#


#Set dataset parameters
trainPath = list(paths.list_images("../dataset/seg_train"))
testPath = list(paths.list_images("../dataset/seg_test"))
#trainPath = list(paths.list_images("../dataset/NatureDatasetReduced/train"))  #to try with less images
#testPath = list(paths.list_images("../dataset/NatureDatasetReduced/test"))    #to try with less images
savingPath = "images/"
numClasses = 6
classNames = natureClassNames
imageSize = 32

#Set execution options
showClassificationReport = True
showConfusionMatrix = True

#Decide whether or not using PCA
usePCA = True

#Decide whether or not performing the KNN tuning and PCA tuning
perform_KNN_tuning = True
perform_pca_tuning = True

#Set the histogram feature extraction method
histogramType = "histograms_RGB"  #histograms_bw / histograms_RGB / histograms_HSV
pixelType = "rawPixels_RGB"  #rawPixels_bw / rawPixels_RGB / meanPixels
knn_classfication_pipeline(histogramType, pixelType)

histogramType = "histograms_bw"
pixelType = "rawPixels_bw"
knn_classfication_pipeline(histogramType, pixelType)

histogramType = "histograms_HSV"
pixelType = "meanPixels"
knn_classfication_pipeline(histogramType, pixelType)




