#import required libraries
from sklearn.neighbors import KNeighborsClassifier
from imutils import paths
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, \
    ConfusionMatrixDisplay
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from CNN.constants import natureClassNames
from scipy.spatial.distance import cdist
from scipy import stats

#Import my developed functions
from my_pca import my_pca, sklearn_pca, my_pca_tuning
from knn_tuning import knn_tuning
from load_data import extractFeatures, get_random_subset


def knn_classfication_pipeline(csvTrainPath, csvTestPath):
    #Load train features and labels
    trainFeatures, trainLabels = extractFeatures(csvTrainPath)

    #Load test features and labels
    testFeatures, testLabels = extractFeatures(csvTestPath)

    #Call the my_pca function to reduce data dimensionality (for pixels)
    if usePCA:
        print("\nRunning Principal Component Analysis algorithm to reduce data dimensionality...")

        if perform_pca_tuning:
            print("[PCA tuning] Perfoming PCA tuning of the goal variance hyperparameter....")
            bestGoalVariance, bestAccuracy, bestN = my_pca_tuning(increment_range=[0.6, 0.95], increment=0.05,
                                             trainFeatures=trainFeatures, trainLabels=trainLabels, testFeatures=testFeatures,
                                             testLabels=testLabels, show_updates=True)
            print("[PCA tuning] The best goal variance found was " + str(np.round(bestGoalVariance, 2)) + " with an accuracy of " +
                    str(np.round(bestAccuracy, 2)) + "% (" + str(bestN), "components were considered)")

        else:  #SKIPPING THE PCA TUNING PROCESS
            #Imposing the goal variance to achieve
            bestGoalVariance = 0.7  #pixels best: 0.7 / 0.75 / 0.7

        #Train data:
        trainFeatures, N_from_training, accuracy = my_pca(trainFeatures, trainLabels, testFeatures,
                                              testLabels, training_phase=True, N=None, goal_variance=bestGoalVariance,
                                              show_plot=False, show_comparison=True)  #reducing feature space

        print("Comparing (but not using) with sklearn PCA results...")
        sklearn_pca(trainFeatures, bestGoalVariance)
        #Test data:
        testFeatures = my_pca(testFeatures, trainLabels, testFeatures, testLabels,
                            training_phase=False, N=N_from_training, goal_variance=bestGoalVariance, show_plot=False,
                            show_comparison=True)  #reducing "pixel" feature space


    #Call the KNN tuning function, to get the "best" hyperparameters
    if perform_KNN_tuning:
        k_values = [1, 3, 5, 7]
        distance_metrics = ['euclidean', 'cosine', 'jaccard', 'mahalanobis']

        print("------------------------------------------------------------")
        K_best, distance_best = knn_tuning(trainFeatures, testFeatures, trainLabels, testLabels, k_values, distance_metrics)

        print("\n[KNN tuning] The best hyperparameters found are: K=" + str(K_best) + "; metric distance=" + distance_best)
        print("------------------------------------------------------------")


    else:  #SKIPPING THE KNN TUNING PROCESS
        #Imposing the KNN parameters
        K_best = 7  # 7 / 7 / 7
        distance_best = 'euclidean'  # euclidean / euclidean / euclidean

        print("[Skip KNN tuning] Using K=" + str(K_best) + " and metric distance=" + distance_best)


    #-----------------------------------------------------------------------------#
    #Try with my implementation of KNN classifier with the "best" hyperparameters:
    #-----------------------------------------------------------------------------#

    print("\n[my KNN] Considering CNN extracted features...", end="")

    #Calculate the distance between all train objects and test objects
    dist = cdist(trainFeatures, testFeatures, metric=distance_best)

    #For each test data, order the distances from the smallest to the largest and find the train indices of the closest ones
    neighbors = np.argsort(dist, axis=0)  #sorting distances in ascending order

    k_neighbors = neighbors[:K_best, :]        #keeping the first K values

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

    #df_cm = pd.DataFrame(confusionMatrix, index=[i for i in classNames], columns=[i for i in classNames])
    if showConfusionMatrix:
        disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classNames)
        fig, ax = plt.subplots(figsize=(12, 8))
        font = {'size': 17}
        plt.rc('font', **font)
        disp.plot(ax=ax, cmap="magma")
        plt.title("Confusion matrix (CNN features): K=" + str(K_best) + "; dist=" + distance_best, fontweight='bold')
        plt.savefig(savingPath + "confusionMatrix_KNN_best.jpg", dpi=300)
        plt.close(disp.figure_)

    """
    if showConfusionMatrix:
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.title("Confusion matrix: CNN features; K=" + str(K_best) + "; dist=" + distance_best, fontweight='bold')
        plt.tight_layout()
        plt.show()
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


#Set execution options
showClassificationReport = True
showConfusionMatrix = True

#Decide whether or not using PCA
usePCA = True

#Decide whether or not performing the KNN tuning and PCA tuning
perform_KNN_tuning = True
perform_pca_tuning = True

#Run the whole KNN pipeline using CNN extracted features
csvTrainPath = "CNN_features/feature_vectors_train.csv"
csvTestPath = "CNN_features/feature_vectors_test.csv"

knn_classfication_pipeline(csvTrainPath, csvTestPath)




