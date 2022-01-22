import pandas
from my_SVM import *
import os
import sys

# Function to extract from the CSV file the image feature vectors extracted from the CNN (ResNet18)
# and they relative labels to specify to which class they belong to.
# This function returns:
# - "featureVectors": numpy array collecting in rows the feature vectors
# - "labels": a list of strings tha specify the class of the i-th vector in featureVectors
def readFeatureExtractedFromCNN(csvPath):
    df = pandas.read_csv(csvPath, delimiter=',', header=None)
    data = pandas.DataFrame.to_numpy(df)
    labels = data[:,0].astype(int).astype(str)
    dictionary = {"0": "buildings", "1": "forest", "2": "glaciers", "3": "mountain", "4": "sea", "5": "street"}
    for i in range(0,labels.size):
        labels[i] = dictionary[labels[i]]
    featureVectors = data[:,1:]
    return featureVectors, labels
# END ===========================================================================================

# Function to change the stdout such that it prints both in terminal
# and in a log file
def printBothInTerminalAndLogFile(fileID):
    # To write both in terminal and file
    class Unbuffered:
        def __init__(self, stream):
            self.stream = stream

        def write(self, data):
            self.stream.write(data)
            self.stream.flush()
            fileID.write(data)  # Write the data of stdout here to a text file as well

        def flush(self):
            pass

    sys.stdout = Unbuffered(sys.stdout)
# END ===============================================================================

# ===================================================================================
def main():
    usePCA = True
    # To print both in terminal but also in a log file
    if not os.path.exists("./images"):
        os.makedirs("./images")
    logFile = open("./images/terminal_output.txt", "w+")
    printBothInTerminalAndLogFile(logFile)
    # ------------------------------------------------

    # Extract from the CSV file the image feature vectors extracted from the CNN (ResNet18)
    # and they relative labels to specify to which class they belong to
    trainFeatVecs, trainLabels = readFeatureExtractedFromCNN("feature_vectors_train.csv") # del train
    testFeatVecs, testLabels = readFeatureExtractedFromCNN("feature_vectors_test.csv") # del test

    # PCA:
    if usePCA:
        bestVariance = 0.9 # found in previous tuning
        trainFeatVecs, N_from_training, accuracy = my_pca(trainFeatVecs, trainLabels, testFeatVecs,
                                                          testLabels, training_phase=True, N=None,
                                                          goal_variance=bestVariance,
                                                          show_plot=False, show_comparison=True)
        testFeatVecs = my_pca(testFeatVecs, trainLabels, testFeatVecs, testLabels,
                              training_phase=False, N=N_from_training, goal_variance=bestVariance, show_plot=False,
                              show_comparison=True)

    # Execute SVM with these feature vectors extracted from the ResNet18 and evaluate their performances
    my_SVC(trainFeatVecs, trainLabels, testFeatVecs, testLabels, "Confusion matrix (CNN features)", showClassificationReport=True)

    logFile.close()
    sys.stdout = sys.__stdout__
    print("\n\n")
    print(">>> Execution terminated! <<<")
# END main() ===============================================================================

if __name__ == "__main__":
    main()