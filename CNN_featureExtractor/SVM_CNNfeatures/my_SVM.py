# SVC: Support Vector CLassification (Multiclass)
import os
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from drawCharts import *
import warnings
import sklearn.exceptions
from sklearn.exceptions import ConvergenceWarning
from my_pca import *
warnings.filterwarnings('error', category=ConvergenceWarning, module='sklearn')


def my_SVC_single_kernel(trainSet, trainLabels, testSet, testLabels, kernel_type, max_num_iter):
    # The following if statement, set a number of maximum iterations only
    # in case of 'linear' kernel (the only one that does not converge with 'NatureDataset')
    if kernel_type == 'linear':
        max_n_iter = max_num_iter
    else:
        max_n_iter = -1

    # Classifier DEFINITION (with the kernel specified in variable 'kernel_type')
    binary_clf = SVC(kernel=kernel_type, max_iter=max_n_iter)
    clf = OneVsRestClassifier(binary_clf)

    # Classifier TRAINING using 'trainSet' vectors of feature and relative labels 'trainLabels'
    # -Try catch- to handle ConvergenceWarning when 'max_iter' iterations
    #             are not enough to reach the convergence ( <=> min. tollerance)
    try:
        clf.fit(trainSet, trainLabels)
        message_end = ""
    except sklearn.exceptions.ConvergenceWarning:
        warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
        clf.fit(trainSet, trainLabels)
        message_end = " (NO CONVERGENCE, # max iterations = {})".format(max_n_iter)
        warnings.filterwarnings('error', category=ConvergenceWarning, module='sklearn')
    except Exception as e:
        raise


    predictedLabels = clf.predict(testSet)

    accuracy = accuracy_score(testLabels, predictedLabels)
    precision = precision_score(testLabels, predictedLabels, average='macro', zero_division=0)
    recall = recall_score(testLabels, predictedLabels, average='macro')
    f1score = f1_score(testLabels, predictedLabels, average='macro')
    conf_mat = confusion_matrix(testLabels, predictedLabels)

    message = "[{:^7s}]: accuracy = {:.2f}, precision = {:.2f}, recall = {:.2f}" \
        .format(kernel_type, accuracy, precision, recall) + message_end
    print(message)

    return accuracy, precision, recall, f1score, conf_mat, clf, predictedLabels
# END my_SVC_single_kernel()


def plot_SVM_histogram_kernels(barLabels, values, title, xlabel, ylabel, saveFig=False, savingPath=None, showFig=False):
    font = {'size': 16}
    fig = plt.figure(figsize=(12, 8))  # width x height
    plt.rc('font', **font)
    plt.bar(barLabels, values,
            color=[(0.533, 0.67, 0.81), (0.95, 0.627, 0.34), (0.525, 0.7, 0.498), (0.90,0.47,0.84)],
            edgecolor='grey')
    plt.grid(axis='y', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel(xlabel, fontweight='bold', fontsize=12)
    plt.ylabel(ylabel, fontweight='bold', fontsize=12)
    plt.title(title, fontweight='bold', fontsize=18)
    plt.text(23, 45, r'$\mu=15, b=3$')
    values = list(values)
    values = [round(float(i), 2) for i in values]
    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(values):
        plt.text(xlocs[i] - 0.15, v, str(v), fontsize=28)
    #plt.tight_layout()
    plt.show(block=False)

    # Save this fig
    if saveFig:
        if not os.path.exists(savingPath):
            os.makedirs(savingPath)
        fileName = title.replace(" ", "_") + "_" + ylabel
        plt.savefig(savingPath + fileName + ".jpg")

    if showFig:
        return fig
    else:
        plt.close(fig)
        return None
# END plot_SVM_histogram_kernels()


def my_SVC(trainSet, trainLabels, testSet, testLabels, featureType, max_num_iter=10000, showClassificationReport=False):
    # Possible kernels type to use
    kernels = ["rbf", "poly", "sigmoid", "linear"]
    kernels_verbose = ["RBF", "POLYNOMIAL", "SIGMOID", "LINEAR"]

    # Define dictionaries to store evaluation metrics for each kernel type
    accuracies = {}; precisions = {}; recalls = {}; f1scores = {}; conf_mats = {}; classifiers = {}; predictedLabels = {}

    print("\nMulticlass Support Vector Classification on 'NatureDataset': " + featureType)
    print("======================================================================================================")

    best_acc = 0.0
    best_kernel = ""
    # Iterate over all the possible kernels, training the Support Vector Machine
    # and retrieving evaluation metrics values
    for kernel in kernels:
        accuracies[kernel], precisions[kernel], recalls[kernel], f1scores[kernel], conf_mats[kernel], classifiers[kernel], predictedLabels[kernel] = \
            my_SVC_single_kernel(trainSet, trainLabels, testSet, testLabels, kernel, max_num_iter)
        if accuracies[kernel] > best_acc:
            best_kernel = kernel
            best_acc = accuracies[kernel]
    # END for (kernel in kernels)
    print("======================================================================================================\n")
    if showClassificationReport:
        print("\nThe best kernel was: ", best_kernel)
        print("Classification Report:\n", classification_report(testLabels, predictedLabels[best_kernel]))
    precision, recall, fscore, support = precision_recall_fscore_support(testLabels, predictedLabels[best_kernel], average="macro")
    print("precision = {:.3f}\nrecall = {:.3f}\nfscore = {:.3f}\nsupport ={}\n".format(precision, recall, fscore, support))

    folder_path = 'images/' + ("".join(featureType.split())) + "/"

    # Save accuracies, precision, recall, f1-score histogram
    plot_SVM_histogram_kernels(kernels_verbose, np.array(list(accuracies.values())), featureType, "Kernels", "ACCURACY", saveFig=True, savingPath=folder_path, showFig=True)
    plot_SVM_histogram_kernels(kernels_verbose, np.array(list(precisions.values())), featureType, "Kernels", "PRECISION", saveFig=True, savingPath=folder_path)
    plot_SVM_histogram_kernels(kernels_verbose, np.array(list(recalls.values())), featureType, "Kernels", "RECALL", saveFig=True, savingPath=folder_path)
    plot_SVM_histogram_kernels(kernels_verbose, np.array(list(f1scores.values())), featureType, "Kernels", "f1-score", saveFig=True, savingPath=folder_path)

    [fig, ax] = create_3bars_chart(kernels_verbose, dataY_1=np.array(list(accuracies.values())), dataY_2=np.array(list(precisions.values())),
                       dataY_3=np.array(list(recalls.values())), labelsY=["ACCURACIES","PRECISION","RECALL"],
                       savingName=folder_path+("".join(featureType.split()))+"_acc_prec_rec.jpg", title=featureType,
                       plotLabelX="Kernels", plotLabelY="")
    plt.close(fig)

    [fig, ax] = create_4bars_chart(kernels_verbose, dataY_1=np.array(list(accuracies.values())), dataY_2=np.array(list(precisions.values())),
                       dataY_3=np.array(list(recalls.values())), dataY_4=np.array(list(f1scores.values())),
                       labelsY=["ACCURACIES","PRECISION","RECALL","f1-score"],
                       savingName=folder_path+("".join(featureType.split()))+"_acc_prec_rec_F1SCORE.jpg",
                       title=featureType, plotLabelX="Kernels", plotLabelY="")
    plt.close(fig)

    # Save confusion matrices computed previously and stored in variable conf_mats
    for kernel in kernels:
        #f = plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mats[kernel], display_labels=classifiers[kernel].classes_)
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.rc('font', size=21)
        plt.rc('xtick', labelsize=19)
        plt.rc('ytick', labelsize=19)
        disp.plot(ax=ax, cmap="magma", xticks_rotation=45)
        plt.title(featureType+": kernel= "+str(kernel))
        #plt.show(block=False)
        fileName = ("".join(featureType.split())) + "_" + str(kernel) + "_conf_matrix.jpg"
        plt.tight_layout()
        plt.savefig(folder_path + fileName)
        plt.close(disp.figure_) # COMMENT this line if you want to display as figure all the confusion matrices

    return accuracies, precisions, recalls, f1scores, conf_mats
# END mySVM()

def reduce_dimensionality(trainImages, imageSize, trainLabels, testImages, testLabels, goalVarPixel, goalVarHist, trainHistograms, testHistograms):

    print("\nRunning Principal Component Analysis algorithm to reduce data dimensionality (on pixel features)...")
    # Train data:
    trainImages, N_from_training, accuracy = my_pca(trainImages, imageSize, imageSize, trainLabels, testImages,
                                                    testLabels, training_phase=True, N=None,
                                                    goal_variance=goalVarPixel,
                                                    show_plot=False,
                                                    show_comparison=True)  # reducing "pixel" feature space
    # Test data:
    testImages = my_pca(testImages, imageSize, imageSize, trainLabels, testImages, testLabels,
                        training_phase=False, N=N_from_training, goal_variance=goalVarPixel, show_plot=False,
                        show_comparison=True)  # reducing "pixel" feature space
    # ------------------------------------------------------------------------------------------------------------------#
    print(
        "\nRunning Principal Component Analysis algorithm to reduce data dimensionality (on histogram features)...")


    # Train data:
    trainHistograms, N_from_training, accuracy = my_pca(trainHistograms, imageSize, imageSize, trainLabels,
                                                        test_data=testHistograms, test_labels=testLabels,
                                                        training_phase=True, N=None,
                                                        goal_variance=goalVarHist, show_plot=False,
                                                        show_comparison=True)  # reducing "histograms" feature space
    # Test data:
    testHistograms = my_pca(testHistograms, imageSize, imageSize, train_labels=None, test_data=None,
                            test_labels=None,
                            training_phase=False, N=N_from_training, goal_variance=goalVarHist,
                            show_plot=False,
                            show_comparison=True)  # reducing "histograms" feature space

    return trainImages, testImages, trainHistograms, testHistograms

def testDifferentSVMclassifiers(trainSetPath, testSetPath, imageSize, usePCA, max_num_iter=10000, showClassificationReport=False):

    featureTypes = ["rawPixels_bw", "histograms_bw", "rawPixels_RGB", "histograms_RGB", "meanPixels", "histograms_HSV"]

    trainImages, trainHistograms, trainLabels = load_train(trainSetPath, imageSize, featureTypes[0], featureTypes[1])
    testImages, testHistograms, testLabels = load_test(testSetPath, imageSize, featureTypes[0], featureTypes[1])
    # Here, after the loading of the dataset, we may add here PCA, to reduce the dimensionality, and the result,
    # would be given to the next function my_SVC as the parameter "trainImages" and "testImages" and the same
    # in the next points, marked by a (*)

    #-----------------------------------------------
    # BLACK and WHITE pixels and histograms
    #-----------------------------------------------
    # Call the my_pca function to reduce data dimensionality
    if usePCA:
        goalVarPixel = 0.7  #Imposing the goal variance to achieve (PIXELS)
        goalVarHist = 0.9   #Imposing the goal variance to achieve (HISTOGRAMS)
        trainImages, testImages, trainHistograms, testHistograms = reduce_dimensionality(trainImages, imageSize, trainLabels, testImages, testLabels,
                                                                                         goalVarPixel, goalVarHist, trainHistograms, testHistograms)
    # Call SVC
    my_SVC(trainImages, trainLabels, testImages, testLabels, "BLACK and WHITE IMAGES", max_num_iter, showClassificationReport)
    my_SVC(trainHistograms, trainLabels, testHistograms, testLabels, "BLACK and WHITE HISTOGRAMS", max_num_iter, showClassificationReport)
    #------------------------------------------------------------------------------------------------------------------#

    # -----------------------------------------------
    # RGB pixels and histograms
    # -----------------------------------------------
    trainImages, trainHistograms, trainLabels = load_train(trainSetPath, imageSize, featureTypes[2], featureTypes[3])
    testImages, testHistograms, testLabels = load_test(testSetPath, imageSize, featureTypes[2], featureTypes[3])

    # Call the my_pca function to reduce data dimensionality
    if usePCA:
        goalVarPixel = 0.75  #Imposing the goal variance to achieve (PIXELS)
        goalVarHist = 0.8    #Imposing the goal variance to achieve (HISTOGRAMS)
        trainImages, testImages, trainHistograms, testHistograms = reduce_dimensionality(trainImages, imageSize, trainLabels, testImages, testLabels,
                                                                                         goalVarPixel, goalVarHist, trainHistograms, testHistograms)
    # Call SVC
    my_SVC(trainImages, trainLabels, testImages, testLabels, "RGB IMAGES", max_num_iter, showClassificationReport)
    my_SVC(trainHistograms, trainLabels, testHistograms, testLabels, "RGB HISTOGRAMS", max_num_iter, showClassificationReport)
    #------------------------------------------------------------------------------------------------------------------#


    # -----------------------------------------------
    # mean pixels and HSV histograms
    # -----------------------------------------------
    trainImages, trainHistograms, trainLabels = load_train(trainSetPath, imageSize, featureTypes[4], featureTypes[5])
    testImages, testHistograms, testLabels = load_test(testSetPath, imageSize, featureTypes[4], featureTypes[5])

    # Call the my_pca function to reduce data dimensionality
    if usePCA:
        goalVarPixel = 0.7  # Imposing the goal variance to achieve (PIXELS)
        goalVarHist = 0.9   # Imposing the goal variance to achieve (HISTOGRAMS)
        trainImages, testImages, trainHistograms, testHistograms = reduce_dimensionality(trainImages, imageSize, trainLabels, testImages, testLabels,
                                                                                         goalVarPixel, goalVarHist, trainHistograms, testHistograms)
    # Call SVC
    my_SVC(trainImages, trainLabels, testImages, testLabels, "MEAN PIXEL INTENSITY IMAGES", max_num_iter, showClassificationReport)
    my_SVC(trainHistograms, trainLabels, testHistograms, testLabels, "HSV HISTOGRAMS", max_num_iter, showClassificationReport)






