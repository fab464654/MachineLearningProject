

#Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


#Main function to apply PCA algorithm
def my_pca(data, img_w, img_h, train_labels, test_data, test_labels, training_phase, N, goal_variance, show_plot, show_comparison):

    data = np.transpose(data) # "vectorized" images are in columns (N_features x N_images)

    #3 I apply PCA and project the train images into a space of dimension N

    # 3a. I calculate the mean and centre the data
    m = np.mean(data, axis=1)     #for every pixel (i,j) mean between all the images pixels (row vector of nRows x nColumns elements)
    Xc = data - m[:, np.newaxis]  #centralized coordinates over "m" (m 1D vector --> m[:,np.newaxis] column vector)

    # 3b. I calculate the covariance matrix
    C = np.cov(Xc)

    # 3c. Extracting eigenvectors (U) and eigenvalues (lambdas) of the covariance matrix
    lambdas, U = np.linalg.eigh(C)

    # 3d. I order the eigenvalues from largest to smallest
    best_eig_idxs = np.argsort(lambdas)[::-1]  #ascending sort that is then reversed to achieve descending sorting
                                               #argsort returns indeces
    best_eig = lambdas[best_eig_idxs]  #get the "highest" eigenvalues
    best_U = U[:, best_eig_idxs]       #get the associated eigenvectors to their eigenvalues

    # 3e. I check the amount of variance in the data that each eigenvalue carries and set N equal to the number of eigenvectors sufficient to have at least 80%
    # of the total variance.
    d = lambdas.shape[0]  #number of eigenvalues

    if training_phase:
        y = np.cumsum(best_eig) / np.sum(best_eig)  # normalize eigenvalues between 0 and 1 (0= the most significative)
        if show_plot:
            fig, axs = plt.subplots(2)
            axs[0].plot(np.arange(1, d + 1), best_eig)  # draw the line (xlimit = [1,256])
            axs[0].scatter(np.arange(1, d + 1), best_eig)  # scatter the eigenvalues points
            axs[1].plot(np.arange(1, d + 1), y)  # draw the line (xlimit = [1,256])
            axs[1].scatter(np.arange(1, d + 1), y)  # scatter the values
            plt.show()

        N = np.where(y >= goal_variance)[0][0]  #get the first ("0") element that satisfies y>=...
        if show_comparison:
            print("[myPCA] From PCA (implemented step by step) we get that the number of components to have",
                   str(np.round(goal_variance, 2)), "explained variance is", N)

    # 3f. I project the data using the N largest eigenvectors
    first_N_eigenvectors = best_U[:, :N]   #get the first N more significant eigenvectors (previously ordered)
    projectedImages = first_N_eigenvectors.T.dot(Xc)

    if training_phase:
        # Computing the accuracy achieved by the PCA classifier
        accuracy = test_my_pca(first_N_eigenvectors, projectedImages, train_labels, test_data, test_labels, m, show_updates=show_comparison)
        return np.transpose(projectedImages), N, accuracy
    else:
        return np.transpose(projectedImages)

#Function that prints the accuracy of the PCA feature reduction
def test_my_pca(first_N_eigenvectors, projected_images, train_labels, test_data, test_labels, m, show_updates):
    # projected_images: "vectorized" images in columns, with reduced number of features (from PCA)
    #                   (nReducedFeaturePCA x nImages)

    # 4. Calculating the Theta threshold
    from scipy.spatial.distance import cdist
    theta = np.max(cdist(projected_images.T, projected_images.T, 'euclidean'))

    # 5. Centre my test data
    test_data = np.transpose(test_data)  #to have "vectorized" images along columns

    # centralized (pixel intensities of) TEst images (m 1D array --> m[:, np.newaxis] columns vector np.shape=(nElements,1) )
    x_te = test_data - m[:, np.newaxis] # subtract from each column of test_data the column m[:, np.newaxis]

    # test_data images transformed into the reduced feature space (reduced by PCA)
    omega_te = first_N_eigenvectors.T.dot(x_te)
    # x_te: original test images centralized (nOriginalPixels x nImages)
    # first_N_eigenvectors: (nOriginalPixels,nReducedFeatures) (notice. the pixels are features, so they are synonim in this case)
    # omega_te: (nReducedFeatures,nImages) set of reduced test images

    # 6. Calculating the set of epsilon distances
    epsilon = []
    for i in range(test_data.shape[1]): # iterate over all test images
        tmp_test = omega_te[:, i]       # i-th reduced test image
        # tmp_test[:, np.newaxis]: column vector reduced image (i-th reduced TEST image) (nReducedFeatures,1)
        # projected_images: (nReducedFeatures,nTrainImages)
        # tmp_test[:, np.newaxis] - projected_images: projected_images: (nReducedFeatures, nTrainImages), where the j-th column more similar to tmp_test will be similar or equalt to null vector (nReducedFeatures x 1)
        # epsilon[j] = np.linalg.norm(tmp_test[:, np.newaxis] - projected_images, ord=2, axis=0): (nTestImages,) 1D array, where the k-th element is the norm of the above matrix k-th column
        #            '-> the smaller norm in this row, will correspond to the more similar test image in train set "projected_images"
        epsilon.append(np.linalg.norm(tmp_test[:, np.newaxis] - projected_images, ord=2, axis=0)) # ord=2 Euclidean norm/2-norm of COLUMNS
        # v = np.array([x1 x2 ... xN], [y1 y2 ... yN]) -> norm(v,axis=0) = array([sqrt(x1^2+y1^2), sqrt(x2^2+y2^2), ... , sqrt(xN^2+yN^2)])
        #              '-> matrix columns are the vectors of which you compute the norm
        # vs
        # v = np.array([x1 y1], [x2 y2], ... [xN yN]]) -> norm(v,axis=1) = array([sqrt(x1^2+y1^2), sqrt(x2^2+y2^2), ... , sqrt(xN^2+yN^2)])
        #              '-> matrix rows are the vectors of which you compute the norm

    epsilon = np.array(epsilon)
    # epsilon: (nTestImages,nTrainImages) where in each row, the smaller l-th value (of norm) indicates which is the image of
    #           the train set "projected_images" more similar to the test image omega_te[:,l]!

    # 7. I reconstruct the images (transf. from reducedFeatures space to entire resolution space)
    g = first_N_eigenvectors.dot(omega_te)
    # omega_te: (nReducedFeatures,nImages) set of reduced test images
    # first_N_eigenvectors: (nOriginalPixels, nReducedFeatures)
    # g: (nOriginalPixels,nImages) reduced test images projected in the feature space of entire dimension (dimension of
    #                              original input images)

    """
    #  Make an imshow of the original image against the reconstructed one (only the first 5 images)!
    r = 32; c = 32 # modify it manually!
    fig, axs = plt.subplots(5, 2)
    for i in range(5):
        axs[i, 0].imshow(x_te[:, i].reshape((r, c)), cmap='gray')
        axs[i, 1].imshow(g[:, i].reshape((r, c)), cmap='gray')
    plt.show()
    """

    # 8. Calculation xi for classification
    xi = np.linalg.norm(g - x_te, ord=2, axis=0)

    # 9. In which of the 3 cases are we for each test face? Is the corresponding face of the same person? Check the first 5 faces
    ''' # It does not work in our case, because Xc variable is defined out of scope of this function
    fig, axs = plt.subplots(5, 2)
    for i in range(5):
        if xi[i] >= theta:
            print(str(i + 1) + ": It's not a face!")
        elif xi[i] < theta and any(epsilon[i, :] > theta):
            print(str(i + 1) + ": It's a new face!")
        elif xi[i] < theta and np.min(epsilon[i, :]) < theta:
            print(str(i + 1) + ": It's a familiar face! I'll show you!")
            matched_indx = np.argmin(epsilon[i, :])
            axs[i, 0].imshow(x_te[:, i].reshape((r, c)), cmap='gray')
            axs[i, 1].imshow(Xc[:, matched_indx].reshape((r, c)), cmap='gray')
            if i == 0:
                axs[i, 0].set_title('Unknown face!')
                axs[i, 1].set_title('Known face!')
    plt.show()
    '''

    # 10. Calculate the accuracy of the classifier and test how the result changes when N (+/- eigenvectors) is changed
    # Set the prediction equal to -1 in case of classification as no face or new face,
    # equal to the train sample label with lower epsilon if a match is found
    predicted = []
    for i in range(test_data.shape[1]):
        if xi[i] < theta and np.min(epsilon[i, :]) < theta:
            predicted.append(train_labels[np.argmin(epsilon[i, :])])
        else:
            predicted.append(-1)

    predicted = np.array(predicted)

    accuracy = np.sum(predicted == test_labels)
    if show_updates:
        print('[myPCA] Classifier accuracy: ' + "{0:.2f}".format(accuracy / len(test_labels) * 100) + '%\n')
    return accuracy / len(test_labels) * 100


def my_pca_tuning(increment_range, increment, imageSize, trainImages, trainLabels, testImages, testLabels, show_updates):
    startingVariance = increment_range[0]
    finalVariance = increment_range[1]
    actualVariance = startingVariance

    bestGoalVariance = 0
    bestAccuracy = 0
    while actualVariance < finalVariance:
        if show_updates:
            print("[PCA tuning] Considering variance=" + str(np.round(actualVariance, 2)))

        _, N, accuracy = my_pca(trainImages, imageSize, imageSize, trainLabels, testImages, testLabels,
                                training_phase=True, N=None, goal_variance=actualVariance, show_plot=False,
                                show_comparison=show_updates)  #reducing feature space
        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestGoalVariance = actualVariance
            N_best = N

        actualVariance += increment

    return bestGoalVariance, bestAccuracy, N_best


def sklearn_pca(features, goalVariance):

    scaler = MinMaxScaler()
    scaledFeatures = scaler.fit_transform(features)

    pca = PCA(n_components=goalVariance)
    pca.fit_transform(scaledFeatures)

    print("[sklearn PCA] PCA components considering the set variance ratio (" + str(np.round(goalVariance, 2)) + "): "
          + str(pca.explained_variance_ratio_.shape[0]))
