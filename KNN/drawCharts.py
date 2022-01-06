import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
Features --> histogramType = #histograms_bw / histograms_RGB / histograms_HSV
             pixelType =     #rawPixels     / meanPixels     / rawPixels_bw
PCA --> [0.6, 0.9], increment=0.05
KNN --> K = [1,3,5,7], distance_metrics = ['euclidean', 'minkowski', 'cityblock']
'''



def create_3bars_chart(rangeX, dataY_1, dataY_2, dataY_3, labelsY, savingName, title, plotLabelX, plotLabelY):

    savingPath = ""
    barWidth = 0.25
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(dataY_1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, dataY_1, color=(0.533, 0.67, 0.81), width=barWidth, edgecolor='grey', label=labelsY[0])
    ax.tick_params(axis='x', labelsize=18)
    i = 0
    for x_loc in br1:
        plt.text(x_loc - 0.18, dataY_1[i], "{:.2f}".format(dataY_1[i]), fontsize=11);  i = i + 1
    plt.bar(br2, dataY_2, color=(0.95, 0.627, 0.34), width=barWidth, edgecolor='grey', label=labelsY[1])
    ax.tick_params(axis='x', labelsize=18)
    i = 0
    for x_loc in br2:
        plt.text(x_loc - 0.14, dataY_2[i], "{:.2f}".format(dataY_2[i]), fontsize=11);  i = i + 1
    plt.bar(br3, dataY_3, color=(0.525, 0.7, 0.498), width=barWidth, edgecolor='grey', label=labelsY[2])
    ax.tick_params(axis='x', labelsize=18)
    i = 0
    for x_loc in br3:
        plt.text(x_loc - 0.09, dataY_3[i], "{:.2f}".format(dataY_3[i]), fontsize=11);  i = i + 1

    # ---- added for the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.04),
              ncol=4, fancybox=True, shadow=True)
    plt.title(title, fontweight='bold', fontsize=18, y=1.04)
    # -----
    plt.ylabel(plotLabelY, fontweight='bold', fontsize=18)
    plt.xlabel(plotLabelX, fontweight='bold', fontsize=18)
    plt.xticks([r + barWidth for r in range(len(dataY_1))], rangeX)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)

    plt.tight_layout()
    plt.savefig(savingPath + savingName)

    return fig, ax



def create_4bars_chart(rangeX, dataY_1, dataY_2, dataY_3, dataY_4, labelsY, savingName, title, plotLabelX, plotLabelY):

    savingPath = ""
    barWidth = 0.2
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(dataY_1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    plt.bar(br1, dataY_1, color=(0.533, 0.67, 0.81), width=barWidth, edgecolor='grey', label=labelsY[0])
    ax.tick_params(axis='x', labelsize=18)
    i = 0;
    for x_loc in br1:
        plt.text(x_loc - 0.06, dataY_1[i], "{:.2f}".format(dataY_1[i]), fontsize=11); i = i + 1
    plt.bar(br2, dataY_2, color=(0.95, 0.627, 0.34), width=barWidth, edgecolor='grey', label=labelsY[1])
    ax.tick_params(axis='x', labelsize=18)
    i = 0;
    for x_loc in br2:
        plt.text(x_loc - 0.06, dataY_2[i], "{:.2f}".format(dataY_2[i]), fontsize=11); i = i + 1
    plt.bar(br3, dataY_3, color=(0.525, 0.7, 0.498), width=barWidth, edgecolor='grey', label=labelsY[2])
    ax.tick_params(axis='x', labelsize=18)
    i = 0;
    for x_loc in br3:
        plt.text(x_loc - 0.06, dataY_3[i], "{:.2f}".format(dataY_3[i]), fontsize=11); i = i + 1
    plt.bar(br4, dataY_4, color=(0.847, 0.562, 0.9), width=barWidth, edgecolor='grey', label=labelsY[3])
    ax.tick_params(axis='x', labelsize=18)
    i = 0;
    for x_loc in br4:
        plt.text(x_loc - 0.06, dataY_4[i], "{:.2f}".format(dataY_4[i]), fontsize=11); i = i + 1

    # ---- added for the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.04),
              ncol=4, fancybox=True, shadow=True)
    plt.title(title, fontweight='bold', fontsize=18, y=1.04)
    # -----

    plt.ylabel(plotLabelY, fontweight='bold', fontsize=18)
    plt.xlabel(plotLabelX, fontweight='bold', fontsize=18)
    plt.xticks([r + barWidth+0.1 for r in range(len(dataY_1))], rangeX)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)

    #plt.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(savingPath + savingName)

    return fig, ax

"""
#--------------------------------------------------------------------------------------------------------------
#Create the pixels PCA tuning plot
#--------------------------------------------------------------------------------------------------------------
df_1 = pd.read_csv('previouslySavedLogs/PCA_accuracy_rawPixels_RGB.csv')
df_2 = pd.read_csv('previouslySavedLogs/PCA_accuracy_rawPixels_bw.csv')
df_3 = pd.read_csv('previouslySavedLogs/PCA_accuracy_meanPixels.csv')
savingName = "images/PCA_tuning_pixels.jpg"

rangeX = [str(i / 100) for i in range(60, 95, 5)]  # list = [0.6, ..., 0.9]
labels = ['Raw pixels (b&w) features', 'Raw pixels (RGB) features', 'Mean pixels features']
title = '[PCA tuning] Pixel features'
plotLabelX = 'PCA goal variance'
plotLabelY = 'PCA classifier accuracy [%]'
create_3bars_chart(rangeX, dataY_1=df_1.iloc[0], dataY_2=df_2.iloc[0], dataY_3=df_3.iloc[0], labelsY=labels, savingName=savingName, title=title, plotLabelX=plotLabelX, plotLabelY=plotLabelY)
#--------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------
#Create the histograms PCA tuning plot
#--------------------------------------------------------------------------------------------------------------
df_1 = pd.read_csv('previouslySavedLogs/PCA_accuracy_histograms_RGB.csv')
df_2 = pd.read_csv('previouslySavedLogs/PCA_accuracy_histograms_bw.csv')
df_3 = pd.read_csv('previouslySavedLogs/PCA_accuracy_histograms_HSV.csv')
savingName = "images/PCA_tuning_histograms.jpg"

rangeX = [str(i / 100) for i in range(60, 95, 5)]  # list = [0.6, ..., 0.9]
labels = ['Histogram (b&w) features', 'Histogram (RGB) features', 'Histogram (HSV) features']
title = '[PCA tuning] Histograms features'
plotLabelX = 'PCA goal variance'
plotLabelY = 'PCA classifier accuracy [%]'
create_3bars_chart(rangeX, dataY_1=df_1.iloc[0], dataY_2=df_2.iloc[0], dataY_3=df_3.iloc[0], labelsY=labels, savingName=savingName, title=title, plotLabelX=plotLabelX, plotLabelY=plotLabelY)



#--------------------------------------------------------------------------------------------------------------
#Create KNN tuning pixel bw features plot
#--------------------------------------------------------------------------------------------------------------
df = pd.read_csv('previouslySavedLogs/K_accuracy_rawPixels_bw.csv')
savingName = "images/KNN_tuning_pixels_BW.jpg"

rangeX = ['1', '3', '5', '7']
labels = ['Euclidean distance metric', 'Cosine distance metric', 'Jaccard distance metric', 'Mahalanobis distance metric']
title = '[KNN tuning] Raw pixels BW features'
plotLabelX = 'K value'
plotLabelY = 'KNN classifier accuracy [%]'
create_4bars_chart(rangeX, dataY_1=df.iloc[0], dataY_2=df.iloc[1], dataY_3=df.iloc[2], dataY_4=df.iloc[3], labelsY=labels, savingName=savingName, title=title, plotLabelX=plotLabelX, plotLabelY=plotLabelY)
"""





#--------------------------------------------------------------------------------------------------------------
#Create the pixels PCA tuning plot
#--------------------------------------------------------------------------------------------------------------
df_1 = pd.read_csv('previouslySavedLogs/PCA_accuracy_rawPixels_bw.csv')
df_2 = pd.read_csv('previouslySavedLogs/PCA_accuracy_rawPixels_RGB.csv')
df_3 = pd.read_csv('previouslySavedLogs/PCA_accuracy_meanPixels.csv')
savingName = "images/PCA_tuning_pixels.jpg"

rangeX = [str(i / 100) for i in range(60, 95, 5)]  # list = [0.6, ..., 0.9]
labels = ['Raw pixels (b&w) features', 'Raw pixels (RGB) features', 'Mean pixels features']
title = '[PCA tuning] Pixel features'
plotLabelX = 'PCA goal variance'
plotLabelY = 'PCA classifier accuracy [%]'
create_3bars_chart(rangeX, dataY_1=df_1.iloc[0], dataY_2=df_2.iloc[0], dataY_3=df_3.iloc[0], labelsY=labels, savingName=savingName, title=title, plotLabelX=plotLabelX, plotLabelY=plotLabelY)
#--------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------
#Create the histograms PCA tuning plot
#--------------------------------------------------------------------------------------------------------------
df_1 = pd.read_csv('previouslySavedLogs/PCA_accuracy_histograms_bw.csv')
df_2 = pd.read_csv('previouslySavedLogs/PCA_accuracy_histograms_RGB.csv')
df_3 = pd.read_csv('previouslySavedLogs/PCA_accuracy_histograms_HSV.csv')
savingName = "images/PCA_tuning_histograms.jpg"

rangeX = [str(i / 100) for i in range(60, 95, 5)]  # list = [0.6, ..., 0.9]
labels = ['Histogram (b&w) features', 'Histogram (RGB) features', 'Histogram (HSV) features']
title = '[PCA tuning] Histograms features'
plotLabelX = 'PCA goal variance'
plotLabelY = 'PCA classifier accuracy [%]'
create_3bars_chart(rangeX, dataY_1=df_1.iloc[0], dataY_2=df_2.iloc[0], dataY_3=df_3.iloc[0], labelsY=labels, savingName=savingName, title=title, plotLabelX=plotLabelX, plotLabelY=plotLabelY)



#--------------------------------------------------------------------------------------------------------------
#Create KNN tuning pixel bw features plot
#--------------------------------------------------------------------------------------------------------------
df = pd.read_csv('previouslySavedLogs/K_accuracy_rawPixels_bw.csv')
savingName = "images/KNN_tuning_pixels_BW.jpg"

rangeX = ['1', '3', '5', '7']
labels = ['Euclidean distance metric', 'Cosine distance metric', 'Jaccard distance metric', 'Mahalanobis distance metric']
title = '[KNN tuning] Raw pixels BW features'
plotLabelX = 'K value'
plotLabelY = 'KNN classifier accuracy [%]'
create_4bars_chart(rangeX, dataY_1=df.iloc[0], dataY_2=df.iloc[1], dataY_3=df.iloc[2], dataY_4=df.iloc[3], labelsY=labels, savingName=savingName, title=title, plotLabelX=plotLabelX, plotLabelY=plotLabelY)
#--------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------
#Create KNN tuning histograms bw features plot
#--------------------------------------------------------------------------------------------------------------
df = pd.read_csv('previouslySavedLogs/K_accuracy_histograms_bw.csv')
savingName = "images/KNN_tuning_histograms_BW.jpg"
title = '[KNN tuning] Histogram (b&w) features'
create_4bars_chart(rangeX, dataY_1=df.iloc[0], dataY_2=df.iloc[1], dataY_3=df.iloc[2], dataY_4=df.iloc[3], labelsY=labels, savingName=savingName, title=title, plotLabelX=plotLabelX, plotLabelY=plotLabelY)
#--------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------
#Create KNN tuning pixels RGB features plot
#--------------------------------------------------------------------------------------------------------------
df = pd.read_csv('previouslySavedLogs/K_accuracy_rawPixels_RGB.csv')
savingName = "images/KNN_tuning_pixels_RGB.jpg"
title = '[KNN tuning] Raw pixel RGB features'
create_4bars_chart(rangeX, dataY_1=df.iloc[0], dataY_2=df.iloc[1], dataY_3=df.iloc[2], dataY_4=df.iloc[3], labelsY=labels, savingName=savingName, title=title, plotLabelX=plotLabelX, plotLabelY=plotLabelY)
#--------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------
#Create KNN tuning histograms RGB features plot
#--------------------------------------------------------------------------------------------------------------
df = pd.read_csv('previouslySavedLogs/K_accuracy_histograms_RGB.csv')
savingName = "images/KNN_tuning_histograms_RGB.jpg"
title = '[KNN tuning] Histogram (RGB) features'
create_4bars_chart(rangeX, dataY_1=df.iloc[0], dataY_2=df.iloc[1], dataY_3=df.iloc[2], dataY_4=df.iloc[3], labelsY=labels, savingName=savingName, title=title, plotLabelX=plotLabelX, plotLabelY=plotLabelY)
#--------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------
#Create KNN tuning histograms HSV features plot
#--------------------------------------------------------------------------------------------------------------
df = pd.read_csv('previouslySavedLogs/K_accuracy_histograms_HSV.csv')
savingName = "images/KNN_tuning_histograms_HSV.jpg"
title = '[KNN tuning] Histogram (HSV) features'
create_4bars_chart(rangeX, dataY_1=df.iloc[0], dataY_2=df.iloc[1], dataY_3=df.iloc[2], dataY_4=df.iloc[3], labelsY=labels, savingName=savingName, title=title, plotLabelX=plotLabelX, plotLabelY=plotLabelY)
#--------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------
#Create KNN tuning mean pixels features plot
#--------------------------------------------------------------------------------------------------------------
df = pd.read_csv('previouslySavedLogs/K_accuracy_meanPixels.csv')
savingName = "images/KNN_tuning_pixels_MEAN.jpg"
title = '[KNN tuning] Mean pixel features'
create_4bars_chart(rangeX, dataY_1=df.iloc[0], dataY_2=df.iloc[1], dataY_3=df.iloc[2], dataY_4=df.iloc[3], labelsY=labels, savingName=savingName, title=title, plotLabelX=plotLabelX, plotLabelY=plotLabelY)
#--------------------------------------------------------------------------------------------------------------


