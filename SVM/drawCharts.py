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
    i = 0;
    for x_loc in br1:
        plt.text(x_loc - 0.08, dataY_1[i], "{:.2f}".format(dataY_1[i]), fontsize=14);
        i = i + 1
    plt.bar(br2, dataY_2, color=(0.95, 0.627, 0.34), width=barWidth, edgecolor='grey', label=labelsY[1])
    ax.tick_params(axis='x', labelsize=18)
    i = 0;
    for x_loc in br2:
        plt.text(x_loc - 0.08, dataY_2[i], "{:.2f}".format(dataY_2[i]), fontsize=14);
        i = i + 1
    plt.bar(br3, dataY_3, color=(0.525, 0.7, 0.498), width=barWidth, edgecolor='grey', label=labelsY[2])
    ax.tick_params(axis='x', labelsize=18)
    i = 0;
    for x_loc in br3:
        plt.text(x_loc - 0.08, dataY_3[i], "{:.2f}".format(dataY_3[i]), fontsize=14);
        i = i + 1

    # ---- added for the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.04),
              ncol=4, fancybox=True, shadow=True)
    plt.title(title, fontweight='bold', fontsize=18, y=1.04)
    # -----    plt.ylabel(plotLabelY, fontweight='bold', fontsize=18)
    plt.xlabel(plotLabelX, fontweight='bold', fontsize=18)
    plt.xticks([r + barWidth for r in range(len(dataY_1))], rangeX)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
    plt.legend(fontsize=16)

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
    i = 0
    for x_loc in br1:
        plt.text(x_loc - 0.06, dataY_1[i], "{:.2f}".format(dataY_1[i]), fontsize=13); i = i + 1
    plt.bar(br2, dataY_2, color=(0.95, 0.627, 0.34), width=barWidth, edgecolor='grey', label=labelsY[1])
    ax.tick_params(axis='x', labelsize=18)
    i = 0
    for x_loc in br2:
        plt.text(x_loc - 0.06, dataY_2[i], "{:.2f}".format(dataY_2[i]), fontsize=13); i = i + 1
    plt.bar(br3, dataY_3, color=(0.525, 0.7, 0.498), width=barWidth, edgecolor='grey', label=labelsY[2])
    ax.tick_params(axis='x', labelsize=18)
    i = 0
    for x_loc in br3:
        plt.text(x_loc - 0.06, dataY_3[i], "{:.2f}".format(dataY_3[i]), fontsize=13); i = i + 1
    plt.bar(br4, dataY_4, color=(0.847, 0.562, 0.9), width=barWidth, edgecolor='grey', label=labelsY[3])
    ax.tick_params(axis='x', labelsize=18)
    i = 0
    for x_loc in br4:
        plt.text(x_loc - 0.06, dataY_4[i], "{:.2f}".format(dataY_4[i]), fontsize=13); i = i + 1

    plt.title(title, fontweight='bold', fontsize=18)
    plt.xlabel(plotLabelX, fontweight='bold', fontsize=18)
    plt.xticks([r + barWidth+0.1 for r in range(len(dataY_1))], rangeX)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
    plt.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(savingPath + savingName)

    return fig, ax














