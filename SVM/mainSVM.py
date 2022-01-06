import os.path

from my_SVM import *
from imutils import paths
import sys

if not os.path.exists("./images"):
    os.makedirs("./images")
logFile = open("./images/terminal_output.txt","w+")
# To write both in terminal and file
class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        logFile.write(data)    # Write the data of stdout here to a text file as well

    def flush(self):
        pass

sys.stdout=Unbuffered(sys.stdout)

#Set dataset parameters
trainPath = list(paths.list_images("../dataset/seg_train"))
testPath = list(paths.list_images("../dataset/seg_test"))
#trainPath = list(paths.list_images("../NatureDatasetReduced/train"))
#testPath = list(paths.list_images("../NatureDatasetReduced/test"))

imageSize = 32

#Decide whether or not using PCA (given for granted that PCA tuning was previously performed inside KNN code)
usePCA = True

#showClassificationReport = True

testDifferentSVMclassifiers(trainPath, testPath, imageSize, usePCA, showClassificationReport=True)

logFile.close()
sys.stdout = sys.__stdout__
print("\n\n")
print(">>> Execution terminated! <<<")
# To let open figures
# plt.show()



