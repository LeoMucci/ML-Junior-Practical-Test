import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.preprocessing import label_binarize

def computeTopKAccuracy(yTrue, yProba, classLabels, k=5):
    correct = 0
    for i in range(len(yTrue)):
        topKIdx = np.argsort(yProba[i])[::-1][:k]
        topKClasses = classLabels[topKIdx]
        if yTrue[i] in topKClasses:
            correct += 1
    return correct / len(yTrue)

def computeRocAucMulticlass(yTrue, yProba, average='macro'):
    uniqueClasses = np.unique(yTrue)
    yBinarized = label_binarize(yTrue, classes=uniqueClasses)
    aucVal = roc_auc_score(yBinarized, yProba, average=average, multi_class='ovr')
    return aucVal

def computeF1Macro(yTrue, yPred):
    return f1_score(yTrue, yPred, average='macro')

def aggregateRocCurves(yAllTrue, yAllProba, classLabels, nPoints=100):
    tprList = []
    aucVals = []
    # Create a fixed array of FPR values from 0 to 1
    meanFpr = np.linspace(0, 1, nPoints)

    for foldIdx in range(len(yAllTrue)):
        yTrueFold = yAllTrue[foldIdx]
        yProbaFold = yAllProba[foldIdx]

        # Binarize labels for multi-class ROC calculation
        uniqueClasses = classLabels
        yBin = label_binarize(yTrueFold, classes=uniqueClasses)

        # For macro-averaging, we need to sum up TPR for each class and divide
        nClasses = len(uniqueClasses)
        foldTprs = []

        for c in range(nClasses):
            fpr, tpr, _ = roc_curve(yBin[:, c], yProbaFold[:, c])
            # Interpolate TPR at the standard FPR points
            tprInterp = np.interp(meanFpr, fpr, tpr)
            foldTprs.append(tprInterp)
        
        # Calculate mean TPR for this fold
        foldMeanTpr = np.mean(foldTprs, axis=0)
        tprList.append(foldMeanTpr)

        # Calculate AUC for this fold
        aucVal = roc_auc_score(yBin, yProbaFold, average='macro', multi_class='ovr')
        aucVals.append(aucVal)

    # Calculate the average and standard deviation of TPR
    meanTpr = np.mean(tprList, axis=0)
    meanAuc = np.mean(aucVals)

    return meanFpr, meanTpr, meanAuc