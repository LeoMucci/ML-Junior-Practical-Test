import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from metricUtils import computeF1Macro, computeTopKAccuracy, computeRocAucMulticlass, aggregateRocCurves

def runKnnCrossValidation(X, y, distanceMetric, kValues, nSplits=10):
    """
    Tests different k values using cross-validation
    
    Uses StratifiedKFold to maintain class distribution in each fold
    For each k value, calculates the average F1-score across all folds
    
    """
    skf = StratifiedKFold(n_splits=nSplits, shuffle=True, random_state=42)
    results = {}

    for k in kValues:
        f1Scores = []
        for trainIdx, testIdx in skf.split(X, y):
            XTrain, XTest = X[trainIdx], X[testIdx]
            yTrain, yTest = y[trainIdx], y[testIdx]

            knn = KNeighborsClassifier(n_neighbors=k, metric=distanceMetric)
            knn.fit(XTrain, yTrain)
            yPred = knn.predict(XTest)

            f1Val = computeF1Macro(yTest, yPred)
            f1Scores.append(f1Val)

        results[k] = np.mean(f1Scores)
    return results

def runKnnCrossValWithRoc(X, y, bestK, distanceMetric, nSplits=10):
    """
    Runs 10-fold cross-validation and stores true labels and probability predictions
    for each fold to generate average ROC curves
    
    Also calculates:
    - Average F1 score
    - Top-5 accuracy
    - Multi-class AUC
    
    Returns a dictionary with stats and arrays needed for ROC curve aggregation
    """
    skf = StratifiedKFold(n_splits=nSplits, shuffle=True, random_state=42)

    yAllTrue = []
    yAllProba = []
    f1Scores = []
    top5Scores = []
    aucScores = []

    # Get the unique labels to know their order
    # Note: We could also get this from knn.classes_ if we need the same order
    uniqueLabels = np.unique(y)

    for trainIdx, testIdx in skf.split(X, y):
        XTrain, XTest = X[trainIdx], X[testIdx]
        yTrain, yTest = y[trainIdx], y[testIdx]

        knn = KNeighborsClassifier(n_neighbors=bestK, metric=distanceMetric)
        knn.fit(XTrain, yTrain)

        yPred = knn.predict(XTest)
        proba = knn.predict_proba(XTest)

        # Calculate F1 score
        f1Val = computeF1Macro(yTest, yPred)
        f1Scores.append(f1Val)

        # Calculate Top-5 accuracy (if fewer than 5 classes, becomes top-n)
        top5Val = computeTopKAccuracy(yTest, proba, knn.classes_, k=5)
        top5Scores.append(top5Val)

        # Calculate multi-class AUC
        aucVal = computeRocAucMulticlass(yTest, proba)
        aucScores.append(aucVal)

        # Store for ROC curve aggregation later
        yAllTrue.append(yTest)
        yAllProba.append(proba)

    stats = {
        'meanF1': np.mean(f1Scores),
        'stdF1': np.std(f1Scores),
        'meanTop5': np.mean(top5Scores),
        'meanAUC': np.mean(aucScores),
        'stdAUC': np.std(aucScores)
    }
    return stats, yAllTrue, yAllProba