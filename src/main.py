import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tabulate import tabulate

from dataPreprocessing import loadPickleData, flattenHierarchicalData, splitFeaturesLabels
from dataExploration import checkMissingData, plotClassDistribution, basicStats
from visualization import tsneVisualization, plotRocCurvesComparison
from knnClassification import runKnnCrossValidation, runKnnCrossValWithRoc
from metricUtils import aggregateRocCurves

def printF1ScoresAsTable(title, resultsDict):
    """
    Formats and displays F1-scores in a readable table format
    
    Takes a dictionary mapping k values to their corresponding F1-scores
    sorts by k value, and displays using tabulate for better readability
    """
    data = [(k, resultsDict[k]) for k in sorted(resultsDict.keys())]
    df = pd.DataFrame(data, columns=["k", "F1-Score"])

    print("\n" + title)
    print(tabulate(df, headers="keys", showindex=False, tablefmt="psql", floatfmt=".4f"))

def printDetailedStats(title, statsDict):
    """
    Formats and displays model performance statistics in a table
    
    Takes a dictionary of statistics (e.g., meanF1, stdF1, meanTop5, meanAUC)
    and formats them into a readable table using tabulate
    """
    data = [(k, statsDict[k]) for k in statsDict]
    df = pd.DataFrame(data, columns=["Metric", "Value"])

    print("\n" + title)
    print(tabulate(df, headers="keys", showindex=False, tablefmt="psql", floatfmt=".4f"))

def main():
    # 1) Load and Flatten
    picklePath = "data/mini_gm_public_v0.1.p"
    dataDict = loadPickleData(picklePath)
    dfFlattened = flattenHierarchicalData(dataDict)

    # 2) Check data integrity
    checkMissingData(dfFlattened)

    # 3) Basic EDA
    basicStats(dfFlattened)
    plotClassDistribution(dfFlattened)

    # 4) Split features/labels
    X, y = splitFeaturesLabels(dfFlattened)
    print("X shape:", X.shape, "| y shape:", y.shape)

    # 5) t-SNE Visualization (optional, can be commented out if slow)
    tsneVisualization(X, y, maxSamples=1000)

    # 6) Cross-Validation: find best k
    kRange = range(1, 16)
    euclidResults = runKnnCrossValidation(X, y, distanceMetric='euclidean', kValues=kRange, nSplits=10)
    cosineResults = runKnnCrossValidation(X, y, distanceMetric='cosine', kValues=kRange, nSplits=10)

    # Identifying best k for each metric
    bestKEuclid = max(euclidResults, key=euclidResults.get)
    bestKCosine = max(cosineResults, key=cosineResults.get)

    # Display F1-scores in table format
    printF1ScoresAsTable("F1-scores (Euclidean)", euclidResults)
    print(f"Best k (Euclid): {bestKEuclid}")

    printF1ScoresAsTable("F1-scores (Cosine)", cosineResults)
    print(f"Best k (Cosine): {bestKCosine}")

    # 7) Detailed metrics + ROC aggregation for Euclid
    stats_euclid, yAllTrue_euc, yAllProba_euc = runKnnCrossValWithRoc(
        X, y, bestK=bestKEuclid, distanceMetric='euclidean', nSplits=10
    )

    # 8) Detailed metrics + ROC aggregation for Cosine
    stats_cosine, yAllTrue_cos, yAllProba_cos = runKnnCrossValWithRoc(
        X, y, bestK=bestKCosine, distanceMetric='cosine', nSplits=10
    )

    # Format and display detailed statistics in tables
    printDetailedStats("=== Detailed Stats (Euclidean) ===", stats_euclid)
    printDetailedStats("=== Detailed Stats (Cosine) ===", stats_cosine)

    # 9) Aggregate ROC curves across folds
    uniqueClasses = np.unique(y)

    meanFpr_euclid, meanTpr_euclid, meanAuc_euclid = aggregateRocCurves(yAllTrue_euc, yAllProba_euc, uniqueClasses)
    meanFpr_cosine, meanTpr_cosine, meanAuc_cosine = aggregateRocCurves(yAllTrue_cos, yAllProba_cos, uniqueClasses)

    # 10) Plot both curves on the same graph
    plotRocCurvesComparison(meanFpr_euclid, meanTpr_euclid, meanAuc_euclid,
                            meanFpr_cosine, meanTpr_cosine, meanAuc_cosine)
    # Uncomment if plotRocCurvesComparison doesn't call plt.show()
    # plt.show()

if __name__ == "__main__":
    main()