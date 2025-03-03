import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def tsneVisualization(X, y, maxSamples=1000, randomState=42):
    """
    Reduces embeddings to 2D using t-SNE and plots them.
    maxSamples: limit the number of points if the dataset is large.
    """
    if X.shape[0] > maxSamples:
        idx = np.random.choice(X.shape[0], maxSamples, replace=False)
        XSubset = X[idx]
        ySubset = y[idx]
    else:
        XSubset = X
        ySubset = y

    tsne = TSNE(n_components=2, random_state=randomState)
    X_embedded = tsne.fit_transform(XSubset)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], 
                    hue=ySubset, legend=False, palette="tab10")
    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

def plotRocCurvesComparison(meanFpr_euclid, meanTpr_euclid, meanAuc_euclid,
                            meanFpr_cosine, meanTpr_cosine, meanAuc_cosine):
    """
    Plots two ROC curves (Euclidean vs. Cosine) on the same figure, based on
    average TPR/FPR across cross-validation folds.
    
    meanFpr_*: 1D array of averaged FPR points
    meanTpr_*: 1D array of averaged TPR points
    meanAuc_*: float with the average AUC
    """
    plt.figure(figsize=(7, 6))
    plt.plot(meanFpr_euclid, meanTpr_euclid, color='blue',
             label=f"Euclid (AUC={meanAuc_euclid:.2f})")
    plt.plot(meanFpr_cosine, meanTpr_cosine, color='red',
             label=f"Cosine (AUC={meanAuc_cosine:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    
    plt.title("ROC Curve Comparison (Average Over CV Folds)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()
