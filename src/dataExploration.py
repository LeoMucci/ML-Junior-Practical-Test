import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def checkMissingData(df):
    """
    This function shows how many values are missing in each column
    Super useful to know if we need to clean the data before analysis
    """
    missing = df.isnull().sum()
    print("Missing data in each column:")
    print(missing)
    # Could add some code here to remove or fill in missing data

def plotClassDistribution(dfFlattened):
    """
    Creates a bar chart showing how many images we have for each syndrome
    Important to see if the data is balanced or if any syndrome has too few examples
    """
    syndromeCounts = dfFlattened['syndromeId'].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=syndromeCounts.index, y=syndromeCounts.values)
    plt.title("Number of Images by Syndrome")
    plt.xlabel("Syndrome ID")
    plt.ylabel("Image Count")
    plt.show()
    # Seaborn makes the graphs look way better than plain matplotlib!

def basicStats(dfFlattened):
    """
    Shows some basic stats about the dataset:
      - total samples we have
      - how many different syndromes exist
      - min and max samples per syndrome
    
    This is important because if a class has very few examples,
    the model will struggle to learn
    """
    totalSamples = len(dfFlattened)
    uniqueSyndromes = dfFlattened['syndromeId'].nunique()
    counts = dfFlattened['syndromeId'].value_counts()

    print(f"Total number of samples: {totalSamples}")
    print(f"Number of unique syndromes: {uniqueSyndromes}")
    print(f"Samples per syndrome (min to max): {counts.min()} - {counts.max()}")
    # If any syndrome has way fewer examples than others, might be good to use balancing techniques like SMOTE