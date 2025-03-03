import pickle
import pandas as pd
import numpy as np

def loadPickleData(picklePath):
    """
    Loads the pickle file containing the hierarchical dataset
    Using pickle format maintains the data types and structure
    """
    with open(picklePath, 'rb') as file:
        dataDict = pickle.load(file)
    return dataDict

def flattenHierarchicalData(dataDict):
    """
    Transforms the nested dictionary structure:
      {
        'syndrome_id': {
          'subject_id': {
            'image_id': [320-dim embedding]
          }
        }
      }
    
    Into a DataFrame with columns: 
    [syndromeId, subjectId, imageId, embeddingVector]
    
    This makes the data easier to process for analysis and modeling
    """
    rows = []
    for syndromeId, subjectDict in dataDict.items():
        for subjectId, imageDict in subjectDict.items():
            for imageId, embeddingArr in imageDict.items():
                embArray = np.array(embeddingArr, dtype=float)
                rowData = {
                    'syndromeId': syndromeId,
                    'subjectId': subjectId,
                    'imageId': imageId,
                    'embeddingVector': embArray
                }
                rows.append(rowData)

    dfFlattened = pd.DataFrame(rows)
    return dfFlattened

def splitFeaturesLabels(dfFlattened):
    """
    Extracts features (X) and target labels (y) from the flattened DataFrame
    X: feature matrix of shape (n_samples, 320)
    y: target vector containing syndrome IDs
    
    Returns data in the format required for machine learning models
    """
    X = np.vstack(dfFlattened['embeddingVector'].values)
    y = dfFlattened['syndromeId'].values
    return X, y