# ML Junior Practical Test

Hey there! Welcome to **ML Junior Practical Test**! 😃

This is a Machine Learning project where we try to classify syndromes (`syndromeId`) using some fancy 320-dimensional embeddings. Below, I'll guide you through what this project does, how you can run it, and what results you can expect.

## 📌 Table of Contents

- [What’s This About?](#whats-this-about)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running the Project](#running-the-project)
- [Example Output](#example-output)
- [Cool Features](#cool-features)
- [Expected Results](#expected-results)
- [What Could Be Better?](#what-could-be-better)
- [Who Made This?](#who-made-this)

## 🎯 What’s This About?

This project is all about predicting which syndrome a set of embeddings belongs to. Here’s what we do:

- **Pre-process the data** (flatten hierarchical structures).
- **Explore the dataset** (check class distributions and stats).
- **Visualize** it using **t-SNE** (so it looks nice and pretty).
- **Classify** using **KNN** (with Euclidean & Cosine distances).
- **Evaluate performance** using **F1-Score, AUC, and Top-k Accuracy**.

## 📂 Project Structure

```bash
ML-Junior-Practical-Test/
├── data/
│   └── mini_gm_public_v0.1.p   # Dataset in pickle format (hierarchical)
├── src/
│   ├── main.py                 # The boss script that runs everything
│   ├── dataPreprocessing.py     # Loads and flattens dataset
│   ├── dataExploration.py       # EDA (Exploratory Data Analysis)
│   ├── visualization.py         # Handles t-SNE and visualizations
│   ├── knnClassification.py     # Runs KNN and cross-validation
│   ├── metricsUtils.py          # Calculates Top-k, AUC, etc.
├── requirements.txt             # Libraries you need
├── README.md                    # This file!
├── ML_Report.pdf                # Detailed report with methodology & results
└── Interpretation_Answers.pdf   # Answers to interpretation questions
```

## 🛠 Getting Started

1. Clone this repo:

   ```bash
   git clone https://github.com/your-username/ML-Junior-Practical-Test.git
   cd ML-Junior-Practical-Test
   ```

2. Set up a virtual environment (recommended):

   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # Windows
   ```

3. Install all dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Project

1. Navigate to the `src/` folder:

   ```bash
   cd src
   ```

2. Run the main script:

   ```bash
   python main.py
   ```

You’ll see some stats, cool graphs, and performance metrics pop up in your terminal. 🎉

## 📊 Example Output

```bash
First rows of the DataFrame:
   syndromeId  subjectId  imageId      embeddingVector
0   300000082        595     3543  [-0.03718, 1.74148, ...]
1   300000082       2638     1633  [2.4250, 0.1799, ...]
...

Shape of DataFrame: (1116, 4)
X shape: (1116, 320) | y shape: (1116,)

    [CLASS DISTRIBUTION BAR CHART]

t-SNE Visualization: [Displayed in a separate window]

Euclidean Results
+----+-----------+
|  k | F1-Score  |
|----+-----------|
|  1 | 0.6307    |
|  2 | 0.5875    |
|  ..| ...       |
| 15 | 0.7346    |
+----+-----------+

Cosine Results
+----+-----------+
|  k | F1-Score  |
|----+-----------|
|  1 | 0.6755    |
|  7 | 0.7794    |
| ...| ...       |
+----+-----------+

Best k (Euclidean): 15
Best k (Cosine): 7

Top-5 Accuracy (Euclidean): 0.9688
Top-5 Accuracy (Cosine): 0.9420
```

## 🔥 Cool Features

### 📥 Data Handling
- Loads dataset from a pickle file.
- Converts it into a DataFrame with columns (`syndromeId`, `subjectId`, `imageId`, `embeddingVector`).

### 📊 Data Exploration
- Displays dataset statistics.
- Plots a class distribution bar chart.

### 🎨 Visualization (t-SNE)
- Reduces embedding dimensions to 2D.
- Plots a scatter graph colored by syndrome.

### 🤖 Classification (KNN)
- Performs 10-Fold Cross-Validation for `k` values from 1 to 15.
- Compares **Euclidean** and **Cosine** distances.

### 📈 Evaluation Metrics
- **F1-Score** per fold.
- **Top-k Accuracy** (like top-5 accuracy).
- **Multi-class AUC** (One-vs-Rest comparison).
- **Optional ROC Curves** for visualization.

## 🚀 Expected Results

- Find the best `k` value for each distance metric.
- Generate class distribution charts, t-SNE plots, and (optional) ROC Curves.
- Print performance comparison tables for Euclidean vs. Cosine distance.

## 🤔 What Could Be Better?

- **Normalize Embeddings**: Pre-process embeddings for better results.
- **Try Other Models**: Experiment with SVM, Random Forest, or neural networks.
- **Balance the Dataset**: Use oversampling/undersampling for imbalanced classes.
- **Automate Everything**: Add a CI/CD pipeline, containerize with **Docker**, or expose as a **Flask/FastAPI API**.
- **Improve Testing**: Write unit tests inside the `tests/` folder.

## ✨ Who Made This?

Author: [Leonardo Mucci](https://github.com/LeoMucci)

