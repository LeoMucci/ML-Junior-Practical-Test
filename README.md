# ML Junior Practical Test

Welcome to **ML Junior Practical Test**!

This project demonstrates a Machine Learning pipeline for syndrome classification (`syndromeId`) based on 320-dimensional embeddings. Below, you will find usage instructions, module descriptions, and tips for executing and understanding this project.

## 📌 Table of Contents

- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Execution Example](#execution-example)
- [Main Features](#main-features)
- [Expected Results](#expected-results)
- [Possible Improvements](#possible-improvements)
- [Credits and Contact](#credits-and-contact)

## 🎯 Overview

The main objective of this project is to predict which syndrome a set of embeddings belongs to. For this, we use:

- **Pre-Processing** (flatten hierarchical data).
- **Exploratory Analysis** (statistics and class distribution).
- **Visualization** (dimensionality reduction via t-SNE).
- **Classification** (KNN with Euclidean and Cosine distances).
- **Evaluation** (F1-Score, multi-class AUC, Top-k Accuracy).

## 📂 Folder Structure

```bash
ML-Junior-Practical-Test/
├── data/
│   └── mini_gm_public_v0.1.p   # Dataset in pickle format (hierarchical)
├── src/
│   ├── main.py                 # Main script orchestrating the pipeline
│   ├── dataPreprocessing.py     # Dataset loading and flattening
│   ├── dataExploration.py       # Exploratory Data Analysis (EDA)
│   ├── visualization.py         # Functions for t-SNE and other visualizations
│   ├── knnClassification.py     # Classification using KNN + cross-validation
│   ├── metricsUtils.py          # Metric implementations (Top-k, AUC, etc.)
├── requirements.txt             # Required libraries
├── README.md                    # This file
├── ML_Report.pdf                # Complete report (methodology, results)
└── Interpretation_Answers.pdf   # Answers to interpretation questions
```

## 🛠 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/ML-Junior-Practical-Test.git
   cd ML-Junior-Practical-Test
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. If you experience issues with NumPy, try:

   ```bash
   pip install --upgrade numpy
   ```

## 🚀 How to Run

1. Navigate to the `src/` folder:

   ```bash
   cd src
   ```

2. Run the main script:

   ```bash
   python main.py
   ```

During execution, statistics, graphs, and metrics will be displayed in the terminal.

## 📊 Execution Example

```bash
First rows of the DataFrame:
   syndromeId  subjectId  imageId      embeddingVector
0   300000082        595     3543  [-0.03718, 1.74148, ...]
1   300000082       2638     1633  [2.4250, 0.1799, ...]
...

Shape of DataFrame: (1116, 4)
X shape: (1116, 320) | y shape: (1116,)

Top-5 Accuracy (Euclidean): 0.9688
Top-5 Accuracy (Cosine): 0.9420
```

## 🔍 Main Features

### 📥 Data Loading and Pre-Processing

- Load dataset in pickle format.
- Convert to `DataFrame` with columns (`syndromeId`, `subjectId`, `imageId`, `embeddingVector`).

### 📊 Exploratory Data Analysis (EDA)

- Dataset statistics.
- Class distribution (bar chart).

### 🎨 Visualization (t-SNE)

- Dimensionality reduction to 2D.
- Scatter plot with clustering by `syndromeId`.

### 🤖 Classification (KNN)

- 10-Fold Cross-Validation for `k` values from 1 to 15.
- Comparison of **Euclidean** and **Cosine** distances.

### 📈 Metrics and Evaluation

- **F1-Score** average per fold.
- **Top-k Accuracy** (e.g., top-5).
- **Multi-class AUC** (One-vs-Rest).
- **Optional ROC Curves**.

## 📌 Expected Results

- Best `k` value for each distance metric.
- Class distribution charts, t-SNE, and (optional) ROC Curves.
- Comparison between distance metrics (tables in terminal output).

## 🚀 Possible Improvements

- **Normalization**: Normalize embeddings before KNN.
- **Other Models**: Test SVM, Random Forest, or neural networks.
- **Data Balancing**: Oversampling/undersampling for unbalanced classes.
- **Automation & Deployment**: Implement CI/CD pipeline, containerize with **Docker**, or expose API via **Flask/FastAPI**.
- **Unit Testing**: Implement tests in the `tests/` folder.

## 📞 Credits and Contact

Author: [Your Name](https://github.com/your-username)

Feel free to contribute, suggest improvements, or report issues! 🚀
