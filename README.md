# Bank Marketing Classification Project

## 📌 Project Overview

This repository contains a complete **machine learning classification project** that predicts whether a bank customer will subscribe to a **term deposit** based on marketing campaign data. The project walks through the full ML pipeline—from data understanding and preprocessing to dimensionality reduction and model comparison.

The work is implemented in a single Jupyter Notebook and is suitable for academic submission, portfolio demonstration, and beginner-to-intermediate ML practice.

---

## 🎯 Objectives

* Understand and explore the Bank Marketing dataset
* Perform **Exploratory Data Analysis (EDA)**
* Preprocess numerical and categorical features
* Handle class imbalance and feature scaling
* Apply **Principal Component Analysis (PCA)**
* Train and compare multiple classification models
* Evaluate models using standard performance metrics

---

## 🧠 Machine Learning Models Used

* Logistic Regression
* Naive Bayes (GaussianNB)
* Decision Tree Classifier
* Random Forest Classifier
* Neural Network (MLP Classifier)

---

## 🗂️ Dataset Information

* **Dataset Name:** Bank Marketing Dataset
* **Target Variable:** `deposit` (Yes / No)
* **Problem Type:** Binary Classification
* **Source:** UCI Machine Learning Repository
* **Features Include:**

  * Age, job, marital status, education
  * Balance, loan, housing
  * Contact type, campaign duration, previous outcomes

---

## ⚙️ Technologies & Libraries

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / Google Colab

---

## 🔄 Project Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Cleaning and Encoding
4. Feature Scaling
5. Dimensionality Reduction using PCA
6. Model Training
7. Model Evaluation and Comparison

---

## 📊 Evaluation Metrics

* Accuracy Score
* Confusion Matrix
* Precision, Recall, F1-Score (Classification Report)

📦 Bank-Marketing-ML
┣ 📜 finalProject.ipynb
┣ 📜 README.md
┣ 📂 data/
┃ ┗ 📜 bank.csv
┗ 📂 ScreenShots/
┗ 📂 EDA/
┣ 📸 AgeDistribution.png
┣ 📸 OriginalBalanceDistribution.png
┣ 📸 LogTransformedBalance.png
┣ 📸 ageVsBalanca.png
┣ 📸 balanceVsDeposit.png
┣ 📸 contactVsDeposit.png
┣ 📸 jobTypeVsDeposit.png
┗ 📸 ModelPerformanceComparison.png

### ▶ Option 1: Run on Google Colab

1. Upload `finalProject.ipynb` to Google Colab
2. Upload the dataset (`bank.csv`) to Google Drive
3. Mount Google Drive in Colab
4. Update dataset path in the notebook:

```python
file_path = '/content/drive/MyDrive/dataset/bank.csv'
```

5. Run all cells

### ▶ Option 2: Run Locally

1. Clone the repository

```bash
git clone https://github.com/yourusername/Bank-Marketing-ML.git
```

2. Install required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

3. Open the notebook using Jupyter
4. Update dataset path if needed
5. Run all cells

---

## 📌 Results Summary

* PCA reduced dimensionality while preserving most of the variance
* Ensemble models (Random Forest) performed better than basic classifiers
* Neural Network showed strong performance but required more tuning
* The comparison highlights the trade-off between simplicity and accuracy

---

## 🏁 Conclusion

This project demonstrates a **complete end-to-end machine learning workflow** for a real-world classification problem. It emphasizes the importance of preprocessing, dimensionality reduction, and proper model evaluation when building reliable ML solutions.

