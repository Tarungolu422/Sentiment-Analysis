# 🍽️ Restaurant Review Sentiment Analysis

## 🧠 Project Overview
This project performs **sentiment analysis** on restaurant reviews to classify customer feedback as **positive or negative**.  
It uses **TF-IDF vectorization** for text-to-numeric conversion and multiple **machine learning classifiers** with **hyperparameter tuning (GridSearchCV)** for model optimization.  
A **Streamlit frontend** is also developed for real-time user interaction and sentiment prediction.

---

## 📂 Dataset
- **Name:** `Restaurant_Reviews.tsv`  
- **Source:** Restaurant customer feedback dataset  
- **Format:** Tab-separated file (`.tsv`)  
- **Columns:**  
  - `Review` → Customer review text  
  - `Liked` → Target label (1 = Positive, 0 = Negative)

---

## ⚙️ Key Features
✅ Text preprocessing with **NLTK** (tokenization, lemmatization, stopword removal)  
✅ Feature extraction using **TF-IDF Vectorizer**  
✅ Training with **multiple ML classifiers**  
✅ **Hyperparameter tuning** using GridSearchCV  
✅ **AUC, ROC, Bias–Variance** analysis  
✅ **Visualization** of model performance  
✅ **Streamlit frontend** for live sentiment prediction  
✅ Models and vectorizer saved using **Pickle**  

---

## 🧩 Machine Learning Models Used
1. Logistic Regression  
2. Naive Bayes  
3. Support Vector Machine (SVM)  
4. Random Forest  
5. Decision Tree  
6. K-Nearest Neighbors (KNN)  
7. Gradient Boosting  
8. XGBoost  
9. LightGBM  

Each model undergoes **GridSearchCV** for best hyperparameter tuning.

---

## 🚨 Problem Encountered: Underfitting
Initially, the models suffered from **underfitting**, resulting in low accuracy and poor generalization.  
To fix this issue:
- Increased **TF-IDF vectorizer features** (`max_features=2000`)  
- Performed **hyperparameter tuning** for all models  
- Evaluated **bias–variance tradeoff** to ensure better balance  

After optimization, model performance significantly improved, achieving higher test accuracy and stable ROC curves.

---

## 📊 Evaluation Metrics
Each model is evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- AUC Score  
- Bias  
- Variance  

Visualizations:
- Accuracy comparison bar chart  
- ROC curve comparison  

---

## 🧠 Best Model
After tuning and evaluation, the top-performing model is selected automatically based on **highest test accuracy**.  
Both the **best model** and **TF-IDF vectorizer** are saved using pickle for later use in the Streamlit frontend.

---

## 🏆 Model Comparison Results

| Rank | Model | Best Params | Train Acc | Test Acc | Precision | Recall | F1 Score | AUC | Bias | Variance |
|------|-------|-------------|-----------|-----------|------------|---------|-----------|------|--------|-----------|
| 1 | SVM | `{'C': 10, 'kernel': 'rbf'}` | 0.9988 | 0.9750 | 0.9594 | 0.9895 | 0.9742 | 0.9978 | 0.0012 | 0.0238 |
| 2 | Random Forest | `{'max_depth': None, 'n_estimators': 200}` | 0.9988 | 0.9750 | 0.9689 | 0.9791 | 0.9740 | 0.9861 | 0.0012 | 0.0238 |
| 3 | Logistic Regression | `{'C': 10, 'solver': 'lbfgs'}` | 0.9944 | 0.9675 | 0.9450 | 0.9895 | 0.9668 | 0.9943 | 0.0056 | 0.0269 |
| 4 | Decision Tree | `{'criterion': 'entropy', 'max_depth': None}` | 0.9988 | 0.9600 | 0.9397 | 0.9791 | 0.9590 | 0.9608 | 0.0012 | 0.0388 |
| 5 | KNN | `{'n_neighbors': 5, 'weights': 'distance'}` | 0.9988 | 0.9400 | 0.9034 | 0.9791 | 0.9397 | 0.9755 | 0.0012 | 0.0588 |
| 6 | Naive Bayes | `{'alpha': 0.1}` | 0.9750 | 0.9300 | 0.9137 | 0.9424 | 0.9278 | 0.9842 | 0.0250 | 0.0450 |
| 7 | Gradient Boosting | `{'learning_rate': 0.2, 'n_estimators': 200}` | 0.9838 | 0.9100 | 0.9429 | 0.8639 | 0.9016 | 0.9786 | 0.0162 | 0.0737 |
| 8 | XGBoost | `{'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200}` | 0.9606 | 0.8875 | 0.9148 | 0.8429 | 0.8774 | 0.9537 | 0.0394 | 0.0731 |
| 9 | LightGBM | `{'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 200, 'num_leaves': 31}` | 0.8194 | 0.7575 | 0.8092 | 0.6440 | 0.7172 | 0.8313 | 0.1806 | 0.0619 |

---


