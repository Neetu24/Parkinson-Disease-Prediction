# 🧠 Parkinson Disease Prediction using Machine Learning

## 📌 Objective

Develop a machine learning model that can predict whether a person has **Parkinson’s disease** based on biomedical voice features and health metrics.


## 🗃️ Dataset

* **Source**: [UCI Machine Learning Repository – Parkinson’s Disease Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
* **Instances**: 195
* **Attributes**: 23 biomedical voice measurements + `status` (target label)

> **Target Column**: `status` → 0 = healthy, 1 = Parkinson’s disease
> **Features**: Includes average vocal fundamental frequency, jitter, shimmer, HNR, RPDE, DFA, PPE, etc.


## 🚀 Project Goals

### ✅ Step 1: Importing Libraries and Dataset

* Libraries used:

  * `pandas`, `numpy` → Data handling
  * `matplotlib`, `seaborn` → Data visualization
  * `scikit-learn` → Preprocessing, models, metrics
  * `xgboost` → High-performance classifier
  * `imbalanced-learn` → Apply **SMOTE** for class imbalance


### 🔧 Step 2: Data Preprocessing

* Drop irrelevant columns like `name`
* Normalize features using `StandardScaler`
* Check and handle class imbalance using **SMOTE**
* Split dataset: 80% train / 20% test


### 📊 Step 3: Exploratory Data Analysis (EDA)

* Visualize feature distributions using histograms/boxplots
* Generate correlation heatmap to understand relationships
* Identify significant features (e.g., MDVP\:Fo(Hz), spread1, PPE)


### 🤖 Step 4: Model Training and Selection

Train the following models:

* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)
* XGBoost

> All models are evaluated based on:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC


### 📈 Step 5: Model Evaluation & Testing

* Compare model performance visually using ROC Curves
* Choose the best model (usually XGBoost or Random Forest)
* Test the model on unseen data


## 🧪 Tools & Libraries Used

| Tool                 | Description                             |
| -------------------- | --------------------------------------- |
| Python               | Programming language                    |
| Pandas / NumPy       | Data manipulation                       |
| Seaborn / Matplotlib | Data visualization                      |
| Scikit-learn         | Machine learning models & preprocessing |
| XGBoost              | Gradient boosting classifier            |
| Imbalanced-learn     | SMOTE for balancing dataset             |


## 📁 Project Structure

```
📦 parkinson-disease-prediction/
├── parkinson_prediction.py          # Main Python script
├── Parkinson_Prediction.ipynb       # Jupyter notebook (optional)
├── parkinsons.csv                   # Dataset file
├── requirements.txt                 # List of dependencies
└── README.md                        # Project documentation
```

## 📊 Sample Output

* **Best Model**: XGBoost
* **Accuracy**: \~92%
* **ROC-AUC Score**: 0.95+
* **Confusion Matrix & Classification Report** for all models


## ✅ Conclusion

* The project successfully builds a machine learning pipeline for **Parkinson’s disease prediction**.
* **XGBoost** and **Random Forest** perform best in terms of accuracy and ROC-AUC.
* This model can assist in **early diagnosis** using **non-invasive voice analysis**.
