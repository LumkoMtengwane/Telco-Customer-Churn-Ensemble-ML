# Telco-Customer-Churn-Ensemble-ML
End-to-end ML project that predicts customer churn using a stacked ensemble model combining Logistic Regression, Random Forest, and XGBoost. The project addresses class imbalance with SMOTE, performs hyperparameter optimization using RandomizedSearchCV, and evaluates performance using recall-focused metrics, ROC, and Precision–Recall curves.

📌 Overview
Customer churn is a critical business problem where companies aim to identify customers likely to discontinue their services. This project builds an end-to-end churn prediction system using advanced machine learning techniques, focusing on high recall for churned customers, which is essential in real-world retention strategies.
The final model is a stacked ensemble combining Logistic Regression, Random Forest, and XGBoost, optimized using RandomizedSearchCV and evaluated on an unseen test set.

🎯 Objectives
Predict customer churn accurately
Handle severe class imbalance
Maximize recall for churned customers
Build a robust and generalizable ML pipeline

🧾 Dataset
Source: IBM Telco Customer Churn Dataset (Kaggle)
Samples: 7,043 customers
Features: Demographic, service usage, and billing information
Target: Churn (Yes / No)

🔧 Data Preprocessing
Dropped irrelevant identifiers
Label-encoded categorical variables
Standardized continuous features
Checked for outliers using Z-score
Handled class imbalance using SMOTE
Train-test split with stratification

⚙️ Models Used
Logistic Regression
Random Forest Classifier
XGBoost Classifier
These models were combined using a StackingClassifier with stratified cross-validation to reduce overfitting and improve generalization.

🔍 Hyperparameter Optimization
Method: RandomizedSearchCV
Cross-validation: Stratified 5-Fold
Optimization Metric: Recall
Efficient tuning across all base learners in the ensemble

📈 Model Evaluation
Evaluated on an unseen test set using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
ROC Curve & AUC
Precision–Recall Curve
Feature importance analysis (Random Forest)

🔑 Final Test Performance
Recall (Churn): ~0.63
F1-score (Churn): ~0.61
Accuracy: ~0.79
These results demonstrate strong performance on an imbalanced dataset, prioritizing the correct identification of churned customers.

📊 Visualizations
Churn distribution
Correlation heatmap
Confusion matrix
ROC curve
Precision–Recall curve
Feature importance bar chart

🚀 How to Run
                    
    pip install -r requirements.txt

Run the Jupyter notebook:

    jupyter notebook Ensemble_Churn_Predictor.ipynb

🔮 Future Improvements
Replace Label Encoding with One-Hot Encoding
Integrate preprocessing and SMOTE into a Pipeline
Threshold tuning to further improve recall
Model deployment using Streamlit or FastAPI
Cost-sensitive learning

👤 Author
Lumko Mtengwane
Data Science | Machine Learning | Analytics
