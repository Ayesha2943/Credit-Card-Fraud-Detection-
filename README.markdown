# Fraud Detection System

## Overview
This project develops a **machine learning-based fraud detection system** to identify fraudulent financial transactions while minimizing false positives. Built in Python using a Random Forest Classifier, the system processes transaction data (e.g., amount, timestamp, user/merchant details) to distinguish between legitimate and fraudulent activities. The project includes data preprocessing, model training, evaluation, and analysis of misclassifications, making it a practical example of applying machine learning to real-world problems like credit card fraud detection.

## Features
- **Data Preprocessing**: Handles missing values, extracts time-based features (e.g., hour, day of week), encodes categorical variables (user/merchant IDs), and scales numerical features.
- **Machine Learning Model**: Uses a Random Forest Classifier with balanced class weights to address the imbalanced nature of fraud data (fraud cases are rare).
- **Evaluation**: Assesses performance with precision, recall, F1-score, and a confusion matrix, focusing on minimizing false positives (incorrectly flagged legitimate transactions).
- **Misclassification Analysis**: Identifies and explains false positives and false negatives to understand model errors.
- **Visualizations**: Includes feature importance plots, confusion matrix, and precision-recall curve for intuitive insights.
- **Modularity**: Code is adaptable for custom datasets with similar transaction features.

## Dataset
The project uses a simulated dataset with 10,000 transactions, including:
- `user_id`, `merchant_id`: Identifiers for users and merchants.
- `amount`: Transaction amount.
- `timestamp`: Date and time of the transaction.
- `is_fraud`: Binary label (0 for legitimate, 1 for fraudulent, with 2% fraud prevalence).

To use your own dataset, upload a CSV file and update the `pd.read_csv` path in the code.

## Technologies Used
- **Python Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib.
- **Environment**: Google Colab (cloud-based Jupyter Notebook).
- **Model**: Random Forest Classifier.

## How It Works
1. **Data Preparation**: Cleans and transforms transaction data (e.g., extracts hour from timestamp, scales amounts).
2. **Model Training**: Trains a Random Forest model on 70% of the data, using balanced weights to handle rare fraud cases.
3. **Evaluation**: Tests on 30% of the data, reporting precision, recall, and F1-score, with a focus on low false positives.
4. **Analysis**: Visualizes feature importance (e.g., amount, hour) and analyzes misclassified transactions.
5. **Model Saving**: Saves the trained model for future use.

## Results
- **Performance**: Achieves high precision and recall for fraud detection, with detailed metrics in the classification report.
- **Key Features**: Identifies critical attributes like transaction amount and time of day for detecting fraud.
- **Error Analysis**: Explains false positives (e.g., large legitimate transactions flagged) and false negatives (missed frauds).

## How to Run
1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Upload your dataset (if using a custom CSV) to Colab’s file system.
3. Run the notebook cells sequentially.
4. View outputs: classification metrics, confusion matrix, feature importance, and misclassification details.
5. Download the saved model (`fraud_detection_model.pkl`) for reuse.

## Future Improvements
- Add more features (e.g., transaction frequency, geolocation).
- Experiment with other models (e.g., XGBoost, neural networks).
- Implement oversampling techniques (e.g., SMOTE) for better handling of imbalanced data.

## Installation
No local installation is required for Google Colab. For local use (e.g., Jupyter Notebook):
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```


## Acknowledgments
Built as a beginner-friendly machine learning project to demonstrate fraud detection concepts, inspired by real-world financial applications.
