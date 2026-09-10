# Loan Approval Analysis & Prediction

## Project Overview

This project uses a synthetic loan approval dataset containing personal, employment, credit, and financial information to predict whether a loan application will be approved or denied.

The analysis covers the complete machine learning workflow, from data cleaning and preprocessing through to classification model development and evaluation. Logistic Regression and K-Nearest Neighbours (KNN) models were developed and assessed using accuracy and F1-score.

## Dataset

The original dataset contains 20,000 loan application records and 36 attributes, including:

* Applicant demographics and employment status
* Annual and monthly income
* Credit score and credit history
* Loan amount and duration
* Existing debt and debt-to-income ratios
* Payment history and previous loan defaults
* Savings, assets and liabilities
* Home ownership and loan purpose
* Loan approval outcome

After preprocessing, **19,900 records and 50 features** were retained for modelling.

## Data Cleaning & Preprocessing

The dataset was examined for missing values and categorical attributes before modelling.

Key preprocessing steps included:

* Removed `RiskScore` because more than 50% of its values were missing.
* Removed 100 records with missing `MaritalStatus`, following the specified treatment for categorical missing values.
* Imputed missing `Age` values using the column mean.
* Removed `ApplicationDate` because every date was unique and therefore did not provide useful categorical information under the task requirements.
* Applied one-hot encoding to categorical variables including employment status, education level, marital status, home ownership and loan purpose.
* Converted encoded Boolean values into binary numerical variables.

## Modelling Approach

### Logistic Regression

A Logistic Regression model was trained using an 80/20 train-test split.

The baseline model achieved:

* **Test Accuracy:** 89.25%
* **Test F1-score:** 75.98%
* **Training Accuracy:** 88.73%
* **Training F1-score:** 75.01%

The small difference between training and testing performance indicated consistent model performance with no clear evidence of overfitting.

### Recursive Feature Elimination

RFE was applied to identify a smaller set of features while maintaining predictive performance.

The selected reduced-feature Logistic Regression model achieved approximately:

* **Test Accuracy:** 89.75%
* **Test F1-score:** 77.63%

The feature-selection analysis showed that reducing the number of features could maintain or slightly improve predictive performance compared with the baseline model.

### K-Nearest Neighbours

KNN classification was also implemented and evaluated across different values of K. Different distance measures were explored to assess their effect on model performance.

The 1-NN model achieved approximately:

* **Test Accuracy:** 85.9%
* **Test F1-score:** 69.8%

The difference between training and testing performance demonstrated the tendency of 1-NN to overfit the training data.

## Evaluation

Accuracy and F1-score were used to evaluate the classification models.

Accuracy measures the proportion of correctly classified loan applications, while F1-score provides a balance between precision and recall for the approved-loan class.

Using both metrics provides a more informative assessment than accuracy alone when evaluating loan approval predictions.

## Key Takeaways

* Logistic Regression provided strong and relatively consistent predictive performance.
* Feature selection through RFE slightly improved the model's test performance while reducing the feature set.
* 1-NN achieved lower test performance and showed signs of overfitting.
* Model evaluation using both accuracy and F1-score provided a more balanced view of classification performance.

## Tools & Techniques

**Python:**
`pandas` · `NumPy` · `Matplotlib` · `scikit-learn`

**Techniques:**
Data cleaning · Missing-value treatment · One-hot encoding · Train/test splitting · Logistic Regression · Recursive Feature Elimination (RFE) · KNN · Cross-validation · Grid Search · Accuracy · F1-score

## Conclusion

This project demonstrates an end-to-end machine learning workflow for loan approval classification. The results show that Logistic Regression performed strongly on the dataset, while RFE provided a more compact feature set with slightly improved predictive performance. KNN was also explored across different configurations, highlighting the importance of model selection and evaluation when working with classification problems.
