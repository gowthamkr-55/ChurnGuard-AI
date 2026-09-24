# CHURNGUARD AI — Customer Churn Prediction

## 1. Project Overview

**CHURNGUARD AI** is a machine learning project designed to predict whether a customer is likely to churn based on their demographic information, service usage, contract details, billing information, and other customer-related features.

The project uses supervised machine learning classification techniques to identify customers who may be at higher risk of leaving a service. Along with the prediction, the system also generates a churn probability and converts it into a simple **LOW, MEDIUM, or HIGH risk level**.

The project was developed as part of the **MainCrafts SkillSprint Artificial Intelligence & Machine Learning challenge**.

---

## 2. Problem Statement

Customer churn occurs when an existing customer stops using a company's products or services.

For a business, identifying customers who may leave is important because early identification can allow the company to review the customer's situation and consider appropriate retention actions.

The objective of this project is to build a machine learning classification system that learns patterns from historical customer data and predicts whether a customer is likely to churn.

A false negative in this problem occurs when the model predicts that a customer will stay even though the customer actually churns. Therefore, evaluating the model using multiple classification metrics is important rather than relying only on accuracy.

---

## 3. Project Objectives

The main objectives of CHURNGUARD AI are:

* Understand the customer churn prediction problem.
* Explore and analyze customer data.
* Identify useful patterns related to customer churn.
* Preprocess numerical and categorical features.
* Train multiple classification models.
* Compare model performance using classification metrics.
* Improve the selected model by handling class imbalance.
* Generate churn probability for individual customers.
* Convert churn probability into a simple risk category.
* Save the trained model for later use.
* Provide a Streamlit-based customer risk prediction interface.

---

## 4. Dataset

The project uses the **Telco Customer Churn dataset**:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains customer information related to demographics, tenure, telephone services, internet services, additional services, contracts, billing, and churn status.

The notebook initially loads a dataset with **7,043 records and 21 columns**. During preprocessing, the dataset used for model development contains **7,021 usable records**.

### Target Variable

The target variable is:

**Churn**

It represents whether a customer left the service.

* `Yes` → Customer churned
* `No` → Customer did not churn

### Main Features

The project uses customer attributes such as:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

The `customerID` column is not used as a predictive feature.

The notebook confirms the original dataset contains 7,043 rows and 21 columns.

---

## 5. Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit

### Machine Learning Techniques

* Logistic Regression
* Random Forest Classifier
* One-Hot Encoding
* Standard Scaling
* Pipeline-based preprocessing
* Class-weight balancing

### Development Environment

* Jupyter Notebook / Google Colab
* Visual Studio Code for the Streamlit application

The project requirements include Streamlit, Pandas, Scikit-learn, and Joblib.

---

# 6. Project Workflow

The project follows the following machine learning workflow:

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Data Preprocessing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Improvement
   ↓
Final Model Selection
   ↓
Risk Prediction
   ↓
Model Saving
   ↓
Streamlit Deployment
```

---

# 7. Data Loading and Inspection

The dataset was loaded using Pandas and inspected to understand its structure, data types, dimensions, and available features.

The original dataset contains:

* **7,043 records**
* **21 columns**
* Numerical and categorical features
* `Churn` as the target variable

The notebook also checks missing values and data types before proceeding with model development.

---

# 8. Data Preprocessing

Data preprocessing was performed before training the machine learning models.

### 8.1 Removing the Customer Identifier

The `customerID` column identifies individual customers but does not provide meaningful predictive information for churn. Therefore, it was excluded from the model features.

### 8.2 Target Encoding

The `Churn` target was converted into a binary classification format:

```text
No  → 0
Yes → 1
```

### 8.3 Feature Types

The project separates the input features into:

**Numerical features**

* SeniorCitizen
* tenure
* MonthlyCharges
* TotalCharges

**Categorical features**

* gender
* Partner
* Dependents
* PhoneService
* MultipleLines
* InternetService
* OnlineSecurity
* OnlineBackup
* DeviceProtection
* TechSupport
* StreamingTV
* StreamingMovies
* Contract
* PaperlessBilling
* PaymentMethod

### 8.4 Numerical Feature Scaling

Numerical features are standardized using:

```text
StandardScaler
```

This places numerical variables on a comparable scale for Logistic Regression.

### 8.5 Categorical Feature Encoding

Categorical variables are converted into numerical representations using:

```text
OneHotEncoder(handle_unknown="ignore")
```

This allows the machine learning models to process categorical customer information.

The notebook implements these transformations together using a `ColumnTransformer`.

---

# 9. Train-Test Split

The dataset was divided into training and testing sets using an **80:20 split**.

```text
Training data: 5,616 records
Testing data: 1,405 records
```

The split uses:

* `test_size = 0.2`
* `random_state = 42`
* `stratify = y`

Stratification helps maintain the class distribution between the training and testing datasets.

---

# 10. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand patterns and relationships between customer characteristics and churn.

The project includes visualizations for:

### 10.1 Customer Churn Distribution

A count plot was used to observe the distribution of customers who churned and those who remained.

### 10.2 Churn by Contract Type

The project compares churn across:

* Month-to-month
* One year
* Two year

The analysis showed that **month-to-month contract customers had higher churn compared with customers on longer-term contracts**.

### 10.3 Tenure and Churn

The tenure distribution was analyzed using a histogram grouped by churn status.

The analysis found that **customers with lower tenure showed higher churn levels**.

### 10.4 Monthly Charges and Churn

Monthly charges were compared between churned and retained customers.

The analysis found that **higher monthly charges were associated with higher churn**.

These findings are part of the final results recorded in the notebook.

---

# 11. Machine Learning Models

Three model versions were developed and evaluated.

## 11.1 Logistic Regression

The first model was Logistic Regression.

The model was implemented inside a Scikit-learn Pipeline containing:

1. Preprocessing
2. Logistic Regression classifier

Configuration:

```text
max_iter = 1000
random_state = 42
```

### Initial Logistic Regression Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8021 |
| Precision | 0.6599 |
| Recall    | 0.5215 |
| F1 Score  | 0.5826 |

The confusion matrix was:

```text
[[933 100]
 [178 194]]
```

---

## 11.2 Random Forest

A Random Forest classifier was also trained for comparison.

Configuration:

```text
n_estimators = 100
random_state = 42
```

### Random Forest Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.7772 |
| Precision | 0.6073 |
| Recall    | 0.4489 |
| F1 Score  | 0.5162 |

Confusion matrix:

```text
[[925 108]
 [205 167]]
```

The notebook uses these results to compare Logistic Regression and Random Forest.

---

# 12. Model Improvement

The initial Logistic Regression model produced a relatively lower recall for the churn class.

To improve the model's ability to identify churn cases, **class weighting** was introduced.

The improved model uses:

```python
LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)
```

Using `class_weight="balanced"` gives additional consideration to the class distribution during model training.

### Before Improvement

```text
F1 Score = 0.5826
```

### After Improvement

```text
Accuracy  = 0.7416
Precision = 0.5079
Recall    = 0.7796
F1 Score  = 0.6151
```

The improvement increased the F1 score from approximately **0.5826 to 0.6151**, while substantially increasing recall for the churn class.

---

# 13. Final Model

The final selected model is:

**Improved Logistic Regression**

The final model combines:

```text
ColumnTransformer
      +
StandardScaler
      +
OneHotEncoder
      +
Logistic Regression
      +
class_weight="balanced"
```

### Final Model Performance

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 74.16% |
| Precision | 50.79% |
| Recall    | 77.96% |
| F1 Score  | 61.51% |

The final results in the notebook identify Improved Logistic Regression as the selected model.

---

# 14. Why the Improved Model Was Selected

The project focused on identifying customers who may churn.

The improved Logistic Regression model was selected after comparing the tested models and applying an improvement to the original Logistic Regression model.

The class-balanced version produced a higher recall for the churn class and improved the F1 score from the original Logistic Regression result.

Therefore, the final model used in the project is the **Improved Logistic Regression model**.

---

# 15. Customer Risk Prediction

The final model can generate:

1. Churn prediction
2. Churn probability
3. Risk level

The churn probability is converted into three risk categories.

### Risk Classification

| Churn Probability | Risk Level |
| ----------------: | ---------- |
|           `< 40%` | LOW        |
|      `40% – <70%` | MEDIUM     |
|           `≥ 70%` | HIGH       |

This risk-level system is implemented in the Streamlit application as well.

---

# 16. Example Prediction

A sample customer was passed through the final prediction function.

The model generated:

```text
Prediction:
Customer likely to stay

Churn Probability:
33.44%

Risk Level:
LOW
```

This demonstrates how the trained model can be used to evaluate an individual customer's churn risk.

---

# 17. Streamlit Application

As an advanced component of the project, a simple Streamlit web interface was developed.

The application is titled:

**CHURNGUARD AI — Customer Risk Predictor**

The application allows users to enter customer information through an interactive form.

Inputs include:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

The application loads the saved model and performs prediction using the same trained pipeline.

---

# 18. Model Saving

The trained model was saved using Joblib as:

```text
churn_guard_final_model.pkl
```

The saved model contains the trained preprocessing and classification pipeline so that new customer records can be processed consistently during prediction.

The Streamlit application loads this saved model using Joblib.

---

# 19. Business Insights

The analysis produced several useful observations:

### 1. Contract Type

Month-to-month contract customers showed higher churn compared with customers on longer-term contracts.

### 2. Customer Tenure

Customers with lower tenure showed higher churn levels.

### 3. Monthly Charges

Higher monthly charges were associated with higher churn.

These observations can help businesses identify customer groups that may require closer monitoring.

---

# 20. Business Application

CHURNGUARD AI can potentially be used as a decision-support system for customer retention.

Customers identified as higher-risk could be considered for actions such as:

* Customer service reviews
* Proactive communication
* Retention campaigns
* Service-plan reviews
* Targeted offers

The model should support business decision-making rather than automatically determining what action should be taken for a customer.

---

# 21. Limitations

The project has several limitations:

* The model is trained using a specific historical dataset.
* Customer behaviour can change over time.
* Model predictions are probabilistic and are not guarantees that an individual customer will churn.
* The final model has a precision of approximately 50.79%, meaning not every predicted churn case corresponds to an actual churn case.
* Model performance depends on the quality and relevance of the available customer data.
* The project does not include real-time customer behaviour data.
* Further validation would be required before using the model in a production business environment.

The notebook also notes that the model should be re-validated periodically as customer behaviour changes.

---

# 22. Future Improvements

Possible future improvements include:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Testing additional classification algorithms.
* Applying more advanced class-imbalance techniques.
* Feature engineering based on customer behaviour.
* Threshold optimization based on business requirements.
* Cross-validation for more robust evaluation.
* Adding ROC-AUC and Precision-Recall analysis.
* Adding model explainability using feature importance or SHAP.
* Monitoring model performance after deployment.
* Connecting the application to a database.
* Deploying the Streamlit application online.
* Adding customer-level prediction history and reporting.

---

# 23. Project Structure

```text
CHURNGUARD-AI/
│
├── ChurnGuard_AI_Gowtham_MainCrafts.ipynb
├── app.py
├── churn_guard_final_model.pkl
├── requirements.txt
└── README.md
```

### File Description

| File                                     | Description                                    |
| ---------------------------------------- | ---------------------------------------------- |
| `ChurnGuard_AI_Gowtham_MainCrafts.ipynb` | Complete machine learning workflow             |
| `app.py`                                 | Streamlit customer risk prediction application |
| `churn_guard_final_model.pkl`            | Saved trained machine learning pipeline        |
| `requirements.txt`                       | Python dependencies                            |
| `README.md`                              | Project documentation                          |

---

# 24. How to Run the Project

## Step 1 — Clone the Repository

```bash
git clone <your-repository-url>
cd CHURNGUARD-AI
```

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

The project requires:

```text
streamlit
pandas
scikit-learn
joblib
```

## Step 3 — Run the Streamlit Application

```bash
streamlit run app.py
```

The application requires `churn_guard_final_model.pkl` to be present in the same directory as the application.

---

# 25. Key Skills Demonstrated

This project demonstrates practical skills in:

* Python programming
* Pandas data manipulation
* NumPy numerical processing
* Exploratory Data Analysis
* Data visualization
* Data preprocessing
* Categorical encoding
* Feature scaling
* Classification
* Logistic Regression
* Random Forest
* Class imbalance handling
* Model evaluation
* Confusion matrix analysis
* Pipeline construction
* Model persistence using Joblib
* Streamlit application development
* Business interpretation of ML predictions

---

# 26. Final Results

## CHURNGUARD AI — FINAL RESULTS

**Dataset:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Number of records used:** 7,021

**Models tested:**

* Logistic Regression
* Random Forest
* Improved Logistic Regression

**Selected model:**

**Improved Logistic Regression**

### Final Performance

```text
Accuracy  : 0.7416
Precision : 0.5079
Recall    : 0.7796
F1 Score  : 0.6151
```

### Important Findings

1. Month-to-month contract customers showed higher churn compared with customers on longer-term contracts.
2. Customers with lower tenure showed higher churn levels.
3. Higher monthly charges were associated with higher churn.

---

# 27. Conclusion

CHURNGUARD AI demonstrates an end-to-end machine learning workflow for customer churn prediction.

The project begins with dataset inspection and exploratory analysis, followed by preprocessing of numerical and categorical features. Logistic Regression and Random Forest models were trained and evaluated using accuracy, precision, recall, F1 score, and confusion matrices.

An improved Logistic Regression model using balanced class weights was then developed to improve the identification of churn cases. The final model achieved an accuracy of **74.16%**, precision of **50.79%**, recall of **77.96%**, and F1 score of **61.51%** on the test set.

The project was further extended with a Streamlit application that allows users to enter customer information and receive a churn prediction, probability, and risk category.

Overall, CHURNGUARD AI demonstrates the complete process of converting customer data into a practical machine learning-based risk prediction system.
