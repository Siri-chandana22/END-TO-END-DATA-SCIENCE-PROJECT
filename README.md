# END-TO-END-DATA-SCIENCE-PROJECT

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: YARLAGADDA SIRI CHANDANA

*INTERN ID*: CT4MDR3097

*DOMAIN*: DATA SCIENCE

*DURATION*: 4 MONTHS

*MENTOR*: NEELA SANTOSH

# Customer Churn Prediction — End-to-End Data Science Project

It contains a complete end-to-end Customer Churn Prediction project built using Python and popular data science libraries. The objective of this project is to analyze customer behavior and predict whether a customer is likely to churn (leave the service) or continue using it. Predicting churn is extremely valuable for businesses because retaining existing customers is often more cost-effective than acquiring new ones. By identifying customers at risk, companies can take proactive actions such as targeted offers, improved support, or personalized engagement strategies.

## Overview of the Pipeline

The churn prediction pipeline is implemented in multiple structured steps, beginning with loading the dataset and ending with saving the trained model and making a sample prediction. The pipeline is modular, easy to understand, and can be adapted to similar classification problems with minimal changes.

### 1. Loading the Dataset

The first step involves loading the customer churn dataset directly from an online GitHub repository using pandas.read_csv(). This ensures that the project remains reproducible and eliminates the need for manual dataset downloads. After loading, the code prints confirmation messages along with the first five rows of the dataset. This preview helps in understanding the structure of the data, feature names, and sample values. Additionally, dataset information such as column data types and memory usage is displayed, followed by a missing value check to identify incomplete records.

### 2. Data Understanding and Inspection

After loading the dataset, the script performs basic data understanding operations. These include printing dataset information, identifying null values, and examining feature types. This step helps in identifying unnecessary columns, incorrect data types, and potential preprocessing requirements. The churn column represents the target variable, while all other columns are treated as input features used to predict customer behavior.

### 3. Data Preprocessing

Several preprocessing operations are applied to clean and prepare the dataset for machine learning. The customerID column is removed since it does not contribute to prediction. The TotalCharges column is converted into numeric format, handling invalid or blank values using coercion. Rows containing missing values are then removed to ensure data consistency. After cleaning, categorical features are converted into numerical format using one-hot encoding through pandas.get_dummies(). The drop_first=True parameter is used to prevent multicollinearity by avoiding redundant columns. After preprocessing, the dataset becomes fully numeric and suitable for training machine learning models.

### 4. Feature and Target Selection

The processed dataset is divided into features (X) and target variable (y). The churn column is selected as the target, while the remaining columns form the feature set. The target variable is explicitly converted into integer format to ensure compatibility with the model and visualization steps. This separation prepares the dataset for model training.

### 5. Train-Test Split

The dataset is split into training and testing sets using train_test_split() with an 80-20 ratio. The training set is used to train the model, while the testing set is used to evaluate performance on unseen data. A fixed random state is used to ensure reproducibility of results.

### 6. Feature Scaling

Feature scaling is performed using StandardScaler. The scaler is fitted on training data and applied to both training and test sets. Scaling ensures that all numerical features have similar ranges, which improves model performance and convergence. This step is particularly important for algorithms such as logistic regression that are sensitive to feature magnitude.

### 7. Model Building

A Logistic Regression classifier is used to build the churn prediction model. The model is trained using the scaled training dataset. Logistic regression is a widely used classification algorithm that predicts the probability of binary outcomes. The max_iter parameter is increased to ensure proper convergence during training.

### 8. Model Evaluation

After training, predictions are generated for the test dataset. The model is evaluated using accuracy score, classification report, and confusion matrix. Accuracy provides overall correctness, while precision, recall, and F1-score give detailed insights into prediction performance. The confusion matrix shows the number of correct and incorrect predictions for both churn and non-churn classes.

### 9. Visualization

A histogram is plotted using matplotlib to visualize churn distribution. The plot shows the count of customers who churned versus those who did not churn. This visualization helps in understanding class balance and dataset characteristics.

### 10. Saving the Model

The trained model and scaler are saved using pickle. Saving both ensures that the same preprocessing steps can be applied during future predictions. This makes the pipeline deployment-ready and reusable without retraining.

### 11. Sample Prediction

Finally, a sample prediction is performed using a single row from the dataset. The sample is scaled using the saved scaler and passed to the trained model. The output indicates whether the customer is predicted to churn or not. This demonstrates how the model can be used in real-world scenarios.

## Key Features and Advantages

### Automated Data Preprocessing

It automatically cleans the dataset by removing unnecessary columns, fixing data types, and handling missing values.

### Categorical Encoding using One-Hot Encoding

Text values like gender or contract type are converted into numbers so the machine learning model can understand them.

### Feature Scaling using StandardScaler

All numeric values are scaled to a similar range. This helps the model learn better and improves prediction performance.

### Logistic Regression Model Training

A Logistic Regression algorithm is used to train the model and learn patterns from customer data to predict churn.

### Comprehensive Model Evaluation Metrics

The model performance is checked using accuracy, classification report, and confusion matrix to see how well it predicts.

## Conclusion

This Customer Churn Prediction project demonstrates a complete machine learning pipeline that transforms raw customer data into actionable predictions. The workflow integrates data cleaning, preprocessing, feature scaling, model training, evaluation, visualization, and model saving into a single automated script. This project serves as a strong foundation for real-world business analytics, customer retention strategies, and deployment-ready machine learning solutions.

## Outputs

<img width="884" height="354" alt="Image" src="https://github.com/user-attachments/assets/5ce251fe-3683-4cdb-ab67-a562ebf195d0" />

<img width="966" height="327" alt="Image" src="https://github.com/user-attachments/assets/bb4738ce-2890-4911-b04b-d787649d56bf" />

<img width="990" height="364" alt="Image" src="https://github.com/user-attachments/assets/dd07cf82-24dc-4387-9c5d-09963f6b07b5" />

<img width="868" height="367" alt="Image" src="https://github.com/user-attachments/assets/d6b30eb1-5851-4065-a349-9ea96e3afb01" />

<img width="570" height="367" alt="Image" src="https://github.com/user-attachments/assets/3774bf43-c44b-4824-af0d-b270dc2862e3" />

<img width="857" height="327" alt="Image" src="https://github.com/user-attachments/assets/e5c5cd7c-e736-43c3-896d-f5fcc965886c" />

<img width="458" height="82" alt="Image" src="https://github.com/user-attachments/assets/9ef130ac-be18-4a58-b5a9-b92e605b0680" />

![Image](https://github.com/user-attachments/assets/e6076893-4772-4be7-9545-d638ea5f34fb)

<img width="699" height="76" alt="Image" src="https://github.com/user-attachments/assets/37af51c7-38fc-4e73-9ceb-f11b1bad5b18" />
