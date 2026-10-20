# Women-s-Diabetes-Risk-Prediction-System


# Women's Diabetes Risk Prediction System

# Objective
The goal of this project is to develop a web-based application that predicts the risk of diabetes in women using machine learning. Diabetes is a major public health concern globally, especially in developing regions like South Africa, where women face distinct risks shaped by biological, lifestyle and socio-economic factors. The application uses easily obtainable health indicators such as glucose, BMI, age and blood pressure to estimate a user's diabetes risk, store their prediction history, and present the model's performance in an interactive dashboard.

The system combines a **Random Forest classifier implemented from scratch** in Python, a **Flask REST API** back end, and a **JavaScript single-page front end**.

 This is an academic project and a screening aid only. It is not a medical diagnosis tool.

## Key Features

### Risk Prediction
- **Health Input Form:** Users enter eight health indicators: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Age and Diabetes Pedigree Function (DPF).
- **Risk Percentage:** The model returns a diabetes risk percentage based on how many of the 300 decision trees vote "diabetes".
- **Risk Category:** A risk of 40% or above is classified as **HIGH RISK**, and below 40% as **LOW RISK**. The threshold was lowered from 50% to favour recall, because missing a diabetic patient is more costly than a false alarm in a screening context.
- **Personalised Recommendations:** Health advice is shown based on the user's risk category.

### Family History (Diabetes Pedigree Function)
- **Simplified DPF Calculator:** Users answer Yes/No questions about diabetic relatives, and the system calculates a pedigree score automatically.
- **Weighted by Genetic Closeness:** Parent and sibling (0.5), grandparent (0.25), cousin (0.125).

### User Management
- **Registration and Login:** Users create accounts with their name, email and password.
- **Secure Passwords:** Passwords are stored as salted hashes, never in plain text.
- **Session Handling:** Users stay logged in through a signed session cookie until they log out.

### Prediction History and Profile
- **History Table:** Every prediction is saved per user and displayed with its date, inputs, risk percentage and category.
- **Profile Statistics:** Shows total predictions and the number of high-risk and low-risk results.
- **CSV Export:** Users can download their prediction history as a CSV file.

### Model Performance Dashboard
- **Metrics Cards:** Displays Accuracy, Precision, Recall and F1 Score.
- **Confusion Matrix:** Shows true/false positives and negatives with an interpretation.
- **Feature Importance Chart:** A bar chart showing which health indicators influence predictions most.
- **Retrain Model:** Users can retrain and save the model directly from the interface.

### User Interface
- **Login Interface:** A two-panel login and registration screen.
- **Dashboard Interface:** Tab-based navigation between Prediction, Performance, History and Profile without page reloads.
- **Results Modal:** An animated risk gauge, a colour-coded risk bar and recommendations.
- **Responsive Design:** The layout adapts to desktop, tablet and mobile screens.

## Technical Details

### Machine Learning Pipeline
- **Dataset:** Pima Indian Diabetes Dataset (768 female patients, 8 features).
- **Data Splitting:** A custom stratified 80/20 train/test split that preserves the original class ratio in both sets.
- **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique) is applied to the **training set only**, to avoid data leakage into the test set.
- **Decision Tree (from scratch):** Splits are chosen by maximising **information gain** based on entropy. Trees stop growing at the maximum depth, when a node is pure, or when there are too few samples.
- **Random Forest (from scratch):** 300 decision trees, maximum depth of 15, each trained on a **bootstrap sample** (sampling with replacement). Predictions are aggregated through tree voting.
- **Feature Importance:** Calculated from sample-weighted information gain at each split, aggregated and normalised across all trees.
- **Model Persistence:** The trained model and its metadata (metrics, feature names, parameters) are saved with joblib and loaded at start-up.

### Results
Evaluated on a hold-out test set of 152 patients:
Accuracy: 78.29%
Precision: 67.24%
Recall: 73.58%
F1 Score: 70.27%


**Most important features:** Glucose (43.71%), BMI (19.50%), Age (12.84%), Diabetes Pedigree (8.34%).

### System Architecture
- **Front End:** HTML, CSS and vanilla JavaScript. It communicates with the back end using the Fetch API, sending and receiving JSON.
- **Back End:** A Flask REST API that handles routing, input validation, session-based authentication and model serving.
- **Database:** SQLite stores user accounts, prediction history and the original dataset.
- **Separation of Concerns:** The front end handles display, the API handles communication and validation, and the model handles prediction. Each layer can be changed independently.

### Prediction Flow
1. The user clicks **Predict Diabetes Risk**.
2. JavaScript validates the inputs and sends a `POST` request to `/api/predict`.
3. Flask checks the session and validates the data.
4. The Random Forest calculates the risk percentage and category.
5. The prediction is saved to the database.
6. A JSON response is returned and displayed in the results modal.


### Data Structures and Tools
- **NumPy arrays** for feature matrices, bootstrap sampling and tree computations.
- **Pandas DataFrames** for loading the dataset and reading prediction history.
- **Tree nodes (recursive structure)** to represent decision splits and leaf predictions.
- **Counter (dictionary)** for majority voting and class probability calculation.
- **Chart.js** for the feature importance visualisation.


## Conclusion
This application provides a complete tool for predicting and monitoring diabetes risk in women using a machine learning approach. By combining a from-scratch Random Forest model with a secure, user-friendly web interface, it supports early awareness and informed health decisions, while giving transparent insight into how the model performs and which factors matter most.
