# bank_marketing
Bank Marketing Prediction App with XGBoost and Streamlit

Refer to the Streamlit app here for the interface: https://andreaslukita7-bankmarketing.streamlit.app/

This repository contains a Streamlit application that predicts whether a client will subscribe to a term deposit based on various features using the bank marketing dataset.

Dataset

The bank marketing dataset is sourced from the Kaggle website (Some slight variations of dataset are also available on UCI Machine Learning repository). It contains details like age, job, marital status, and other attributes for clients and their response to the bank's marketing campaigns. Link to dataset: https://www.kaggle.com/datasets/yufengsui/portuguese-bank-marketing-data-set

Features

1. Data Exploration: Understand the distribution of various features and their relationship with the target variable.
2. Model: The application uses an XGBoost classifier to predict the likelihood of a client subscribing to a term deposit.
3. Interpretability: Understand the importance of features and how they influence the model's predictions using SHAP values, ICE plots, and more.
Interactive Visualizations: Utilize Streamlit's capabilities for interactive plots and model explanations.

Jupyter Notebook

The Notebooks found on this repo contain initial work on this dataset before deploying on Streamlit. XGBoost Model is tuned using Hyperopt and the original Trials object has been converted into a CSV file to compress the file.

Installation

1. Clone this repository.
2. Navigate to the project directory and create a virtual environment.
3. Install the required packages using `pip install -r requirements.txt`.
4. Run the Streamlit app using `streamlit run app.py`.

Usage

After launching the app, you can:

1. Input client details to get predictions on their likelihood to subscribe to a term deposit.
2. Explore various visualizations for model interpretability.
3. View and understand decision trees from the XGBoost model.

Contributions

Contributions to enhance features, visualizations, or any other suggestions are welcome!