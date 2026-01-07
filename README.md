# AutoJudge
Problem Difficulty Predictor
# Overview
This Project aims to develop an automated system in order to classify the difficulty level of a programming task as easy, medium or hard and also a problem difficulty score to make comparisons within a class. 
# Dataset used  
https://github.com/AREEG94FAHAD/TaskComplexityEval-24 
# Approach and models used
Pre-processing: Handled missing values in text fields, concatenated all text fields into a single string, converted all text to lowercase and removed HTML and LATEX commands and symbols, normalized extra whitespaces. 

Feature Extraction: The combined problem text was vectorized using a TFIDF representation with unigrams and bigrams, capturing important terms and phrases relevant to problem difficulty. For manual features, text length, number of mathematical symbols and frequency of keywords was taken into account.

Models: 
RandomForest classifier for classification and RandomForest regressor for regression was used.

#Evaluation metrics

Classifier:

Accuracy: 50.91%

Precision: 0.5015

Recall: 0.5091

F1-score 0.4858

Regressor:

MAE: 1.706

RMSE: 2.034
# Steps to run the project locally
1.Clone the repository: 

git clone https://github.com/Ainesh-das/Autojudge.git

cd AutoJudge

2.Create and activate a virtual environment:

python -m venv venv

venv\Scripts\activate

3.Install necessary libraries from requirements.txt

pip install -r requirements.txt

4.Run the streamlit interface

streamlit run app.py

# Web Interface 
The web application is built using Streamlit, user has three text boxes to input problem,input and ouyput description. After clicking on predict the class and score are displayed. The models are loaded once and cached, ensuring fast, real-time predictions.
# Demo Video
...
# Personal Details
Ainesh Kumar Das (24114008)


