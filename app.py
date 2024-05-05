import numpy as np
import pandas as pd
import sklearn
import scipy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import streamlit as st

# Load data
credit_card_data = pd.read_csv('creditcard.csv')

# Seperate Legit and Fraud transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Undersampling Legit Data
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = new_dataset.drop(columns= 'Class', axis=1)
Y = new_dataset['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# train logistic regression model
log = LogisticRegression()
log.fit(X_train, Y_train)

# Evaluate model performance
train_acc = accuracy_score(log.predict(X_train),Y_train)
test_acc = accuracy_score(log.predict(X_test),Y_test)

# Web App 
st.title("Credit Card Fraud Detection Model")
input_df = st.text_input('Enter all required features values')
input_df_splitted = input_df.split(',')

submit = st.button("Submit")


if submit:
    features = np.asarray(input_df_splitted,dtype=np.float64)
    prediction = log.predict(features.reshape(1,-1))

    if prediction[0] == 0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fradulant Transaction")    