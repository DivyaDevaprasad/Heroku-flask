# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:46:13 2020

@author: Divya
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('hiring.csv')
print(data)

data.columns = ["experience","test_score","interview_score","Salary"]

#Handing missing values
#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 
                 'six':6, 'seven':7, 'eight':8,'nine':9, 'ten':10, 
                 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

data['experience'].fillna(0, inplace=True)
data['test_score'].fillna(data['test_score'].mean(), inplace=True)
data["experience"] = data["experience"].apply(lambda x: convert_to_int(x))

X = data.iloc[:, :3]
y = data.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
model = LinearRegression()

#Fitting model with trainig data
model.fit(X, y)


# Saving model to disk
import pickle
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
