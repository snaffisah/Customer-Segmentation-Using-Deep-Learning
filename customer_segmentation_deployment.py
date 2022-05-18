# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:33:55 2022

@author: snaff
"""

import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

DATASET_TEST = os.path.join(os.getcwd(),'Data', 'new_customers.csv')
LOG_PATH = os.path.join(os.getcwd(), 'Log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'Saved_path', 'model.h5')

#%%
model = load_model(MODEL_SAVE_PATH)
model.summary()

#%% Deploy
test = pd.read_csv(DATASET_TEST)

test.info() 
# from info, we found that there is missing data for Ever_Married, Graduated, 
# Profession, Work_experience, Family_size, Var_1. Also, there are total of 
# 7 columns in Strings datatype. Total dataset is 8068 with 11 column

test.head()

# Step 2) Data Cleaning

# check for duplicate
test[test.duplicated()]

# Check NaN value for all column
test.isna().sum()

# Replce with fillna method ffill and bfill
test['Ever_Married'].fillna(method='ffill',inplace=True)
test['Profession'].fillna(method='ffill',inplace=True)
test['Graduated'].fillna(method='bfill',inplace=True)
test['Work_Experience'].fillna(method='bfill',inplace=True)
test['Family_Size'].fillna(method='bfill',inplace=True)
test['Var_1'].fillna(method='bfill',inplace=True)

# Verify again on NaN value
test.isna().sum()

# Convert strings into integer using label encoder
le = LabelEncoder()
test['Gender'] = le.fit_transform(test['Gender'])
# 0=Female,1=Male
test['Ever_Married'] = le.fit_transform(test['Ever_Married'])
# 0=Yes,1=No
test['Graduated'] = le.fit_transform(test['Graduated'])
# 0=Yes,1=No
test['Profession'] = le.fit_transform(test['Profession'])
# 0=Artist,1=Doctor,2=Engineer,3=Entertaiment,4=Executive,5=Healtcare,
# 6=Homemaker,7=Lawyer
test['Spending_Score'] = le.fit_transform(test['Spending_Score'])
# 0=Average,1=High,2=Low
test['Var_1'] = le.fit_transform(test['Var_1'])
# 0=Cat_1,1=Cat_2,2=Cat_3,3=Cat_4,4=Cat_5,5=Cat_6,6=Cat_7

#train['Segmentation'] = le.fit_transform(train['Segmentation'])
# 0=A, 1=B, 2=C, 3=D

x = test.drop(labels=['Segmentation'], axis=1) # features 
y = test['Segmentation'] # target

mms_scaler = MinMaxScaler()
x = mms_scaler.fit_transform(x)

#%% model prediction

y_predicted = [] 

for test in x:
    y_predicted.append(model.predict(np.expand_dims(test,axis=0)))
   
y_predicted = np.array(y_predicted)

newy = y_predicted.reshape(1)

Segmentation_dict = {0:'A', 1:'B', 2:'C', 3:'D'}

y_new = (pd.Series(newy)).map(Segmentation_dict)
y_new1 = list(y_new)

# Combine column
combine_column = x + y_new1
print(combine_column)

# Convert to csv
combine_column.to_csv("New_Customer_Segmentation", index=False)