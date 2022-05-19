# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:33:55 2022

@author: snaff
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

DATASET_TEST = os.path.join(os.getcwd(),'Data', 'new_customers.csv')
LOG_PATH = os.path.join(os.getcwd(), 'Log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'Saved_path', 'model.h5')
NEW_CSV_FILE = os.path.join(os.getcwd(),'New_Customer_Segmentation.csv')

#%%
model = load_model(MODEL_SAVE_PATH)
model.summary()

#%% Deploy
df_test = pd.read_csv(DATASET_TEST)

df_test.info() 
# from info, we found that there is missing data for Ever_Married, Graduated, 
# Profession, Work_experience, Family_size, Var_1. Also, there are total of 
# 7 columns in Strings datatype. Total dataset is 8068 with 11 column
test=df_test

test.head()

# Step 2) Data Cleaning

# check for duplicate
test[test.duplicated()]

# Check NaN value for all column
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(test.isna())
plt.show()
plt.figure()

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

test1 = test
# Convert strings into integer using label encoder
le = LabelEncoder()
test1['Gender'] = le.fit_transform(test1['Gender'])
# 0=Female,1=Male
test1['Ever_Married'] = le.fit_transform(test1['Ever_Married'])
# 0=Yes,1=No
test1['Graduated'] = le.fit_transform(test1['Graduated'])
# 0=Yes,1=No
test1['Profession'] = le.fit_transform(test1['Profession'])
# 0=Artist,1=Doctor,2=Engineer,3=Entertaiment,4=Executive,5=Healtcare,
# 6=Homemaker,7=Lawyer
test1['Spending_Score'] = le.fit_transform(test1['Spending_Score'])
# 0=Average,1=High,2=Low
test1['Var_1'] = le.fit_transform(test1['Var_1'])
# 0=Cat_1,1=Cat_2,2=Cat_3,3=Cat_4,4=Cat_5,5=Cat_6,6=Cat_7

x = test1.drop(labels=['ID','Work_Experience','Segmentation'], axis=1) # features
y = test1['Segmentation'] # target

mms_scaler = MinMaxScaler()
x = mms_scaler.fit_transform(x)

#%% model prediction

y_predicted = [] 

for test in x:
    y_predicted.append(model.predict(np.expand_dims(test,axis=0)))
   
y_predicted = np.array(y_predicted)

newy = y_predicted.reshape(2627)

Segmentation_dict = {0:'A', 1:'B', 2:'C', 3:'D'}

y_new = (pd.Series(newy, name='Segmentation')).map(Segmentation_dict)

# Combine column
column_test = df_test.drop(labels=['Segmentation'],axis=1)
result = pd.concat([column_test,y_new], axis=1)
print(result)

result.info()

# Convert to csv
result.to_csv(NEW_CSV_FILE, index=False)
