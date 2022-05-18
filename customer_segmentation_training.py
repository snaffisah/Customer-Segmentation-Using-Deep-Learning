# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:33:53 2022

@author: snaff
"""

import os
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


DATASET_TRAIN = os.path.join(os.getcwd(),'Data', 'train.csv')
LOG_PATH = os.path.join(os.getcwd(),'Log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'Saved_path', 'model.h5')
 
#%% EDA
# Step 1) Import data
train = pd.read_csv(DATASET_TRAIN)

train.info() 
# from info, we found that there is missing data for Ever_Married, Graduated, 
# Profession, Work_experience, Family_size, Var_1. Also, there are total of 
# 7 columns in Strings datatype. Total dataset is 8068 with 11 column

train.head()

# Step 2) Data Cleaning

# check for duplicate
train[train.duplicated()]

# Check NaN value for all column
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(train.isna())
plt.show()
plt.figure()

train.isna().sum()

# Replce with fillna method ffill and bfill
train['Ever_Married'].fillna(method='ffill',inplace=True)
train['Profession'].fillna(method='ffill',inplace=True)
train['Graduated'].fillna(method='bfill',inplace=True)
train['Work_Experience'].fillna(method='bfill',inplace=True)
train['Family_Size'].fillna(method='bfill',inplace=True)
train['Var_1'].fillna(method='bfill',inplace=True)

# Verify again on NaN value
train.isna().sum()

# Convert strings into integer using label encoder
le = LabelEncoder()
train['Gender'] = le.fit_transform(train['Gender'])
# 0=Female,1=Male
train['Ever_Married'] = le.fit_transform(train['Ever_Married'])
# 0=Yes,1=No
train['Graduated'] = le.fit_transform(train['Graduated'])
# 0=Yes,1=No
train['Profession'] = le.fit_transform(train['Profession'])
# 0=Artist,1=Doctor,2=Engineer,3=Entertaiment,4=Executive,5=Healtcare,
# 6=Homemaker,7=Lawyer
train['Spending_Score'] = le.fit_transform(train['Spending_Score'])
# 0=Average,1=High,2=Low
train['Var_1'] = le.fit_transform(train['Var_1'])
# 0=Cat_1,1=Cat_2,2=Cat_3,3=Cat_4,4=Cat_5,5=Cat_6,6=Cat_7
train['Segmentation'] = le.fit_transform(train['Segmentation'])
# 0=A, 1=B, 2=C, 3=D

# Step 4) Feature selection
# Plot correlation heatmap
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(train.corr(), annot=True, cmap=plt.cm.Reds, linewidths=.5, ax=ax)
plt.show()
plt.figure()
# Highly correlation Ever_Married,Age, Proffession
# Low correlation ID


x = train.drop(labels=['ID','Work_Experience','Segmentation'], axis=1) # features
 # ID has low corr, Work_Experience has a lot of nan value
y = train['Segmentation'] # target

# Step 5) Data Pre-processing

# Min Max Scaler
mms_scaler = MinMaxScaler()
x = mms_scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, 
                                                    random_state=0)

# Check the accuracy using machine learning
# Standard scaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Logistic regression method
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# prediction
y_predict = classifier.predict(x_test)

# Result of prediction model
cm = confusion_matrix(y_test, y_predict)
class_report = classification_report(y_test, y_predict)
score = accuracy_score(y_test, y_predict)

# Print result
print(cm)
print(class_report)
print("The model accuracy is:" ,score*100,)

#%% Callback  for DL

log_dir = os.path.join(LOG_PATH, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

# tensorboard callback
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# early stopping callback
early_stopping_callback = EarlyStopping(monitor='loss', patience=10)

#%% DL model

# Sequential

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='adam', loss='mse', metrics='mse')
model.summary()

hist = model.fit(x_train, y_train, epochs=100, 
#                 batch_size=128,
                 validation_data=(x_test,y_test),
                 callbacks=[tensorboard_callback, early_stopping_callback])

# Model  compile loss = mse, mae, metrics = mse
hist.history.keys()
training_loss = hist.history['loss']
training_acc = hist.history['mse']
validation_loss = hist.history['val_loss']
validation_acc = hist.history['val_mse']

# Losses during training
plt.figure()
plt.plot(training_loss) # during training
plt.plot(validation_loss) # validation loss
plt.title('training loss and validation loss')
plt.xlabel('epoch')
plt.ylabel('Cross entropy loss')
plt.legend(['training loss', 'validation loss'])
plt.show()

plt.figure()
plt.plot(training_acc) # during training
plt.plot(validation_acc) # validation loss
plt.title('training mse and validation mse')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend(['training mse', 'validation mse'])
plt.show()

#%% Model Deployment
model.save(MODEL_SAVE_PATH)