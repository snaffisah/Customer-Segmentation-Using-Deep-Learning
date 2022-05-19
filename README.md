# Customer_Segmentation_Deep_Learning
(Still need a lot of improvement! Please read 1st)

## Description
Objective: Create a classifier model to group customer's data using deep learning and return result in a new csv file

* Model training - Deep learning
* Method: Sequential
* Module: Tensorflow

In this analysis, dataset used from https://www.kaggle.com/datasets/abisheksudarshan/customer-segmentation

### About The Dataset:
There is 2 datasets used in this analysis
* train.csv (8067 entries data with 11 columns)
* test.csv (2627 entries data with 11 columns)

Our target is the 11th columns, named Segmentation. Train dataset will be used to create the model dan test dataset will be used to run the deployment and classified the entries data into 4 segmentation (A,B,C,D)

### EDA:
During EDA steps, 
Few columns have missing value:
<p align="center">
  <img width="440" src="https://github.com/snaffisah/Customer-Segmentation-Using-Deep-Learning/blob/main/Image/missing%20data.png">
</p>

Few columns in String datatype :
<p align="center">
  <img width="640" src="https://github.com/snaffisah/Customer-Segmentation-Using-Deep-Learning/blob/main/Image/before_encode.JPG">
</p>

Correlation between the data:
<p align="center">
  <img width="540" src="https://github.com/snaffisah/Customer-Segmentation-Using-Deep-Learning/blob/main/Image/train_corr.png">
</p>

In this analysis, missing value were replaced using fillna method (ffill and bfill), string datatype converted using label decoder and columns 'ID', 'Work_Experience', 'Segmentation' were dropped from the feature selection.

### Prediction accuracy using Machine Learning:
By using logistic regression method, percentace of accuracy achived only 50%

NOTE: Trying to improve the accuracy using few other method/feature selection, but sill not able to get more that this. Please let me know if you know the best way. TQ =)

### Deep Learning
Sequential model with 3 dense layers and 1 output were used in this analysis:
<p align="center">
  <img width="440" src="https://github.com/snaffisah/Customer-Segmentation-Using-Deep-Learning/blob/main/Image/model%20summary.JPG">
</p>

Model was trained until epoch 11 with loss: 1.6185, mse: 1.6185, val_loss: 1.5911, val_mse: 1.5911 and a STRAIGHT LINE GRAPH were produced =(
<p align="center">
  <img width="440" src="https://github.com/snaffisah/Customer-Segmentation-Using-Deep-Learning/blob/main/Image/train_val%20loss.png">
</p>

<p align="center">
  <img width="440" src="https://github.com/snaffisah/Customer-Segmentation-Using-Deep-Learning/blob/main/Image/train_val%20mse.png">
</p>

NOTE: Trying to restart spyder, but output still same. Probably I do mistake with the Input. Please do check for me. TQ

### Result
Managed to deploy the model and a new csv file named 'New_Customer_Segmentation.csv' was auto saved in current directory. Yeyyyyy.
<p align="center">
  <img width="440" src="https://github.com/snaffisah/Customer-Segmentation-Using-Deep-Learning/blob/main/Image/result_csv.JPG">
</p>

NOTE: Yes, I know result seems unlogical. A lots of improvement is needed in the coding. 

## Future Improvement
- Try different approach to get higher accuracy
- Check the input of Deep learning Model
- Classes and functions should be applied on this coding

Enjoy!

