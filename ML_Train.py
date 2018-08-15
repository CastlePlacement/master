# -*- coding: utf-8 -*-
# python3 "C:\Users\Zhixiong Cheng\Desktop\castle placement\___LIZZY_FRANK__Blendr.io_-_more_info_requested_on_Filemaker_pro_-_Amazon_machine_learning\ML_v2\ML_v2\ML_Train.py"

"""
Created on Thu Dec  7 14:15:46 2017

@author: Barbara
"""
print("Starting traing..../n")
# In[116]:
### Import data from Filemaker

# Import Python ODBC package
import pyodbc
# Import panadas to mangage data structure
import pandas as pd
import os
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

print("Finishing importing libraries..../n")
'''
# connect to filemaker
CONNECTION_STRING = "DSN=filemaker;UID=InternTemp;PWD=Castle0905"

connection = pyodbc.connect(CONNECTION_STRING)
cursor = connection.cursor()
# Get data from table MachineLearningTestData2
rows = cursor.execute("select * from MachineLearningTestData2")
data=rows.fetchall()
col_names = [column[0] for column in cursor.description]
connection.close()

# df is a pandas dataframe containing data in MachineLearningTestData2
df = pd.DataFrame.from_records(data,columns=col_names)

usedfield=['MLSize1', 'MLGeography1', 'MLStages1','MLIndustries1', 'MLFundType1', 'MLREType1', 'MLFinancialInvestorYN1','MLStrategicInvestorYN1', 'MLInterestedDeals1', 'MLCompany1','MLCreditType1', 'MLLendingType1', 'MLREProperties1']

print("Finishing importing training and testign data from filemaker..../n")
'''

# Load csv files for training:
import os
src = os.path.dirname(os.path.realpath(__file__))
os.chdir(src)

# In[129]:
### Run machine learning algorithm

### Import and Clean data ###

from sklearn import preprocessing
# import sklearn which is for machine learning packages
from sklearn.linear_model import LogisticRegression
# import the package to split the data into train an test
from sklearn.cross_validation import train_test_split
# import the package to get the summary of prediction
from sklearn.metrics import classification_report
# import the package to create confusion matrix
from sklearn.metrics import confusion_matrix

#------------------------------------------------------------------------------------------
# First for PE only:
Deal_data=pd.read_csv('PE.csv', sep=',',header=None)
# select the fields we need to train our model (from MLSize1 to MLInterestedDeals1)
X=Deal_data[list(range(13))]
# set y= "MLStatusCategory1"
y=Deal_data[13]

### Training ###

# Fit the data in Logistic model
# Split Train and test data into X_train y_train(70%) and X_test y_test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
# Build logistic model (Machine Learning Model)
LogReg = LogisticRegression(class_weight='balanced')
# Fit the model with data
LogReg.fit(X_train, y_train)

print("Finishing logistic regression..../n")

#print y_train.value_counts()
#print y_test.value_counts()


### Parameters Tuning ###

# #Simple K-Fold cross validation. 10 folds.
# from sklearn.model_selection import KFold # import KFold
# kf = KFold(n_splits=2) # Define the split - into 2 folds 
# kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
# KFold(n_splits=2, random_state=None, shuffle=False)

# for train_index, test_index in kf.split(X):
#  print(“TRAIN:”, train_index, “TEST:”, test_index)
#  X_train, X_test = X[train_index], X[test_index]
#  y_train, y_test = y[train_index], y[test_index]




# cv = cross_validation.KFold(len(X), n_folds=10, indices=False)

# results = []
# # "Error_function" can be replaced by the error function of your analysis
# for traincv, testcv in cv:
#         probas = LogReg.fit(X_train[traincv], y_train[traincv]).predict_proba(X_train[testcv])
#         results.append( Error_function )
# print "Results: " + str( np.array(results).mean() )




### Final Model and Predictive Results ###
y_pred = LogReg.predict(X_test)
# create confusion matrix from prediction and true test data
confusion_matrix1 = confusion_matrix(y_test, y_pred)

# Calculate four numbers 'False Positive', 'True Positive', 'True Negative', 'False Negative'
[FN,FP],[TN,TP] = confusion_matrix1

# Calculate the target accuracy 'TP/(TP+TN)' we need
target=float(TP)/(TP+TN)

print(classification_report(y_test, y_pred))
print(pd.DataFrame(confusion_matrix1,index=['False','True'],columns=['Negative','Positive']))
print('\n')
print('Target accuraty is {:0.3f}'.format(target))


# save the model to disk
import pickle

filename = 'finalized_model_PE.sav'
import os
#dir_fd = os.open('C:/Users/Zhixiong Cheng/Desktop/castle placement/___LIZZY_FRANK__Blendr.io_-_more_info_requested_on_Filemaker_pro_-_Amazon_machine_learning/ML_v2/ML_v2', os.O_RDONLY)
#src = 'C:/Users/Zhixiong Cheng/Desktop/castle placement/___LIZZY_FRANK__Blendr.io_-_more_info_requested_on_Filemaker_pro_-_Amazon_machine_learning/ML_v2/ML_v2'
#src = os.path.expanduser('~/Documents')
src = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(src, filename)
print(path)
with open(path, "wb") as f:
    #f.write("chcjcj".encode())
    pickle.dump(LogReg, f)






#pickle.dump(LogReg, open(filename, 'wb'), protocol=2)
#pickle.dump(LogReg, open(filename, 'wb'))
print("finish 1st\n")
#------------------------------------------------------------------------------------------
# Second for Debt only:
Deal_data=pd.read_csv('Debt.csv', sep=',',header=None)
# select the fields we need to train our model (from MLSize1 to MLInterestedDeals1)
X=Deal_data[list(range(13))]
# set y= "MLStatusCategory1"
y=Deal_data[13]

### Training ###

# Fit the data in Logistic model
# Split Train and test data into X_train y_train(70%) and X_test y_test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
# Build logistic model (Machine Learning Model)
LogReg = LogisticRegression(class_weight='balanced')
# Fit the model with data
LogReg.fit(X_train, y_train)

print("Finishing logistic regression..../n")

#print y_train.value_counts()
#print y_test.value_counts()


### Parameters Tuning ###

# #Simple K-Fold cross validation. 10 folds.
# from sklearn.model_selection import KFold # import KFold
# kf = KFold(n_splits=2) # Define the split - into 2 folds 
# kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
# KFold(n_splits=2, random_state=None, shuffle=False)

# for train_index, test_index in kf.split(X):
#  print(“TRAIN:”, train_index, “TEST:”, test_index)
#  X_train, X_test = X[train_index], X[test_index]
#  y_train, y_test = y[train_index], y[test_index]




# cv = cross_validation.KFold(len(X), n_folds=10, indices=False)

# results = []
# # "Error_function" can be replaced by the error function of your analysis
# for traincv, testcv in cv:
#         probas = LogReg.fit(X_train[traincv], y_train[traincv]).predict_proba(X_train[testcv])
#         results.append( Error_function )
# print "Results: " + str( np.array(results).mean() )




### Final Model and Predictive Results ###
y_pred = LogReg.predict(X_test)
# create confusion matrix from prediction and true test data
confusion_matrix1 = confusion_matrix(y_test, y_pred)

# Calculate four numbers 'False Positive', 'True Positive', 'True Negative', 'False Negative'
[FN,FP],[TN,TP] = confusion_matrix1

# Calculate the target accuracy 'TP/(TP+TN)' we need
target=float(TP)/(TP+TN)

print(classification_report(y_test, y_pred))
print(pd.DataFrame(confusion_matrix1,index=['False','True'],columns=['Negative','Positive']))
print('\n')
print('Target accuraty is {:0.3f}'.format(target))


# save the model to disk
import pickle

filename = 'finalized_model_Debt.sav'
import os
#dir_fd = os.open('C:/Users/Zhixiong Cheng/Desktop/castle placement/___LIZZY_FRANK__Blendr.io_-_more_info_requested_on_Filemaker_pro_-_Amazon_machine_learning/ML_v2/ML_v2', os.O_RDONLY)
#src = 'C:/Users/Zhixiong Cheng/Desktop/castle placement/___LIZZY_FRANK__Blendr.io_-_more_info_requested_on_Filemaker_pro_-_Amazon_machine_learning/ML_v2/ML_v2'
#src = os.path.expanduser('~/Documents')
src = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(src, filename)
print(path)
with open(path, "wb") as f:
    #f.write("chcjcj".encode())
    pickle.dump(LogReg, f)

print("finish 2nd\n")

#------------------------------------------------------------------------------------------
# Third for PEDebt only:
Deal_data=pd.read_csv('PEDebt.csv', sep=',',header=None)
# select the fields we need to train our model (from MLSize1 to MLInterestedDeals1)
X=Deal_data[list(range(13))]
# set y= "MLStatusCategory1"
y=Deal_data[13]

### Training ###

# Fit the data in Logistic model
# Split Train and test data into X_train y_train(70%) and X_test y_test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
# Build logistic model (Machine Learning Model)
LogReg = LogisticRegression(class_weight='balanced')
# Fit the model with data
LogReg.fit(X_train, y_train)

print("Finishing logistic regression..../n")

#print y_train.value_counts()
#print y_test.value_counts()


### Parameters Tuning ###

# #Simple K-Fold cross validation. 10 folds.
# from sklearn.model_selection import KFold # import KFold
# kf = KFold(n_splits=2) # Define the split - into 2 folds 
# kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
# KFold(n_splits=2, random_state=None, shuffle=False)

# for train_index, test_index in kf.split(X):
#  print(“TRAIN:”, train_index, “TEST:”, test_index)
#  X_train, X_test = X[train_index], X[test_index]
#  y_train, y_test = y[train_index], y[test_index]




# cv = cross_validation.KFold(len(X), n_folds=10, indices=False)

# results = []
# # "Error_function" can be replaced by the error function of your analysis
# for traincv, testcv in cv:
#         probas = LogReg.fit(X_train[traincv], y_train[traincv]).predict_proba(X_train[testcv])
#         results.append( Error_function )
# print "Results: " + str( np.array(results).mean() )




### Final Model and Predictive Results ###
y_pred = LogReg.predict(X_test)
# create confusion matrix from prediction and true test data
confusion_matrix1 = confusion_matrix(y_test, y_pred)

# Calculate four numbers 'False Positive', 'True Positive', 'True Negative', 'False Negative'
[FN,FP],[TN,TP] = confusion_matrix1

# Calculate the target accuracy 'TP/(TP+TN)' we need
target=float(TP)/(TP+TN)

print(classification_report(y_test, y_pred))
print(pd.DataFrame(confusion_matrix1,index=['False','True'],columns=['Negative','Positive']))
print('\n')
print('Target accuraty is {:0.3f}'.format(target))


# save the model to disk
import pickle

filename = 'finalized_model_PEDebt.sav'
import os
#dir_fd = os.open('C:/Users/Zhixiong Cheng/Desktop/castle placement/___LIZZY_FRANK__Blendr.io_-_more_info_requested_on_Filemaker_pro_-_Amazon_machine_learning/ML_v2/ML_v2', os.O_RDONLY)
#src = 'C:/Users/Zhixiong Cheng/Desktop/castle placement/___LIZZY_FRANK__Blendr.io_-_more_info_requested_on_Filemaker_pro_-_Amazon_machine_learning/ML_v2/ML_v2'
#src = os.path.expanduser('~/Documents')
src = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(src, filename)
print(path)
with open(path, "wb") as f:
    #f.write("chcjcj".encode())
    pickle.dump(LogReg, f)

print("Done training!..../n")


 
