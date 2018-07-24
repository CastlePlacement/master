print("Starting to do prediction.../n")

# In[135]:
### Export data back to filemaker
import os
import pickle
#src = os.path.expanduser('~/Documents')
src = os.path.dirname(os.path.realpath(__file__))
#os.chdir('C:/Users/Zhixiong Cheng/Desktop/castle placement/___LIZZY_FRANK__Blendr.io_-_more_info_requested_on_Filemaker_pro_-_Amazon_machine_learning/ML_v2/ML_v2')
os.chdir(src)
#Import LogReg function from ML_train
# load the model from disk
print("Loading model..../n")
LogReg = pickle.load(open('finalized_model.sav', 'rb'))#, encoding="ISO-8859-1"
#pickle.dump(worddict, open('finalized_model.sav', 'rb'), protocol=2)

import pyodbc
# Import panadas to mangage data structure
import pandas as pd
import numpy as np


# read deal data:
Deal_data=pd.read_csv('deal.csv', sep=',',header=None)
Deal_data = Deal_data[list(range(14))]
#pd.concat([Deal_data]*5)
Deal_data = np.array(Deal_data)

# read contact data:
contact_data = pd.read_csv('contact.csv', sep=',',header=None)

# read all local deals:
local_deal=pd.read_csv('deal_all.csv', sep=',',header=None)

#local_deal = local_deal.drop(4, axis = 1)

# combine contact table and deal_all table:
temp_tab = contact_data.merge(local_deal, on=[0, 1, 2], how='left')
#temp_tab.astype(str).groupby(temp_tab[0], as_index=False).agg('\x0b'.join)
contact_full = temp_tab.astype(str).groupby(temp_tab[2], as_index=False).agg({0 : lambda x: x.iloc[0], 1 : lambda x: x.iloc[0], 
                                                               2 : lambda x: x.iloc[0], '3_x' : lambda x: x.iloc[0],
                                                               4 : lambda x: x.iloc[0], 5 : lambda x: x.iloc[0],
                                                               6 : lambda x: x.iloc[0], 7 : lambda x: x.iloc[0],
                                                               8 : lambda x: x.iloc[0], 9 : lambda x: x.iloc[0],
                                                               10 : lambda x: x.iloc[0], 11 : lambda x: x.iloc[0],
                                                               12 : lambda x: x.iloc[0], 13 : lambda x: x.iloc[0],
                                                               14 : lambda x: x.iloc[0], '3_y': '\x0b'.join})
contact_full.columns = list(range(16))

# Calculate the matrix
import numpy as np
# get comparision matrix:
compare_mat = np.zeros((len(contact_full), 13))
#print(compare_mat)
# read deal data:
Deal_data=pd.read_csv('deal.csv', sep=',',header=None)
Deal_data = Deal_data[list(range(14))]
#pd.concat([Deal_data]*5)
Deal_data = np.array(Deal_data)
#np.repeat(Deal_data, [2], axis=0)
Deal_data = np.repeat(Deal_data, [len(contact_data)], axis=0)
#contact_data = np.array(contact_data[list(range(17))])
#print(Deal_data)
#print(contact_full)
# contact_full starts from index 3, deal_data starts from index 1
#print(contact_full)
#print(contact_full[15][0])
#print(Deal_data[0])
#print(set(contact_full[0][3].split('\x0b')))
for row in range(len(compare_mat)):
    for col in range(13):
        try:
            #if contact_data[row][col] is not np.nan and Deal_data[row][col] is not np.nan:
            compare_mat[row][col] = bool(set(contact_full[col+3][row].split('\x0b')) & set(Deal_data[row][col+1].split('\x0b'))) -1 + 1
        except:
            #compare_mat[row][col] = np.nan
            compare_mat[row][col] = 0.5

compare_mat = pd.DataFrame(compare_mat)
compare_mat.columns = ['MLSize1', 'MLGeography1', 'MLStages1','MLIndustries1', 'MLCompany1','MLCreditType1', 'MLLendingType1', 'MLFundType1', 'MLREType1', 'MLREProperties1', 'MLFinancialInvestorYN1','MLStrategicInvestorYN1', 'MLInterestedDeals1']

# ML:
data = compare_mat

print("Computing Probs..../n")

# create DealStatusID and prediction which we want to export
all_pred=[round(x,2) for x in LogReg.predict_proba(data)[:,1]]
FirstN=contact_full[0].copy()
FirstN = FirstN.astype(str)#.replace(r"\'", "\"")
LastN=contact_full[1].copy()
LastN = LastN.astype(str)
Email=contact_full[2].astype(str)
DealID = pd.DataFrame(Deal_data)[0].astype('int32')

import re
for i in range(len(LastN)):
    LastN[i] = LastN[i].replace("\'", "\"")
for i in range(len(FirstN)):
    FirstN[i] = FirstN[i].replace("\'", "\"")
for i in range(len(Email)):
    Email[i] = Email[i].replace("\'", "\"")

    
print("Inserting result to table MachineLearningPredictionData3..../n")
CONNECTION_STRING = "DSN=filemaker;UID=InternTemp;PWD=Castle0905"
connection = pyodbc.connect(CONNECTION_STRING)
cursor = connection.cursor()
# Change the format of output data required by SQL
output_string=str(list(zip(DealID, FirstN,LastN,Email,all_pred))).strip('[]')#.replace("\'", "\"")

try:
    cursor.execute('delete from MachineLearningPredictionData3')
    # Export prediction into "MachineLearningPredictionData3"
    cursor.execute("insert into MachineLearningPredictionData3(DealID, FirstName, LastName, Email, MLPrediction1) values " + output_string)
finally:
    connection.commit()

    connection.close()

print("Finishing prediction!..../n")