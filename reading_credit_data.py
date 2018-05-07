# -*- coding: utf-8 -*-
"""
Created on Sun May  6 12:28:24 2018

@author: QLR
"""

import numpy as np
import pandas
from sklearn import preprocessing
import sklearn

def discrete2nbr(class_name,DATASET):
    
    temp=set(DATASET[class_name])
    try:
        temp.remove(class_name)
    except KeyError:
        pass
    mapping={label:idx for idx,label in enumerate(temp)}

    return mapping
    

def reading_original_data():
    
    path="F:\\2_BWD\\Credit\\Lending club data\\clean_data\\"
    data_path="data_yes.xlsx"
    
    data=pandas.read_excel(path+data_path,convert_float=True)
    # dealing with missing data 
    for i in range(len(data["emp_length"])):
        if pandas.isnull(data['emp_length'][i])==True or data["emp_length"][i]=='n/a':
            data["emp_length"][i]=0.0
        if pandas.isnull(data['last_pymnt_d'][i])==True or data["last_pymnt_d"][i]=='n/a':
            data["last_pymnt_d"][i]=0.0

    data_x=data.iloc[:,0:-1]
    data_y=data['is_loan']
    
    # dealing with discrete non number data
    term_mapping=discrete2nbr(class_name="term",DATASET=data_x)
    data_x['term'] = data_x['term'].map(term_mapping)
    
    home_ownership=discrete2nbr(class_name="home_ownership",DATASET=data_x)
    data_x["home_ownership"] = data_x["home_ownership"].map(home_ownership)
    
    verification_status_mapping=discrete2nbr(class_name="verification_status",DATASET=data_x)
    data_x["verification_status"]=data_x['verification_status'].map(verification_status_mapping)
    
    purpose_mapping=discrete2nbr(class_name="purpose",DATASET=data_x)
    data_x['purpose']=data_x['purpose'].map(purpose_mapping)
    
    # transfer data to np.array
    data_x_clean=np.array(data_x.values.tolist())
    data_y_clean=np.array(data_y.values.tolist())
    # scale data into 0`1 or -1`1 with normalization

    
    min_max_scaler = preprocessing.MinMaxScaler()
    data_x_minmax = min_max_scaler.fit_transform(data_x_clean)
    X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split( data_x_minmax, data_y_clean, test_size=0.20, random_state=42)
    
    return X_train, X_test, Y_train, Y_test
    
    
from sklearn.decomposition import PCA 

class feature_selection():
    
    def  pca(data_x):
        
        data_x_decom=PCA(n_components="mle").fit_transform(data_x)
    
        return data_x_decom    

    
    