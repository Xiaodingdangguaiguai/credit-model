# -*- coding: utf-8 -*-
"""
Created on Sun May  6 12:28:24 2018

@author: QLR
"""

from reading_credit_data import reading_original_data

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import  sklearn.metrics as metrics


def lr_model(data_x,data_y):
    
    lr = LogisticRegressionCV( random_state=0,class_weight="balanced",max_iter=1000,solver="sag",penalty="l2")
    lr.fit(data_x,data_y)

    return lr
    
def main():
    X_train, X_test, Y_train, Y_test=reading_original_data()
    #data_x_decom=feature_selection.pca(X_train)
    
    lr=lr_model(X_train,Y_train)
    y_pred=lr.predict(X_test)
    print ("training score "+str(lr.score(X_train,Y_train)))
    print ("test score "+ str(lr.score(X_test,Y_test)))
    print ("f1 score "+str(metrics.f1_score(Y_test, y_pred)))
    print ("recall rate score "+str(metrics.recall_score(Y_test, y_pred)))
    print ("accuracy score "+str(metrics.accuracy_score(Y_test, y_pred)))
   
main()    