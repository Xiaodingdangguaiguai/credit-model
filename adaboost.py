# -*- coding: utf-8 -*-
"""
Created on Sun May  6 12:28:24 2018

@author: QLR
"""

from reading_credit_data import reading_original_data
from reading_credit_data import feature_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
    
from sklearn.externals import joblib

def adaboost_model(data_x,data_y):
    
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=300, learning_rate=0.8)
    bdt.fit(data_x,data_y)
    
    return bdt
    
def main():
    X_train, X_test, Y_train, Y_test=reading_original_data()
    #data_x_decom=feature_selection.pca(X_train)
    
    bdt=adaboost_model(X_train,Y_train)
    bdt.score(X_train,Y_train)
    bdt.score(X_test,Y_test)    
    