# -*- coding: utf-8 -*-
"""
Created on Sun May  6 12:28:24 2018

@author: QLR
"""

from reading_credit_data import reading_original_data
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
    
import sklearn.metrics as metrics

def adaboost_model(data_x,data_y):
    
    bdt = AdaBoostClassifier(DecisionTreeClassifier(class_weight="balanced",max_depth=20, min_samples_split=10, min_samples_leaf=4),
                         algorithm="SAMME",
                         n_estimators=300, learning_rate=0.8)
    bdt.fit(data_x,data_y)
    
    return bdt
    
def main():
    X_train, X_test, Y_train, Y_test=reading_original_data()
    #data_x_decom=feature_selection.pca(X_train)
    
    bdt=adaboost_model(X_train,Y_train)
    y_pred=bdt.predict(X_test)
    print ("training score "+str(bdt.score(X_train,Y_train)))
    print ("test score "+ str(bdt.score(X_test,Y_test)))
    print ("f1 score "+str(metrics.f1_score(Y_test, y_pred)))
    print ("recall rate score "+str(metrics.recall_score(Y_test, y_pred)))
    print ("accuracy score "+str(metrics.accuracy_score(Y_test, y_pred)))
    #print (metrics.auc(y_pred,Y_test))

main()    