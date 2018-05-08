# -*- coding: utf-8 -*-
"""
Created on Sun May  6 12:28:24 2018

@author: QLR
"""

from reading_credit_data import reading_original_data

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

def gbdt_model(data_x,data_y):
    
    gbm0 = GradientBoostingClassifier(random_state=10)
    gbm0.fit(data_x,data_y)
    y_pred = gbm0.predict(data_x)
    y_predprob = gbm0.predict_proba(data_x)[:,1]
    print ("Accuracy : %.4g" % metrics.accuracy_score(data_y, y_pred))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(data_y, y_predprob))
    

    
#    param_test1 =[] 
#    for i in range(20,81,10):
#        param_test1.append([{'n_estimators':i}])
#    gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
#                                  min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10), 
#                       param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
#    gsearch1.fit(data_x,data_y)
#    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

    return gbm0
    
def main():
    X_train, X_test, Y_train, Y_test=reading_original_data()
    #data_x_decom=feature_selection.pca(X_train)
    
    bdt=gbdt_model(X_train,Y_train)
    y_pred=bdt.predict(X_test)
    print ("training score "+str(bdt.score(X_train,Y_train)))
    print ("test score "+ str(bdt.score(X_test,Y_test)))
    print ("f1 score "+str(metrics.f1_score(Y_test, y_pred)))
    print ("recall rate score "+str(metrics.recall_score(Y_test, y_pred)))
    print ("accuracy score "+str(metrics.accuracy_score(Y_test, y_pred)))
 
main()    