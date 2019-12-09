# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 10:30:04 2019

@author: keert
"""

#
# Statistical Machine Learning – Fall 2019
# Purchase Capacity Prediction based on User demographics and Product information 
# Keerthiraj Nagaraj
# Electrical and Computer Engineering, University of Florida



# =============================================================================
# =============================================================================
# # Importing libraries
# =============================================================================
print('Importing libraries.......')


#basic libraries
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
import time

plt.style.use('ggplot')
warnings.filterwarnings("ignore")
# data processing libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Performance analysis libraries
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_curve

# =============================================================================


# =============================================================================
# =============================================================================
# # Defining necessary functions
# =============================================================================


def getCount(df, data_df, feat_name):
    
    group_df = data_df.groupby(feat_name)
    
    cnt_dict = {}
    for name, group in group_df:
        cnt_dict[name] = group.shape[0]

    cnt_list = []
    for index, row in df.iterrows():
        name = row[feat_name]
        cnt_list.append(cnt_dict.get(name, 0))
    
    return cnt_list

# =============================================================================

time_list = []

time_list.append(time.time())

# =============================================================================
# =============================================================================
# # Exporting Black Friday sales data
# =============================================================================
print('Exporting data.......')

data = pd.read_csv('Datasets/black friday sales/train.csv')

# Data head
data_head = data.head()

# Data Info
data_info = data.info()

# Data describe
data_describe = data.describe()
# =============================================================================

time_list.append(time.time())
# =============================================================================
# =============================================================================
# # Data Preprocessing
# =============================================================================
print('Data Preprocessing.......')

# finding columns with null values
data_null_sum = data.isnull().sum()

# removing columns with null values
data_without_na = data.drop(columns = ['Product_Category_2', 'Product_Category_3'])
data = data_without_na

data_unique_count = data.apply(lambda x: len(x.unique()))

# Dealing with categorical features

# 1. Gender - It has 2 levels
gender_vals = {'F':0, 'M':1}
data["Gender"] = data["Gender"].apply(lambda gen: gender_vals[gen])

# 2. Age group - It has 7 levels
age_vals = {'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6}
data["Age"] = data["Age"].apply(lambda ag: age_vals[ag])

# 3. City Category - it has 3 levels
city_vals = {'A':0, 'B':1, 'C':2}
data["City_Category"] = data["City_Category"].apply(lambda ci: city_vals[ci])


# 4. Stay in years has 5 levels
lencoder = LabelEncoder()

data['Stay_In_Current_City_Years'] = lencoder.fit_transform(data['Stay_In_Current_City_Years'])
data = pd.get_dummies(data, columns=['Stay_In_Current_City_Years'])


# Addtional features - Counts

data["User_ID_Count"]  = getCount(data, data, "User_ID")
data["Product_ID_Count"]  = getCount(data, data, "Product_ID")
data["Product_Category_1_Count"]  = getCount(data, data, "Product_Category_1")

# Removing User ID and Product ID
data = data.drop(columns = ['User_ID', 'Product_ID'])


# Input features and target names definition
in_features = data.columns.drop(['Purchase'])
target = 'Purchase'


# Training and testing split (random split)
random.seed = 0

train_id = random.sample(range(0,data.shape[0]), 440054)
test_id = list(set(np.arange(0,data.shape[0])) - set(train_id))

train_data = data.iloc[train_id, :]
test_data = data.iloc[test_id, :]


train_data["Purchase_level"] = train_data[target] > np.quantile(train_data[target], 0.75)
test_data["Purchase_level"] = test_data[target] > np.quantile(test_data[target], 0.75)

train_data["Purchase_level"] = train_data["Purchase_level"].apply(lambda x: int(x==True))
test_data["Purchase_level"] = test_data["Purchase_level"].apply(lambda x: int(x==True))


in_features = train_data.columns.drop(['Purchase_level', 'Purchase'])
target = 'Purchase_level'
# =============================================================================



time_list.append(time.time())
# =============================================================================
# =============================================================================
# # Regression Model Training
# =============================================================================
print('Classification Model Training.......')


# Multiple Linear Regression
print('Logistic Regression')

params_log = {'max_iter' : [50, 100, 200] }

log_reg = LogisticRegression(n_jobs=-1, solver='sag', random_state=0, class_weight= 'balanced')

log_grid = GridSearchCV(log_reg, params_log, verbose=2)
log_grid.fit(train_data[in_features], train_data[target])


log_test_predictions = log_grid.best_estimator_.predict(test_data[in_features])
log_test_predictions_score = log_grid.best_estimator_.predict_proba(test_data[in_features])[:,1]

f1_log = f1_score(test_data[target].values, log_test_predictions)

print("F1-score: ", f1_log) 
fpr_log, tpr_log, threshold = roc_curve(test_data[target].values, log_test_predictions_score)
plt.figure()
plt.plot(fpr_log, tpr_log)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LOG ROC curve')
plt.grid(True)
plt.savefig('log_roc.png', dpi = 600)


log_best = log_grid.best_params_

filename = 'model_class_LOG_gCV.sav'
pickle.dump(log_grid.best_estimator_, open(filename, 'wb'))


time_list.append(time.time())

# =============================================================================
# # Decision Tree Regression
# =============================================================================
print('Decision Tree Regression')

params_dt = {'max_depth': [5, 10, 20], 'min_samples_leaf': [50, 100]}


dec_trees = DecisionTreeClassifier(random_state=0)

td_grid = GridSearchCV(dec_trees, params_dt, verbose=2)
td_grid.fit(train_data[in_features], train_data[target])

#Predictions

td_test_predictions = td_grid.best_estimator_.predict(test_data[in_features])
td_test_predictions_score = td_grid.best_estimator_.predict_proba(test_data[in_features])[:,1]

f1_dt = f1_score(test_data[target].values, td_test_predictions)

print("F1-score: ",f1_dt) 
fpr_dt, tpr_dt, threshold = roc_curve(test_data[target].values, td_test_predictions_score)
plt.figure()
plt.plot(fpr_dt, tpr_dt)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DT ROC curve')
plt.grid(True)
plt.savefig('dt_roc.png', dpi = 600)


dt_best = td_grid.best_params_

filename = 'model_class_DT_gCV.sav'
pickle.dump(td_grid.best_estimator_, open(filename, 'wb'))


time_list.append(time.time())
# =============================================================================
# # Random Forest Regression
# =============================================================================
print('Random Forest Regression')


params_rf = {'n_estimators' : [5, 10], 'max_depth': [5, 10, 20]}


rf_model = RandomForestClassifier(n_jobs = -1, criterion='gini', 
                       min_samples_leaf=5, 
                       oob_score = True, 
                       random_state=0)


rf_grid = GridSearchCV(rf_model, params_rf, 
                       verbose=2)

rf_grid.fit(train_data[in_features], train_data[target])



#Predictions
rf_test_predictions = rf_grid.best_estimator_.predict(test_data[in_features])
rf_test_predictions_score = rf_grid.best_estimator_.predict_proba(test_data[in_features])[:,1]

f1_rf = f1_score(test_data[target].values, rf_test_predictions)

print("F1-score: ", f1_rf) 
fpr_rf, tpr_rf, threshold = roc_curve(test_data[target].values, rf_test_predictions_score)
plt.figure()
plt.plot(fpr_rf, tpr_rf)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RF ROC curve')
plt.grid(True)
plt.savefig('rf_roc.png', dpi = 600)


rf_best = rf_grid.best_params_

filename = 'model_class_RF_gCV.sav'
pickle.dump(rf_grid.best_estimator_, open(filename, 'wb'))




time_list.append(time.time())
# =============================================================================
# # XGBoost
# =============================================================================
print('Extreme Gradient Boosting Decision Tree Regression')

params_xgb = {'n_estimators': [500, 1000],   
              'max_depth': [5, 10, 20]}


xgb = XGBClassifier(nthread=-1, objective='binary:logistic',
                    booster='gbtree', 
                    tree_method='auto', 
                    learning_rate=0.05, 
                    random_state=0) 

xgb_grid = GridSearchCV(xgb, params_xgb,
                    verbose=2)

xgb_grid.fit(train_data[in_features], train_data[target])


#Predictions
xgb_test_predictions = xgb_grid.best_estimator_.predict(test_data[in_features])
xgb_test_predictions_score = xgb_grid.best_estimator_.predict_proba(test_data[in_features])[:,1]

f1_xgb = f1_score(test_data[target].values, xgb_test_predictions)

print("F1-score: ", f1_xgb) 
fpr_xgb, tpr_xgb, threshold = roc_curve(test_data[target].values, xgb_test_predictions_score)
plt.figure()
plt.plot(fpr_xgb, tpr_xgb)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGB ROC curve')
plt.grid(True)
plt.savefig('xgb_roc.png', dpi = 600)


xgb_best = xgb_grid.best_params_

filename = 'model_class_XGB_gCV.sav'
pickle.dump(xgb_grid.best_estimator_, open(filename, 'wb'))

# =============================================================================

time_list.append(time.time())
# =============================================================================
# Model prediction graphs
# =============================================================================


dec_tree_pred = td_test_predictions
rf_pred = rf_test_predictions



print('Model prediction graphs.......')


plt.figure()
plt.plot(fpr_log, tpr_log, label = 'LOG')
plt.plot(fpr_dt, tpr_dt, label = 'DT')
plt.plot(fpr_rf, tpr_rf, label = 'RF')
plt.plot(fpr_xgb, tpr_xgb, label = 'XGBoost')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves')
plt.grid(True)
plt.legend()
plt.savefig('roc_all_gCV_class.png', dpi = 600)



xtick_names = ['MLR', 'DT', 'RF', 'XGBoost']
x_tick_id = np.arange(len(xtick_names))


test_f1_all = [f1_log, f1_dt, f1_rf, f1_xgb]


plt.figure()
plt.plot(test_f1_all, '.', MarkerSize = 20)
plt.plot(test_f1_all, '-')
plt.xticks(x_tick_id, ('LOG', 'DT', 'RF', 'XGBoost'))
plt.xlabel('Classification algorithm')
plt.ylabel('F1-score')
plt.title('F1-score comparison')
plt.grid(True)
plt.legend()
plt.savefig('f1_all_gCV_class.png', dpi = 600)


y_test = test_data[target].values
y_test = y_test.reshape(len(y_test), 1)

log_test_predictions = log_test_predictions.reshape(len(log_test_predictions), 1)
dec_tree_pred = dec_tree_pred.reshape(len(dec_tree_pred), 1)
rf_pred = rf_pred.reshape(len(rf_pred), 1)
xgb_test_predictions = xgb_test_predictions.reshape(len(xgb_test_predictions), 1)

class_all_results = np.concatenate((y_test, log_test_predictions, dec_tree_pred, rf_pred, 
                                  xgb_test_predictions), axis = 1)

class_all_pd = pd.DataFrame(class_all_results)
class_all_pd.to_csv('class_all_predictions_gCV.csv')


log_test_predictions_score = log_test_predictions_score.reshape(len(log_test_predictions_score), 1)
td_test_predictions_score = td_test_predictions_score.reshape(len(td_test_predictions_score), 1)
rf_test_predictions_score = rf_test_predictions_score.reshape(len(rf_test_predictions_score), 1)
xgb_test_predictions_score = xgb_test_predictions_score.reshape(len(xgb_test_predictions_score), 1)

class_all_results = np.concatenate((y_test, log_test_predictions_score, td_test_predictions_score,
                                    rf_test_predictions_score, 
                                    xgb_test_predictions_score), axis = 1)

class_all_pd = pd.DataFrame(class_all_results)
class_all_pd.to_csv('class_all_scores_gCV.csv')


best_parameters = [log_best, dt_best, rf_best, xgb_best]