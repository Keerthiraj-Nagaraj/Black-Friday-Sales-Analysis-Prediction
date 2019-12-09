# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:36:09 2019

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
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Performance analysis libraries
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# =============================================================================


# =============================================================================
# =============================================================================
# # Defining necessary functions
# =============================================================================

def train_model(model_name, traindata, testdata, in_features, target):
    
    #Fitting the model
    model_name.fit(traindata[in_features], traindata[target])
    
    #Predictions for training and testing data
    train_predictions = model_name.predict(traindata[in_features])
    
    
    cv_score = cross_val_score(model_name, traindata[in_features],(traindata[target]) , 
    cv=10, scoring='neg_mean_squared_error')
    
    cv_score = np.sqrt(np.abs(cv_score))
    
    
    test_predictions = model_name.predict(testdata[in_features])
    
    
    train_rsme = np.sqrt(mean_squared_error((traindata[target]).values, train_predictions))
    test_rmse = np.sqrt(mean_squared_error((testdata[target]).values, test_predictions))
    
    print("\n Training and Testing Performance Report")
    print("Train RMSE : %.4g" % np.sqrt(mean_squared_error((traindata[target]).values, train_predictions)))
    print("Test RMSE : %.4g" % np.sqrt(mean_squared_error((testdata[target]).values, test_predictions)))
    
    print("CV Score : Mean - %.4g | Std - %.4g" % (np.mean(cv_score),np.std(cv_score)))
    
    
    rmse_val = [train_rsme, test_rmse, np.mean(cv_score)]
    
    
    
    return rmse_val, test_predictions

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

# =============================================================================


time_list.append(time.time())
# =============================================================================
# =============================================================================
# # Regression Model Training
# =============================================================================
print('Regression Model Training.......')


# Multiple Linear Regression
print('Multiple Linear Regression')
lin_reg = LinearRegression(normalize=True)
lin_rmse, lin_pred = train_model(lin_reg, train_data, test_data, in_features, target)

filename = 'model_lin_reg.sav'
pickle.dump(lin_reg, open(filename, 'wb'))

print(r2_score(test_data[target].values, lin_pred)) 


time_list.append(time.time())
# Decision Tree Regression
print('Decision Tree Regression')

params_dt = {'max_depth': [5, 10, 15, 20], 'min_samples_leaf': [50, 100]}


dec_trees = DecisionTreeRegressor(random_state=0)

td_grid = GridSearchCV(dec_trees, params_dt, verbose=1)
td_grid.fit(train_data[in_features], train_data[target])

print(r2_score(test_data[target].values, td_grid.best_estimator_.predict(test_data[in_features]))) 


#Predictions
td_train_predictions = td_grid.best_estimator_.predict(train_data[in_features])
td_test_predictions = td_grid.best_estimator_.predict(test_data[in_features])

td_train_rmse = np.sqrt(mean_squared_error((train_data[target]).values, td_train_predictions))
td_test_rmse = np.sqrt(mean_squared_error((test_data[target]).values, td_test_predictions))

print("DT Train RMSE : %.4g" % td_train_rmse)
print("DT Test RMSE : %.4g" % td_test_rmse)

filename = 'model_dec_trees_gCV.sav'
pickle.dump(td_grid.best_estimator_, open(filename, 'wb'))


time_list.append(time.time())
# Random Forest Regression
print('Random Forest Regression')


params_rf = {'n_estimators' : [5, 10], 'max_depth': [5, 10, 15], 
             'min_samples_leaf': [5, 10, 20]}


rf_model = RandomForestRegressor(n_jobs = -1, criterion='mse', 
                       min_samples_leaf=5, 
                       oob_score = True, 
                       random_state=0)

rf_grid = GridSearchCV(rf_model, params_rf, 
                       verbose=1)

rf_grid.fit(train_data[in_features], train_data[target])

print(r2_score(test_data[target].values, rf_grid.best_estimator_.predict(test_data[in_features]))) 


#Predictions
rf_train_predictions = rf_grid.best_estimator_.predict(train_data[in_features])
rf_test_predictions = rf_grid.best_estimator_.predict(test_data[in_features])

rf_train_rmse = np.sqrt(mean_squared_error((train_data[target]).values, rf_train_predictions))
rf_test_rmse = np.sqrt(mean_squared_error((test_data[target]).values, rf_test_predictions))

print("RF Train RMSE : %.4g" % rf_train_rmse)
print("RF Test RMSE : %.4g" % rf_test_rmse)

filename = 'model_rf_gCV.sav'
pickle.dump(rf_grid.best_estimator_, open(filename, 'wb'))




time_list.append(time.time())
# XGBoost
print('Extreme Gradient Boosting Decision Tree Regression')

params_xgb = {'n_estimators': [500, 1000],
              'gamma':[i/10.0 for i in range(3,5)],  
              'subsample':[i/10.0 for i in range(6,11)], 
              'max_depth': [5, 10, 15]}


xgb = XGBRegressor(nthread=-1, objective='reg:squarederror',
                    booster='gbtree', 
                    tree_method='auto', 
                    learning_rate=0.05, 
                    random_state=0) 

xgb_grid = GridSearchCV(xgb, params_xgb,
                    verbose=1)

xgb_grid.fit(train_data[in_features], train_data[target])


print(r2_score(test_data[target].values, xgb_grid.best_estimator_.predict(test_data[in_features]))) 

#Predictions
xgb_train_predictions = xgb_grid.best_estimator_.predict(train_data[in_features])
xgb_test_predictions = xgb_grid.best_estimator_.predict(test_data[in_features])

xgb_train_rmse = np.sqrt(mean_squared_error((train_data[target]).values, xgb_train_predictions))
xgb_test_rmse = np.sqrt(mean_squared_error((test_data[target]).values, xgb_test_predictions))

print("XGB Train RMSE : %.4g" % xgb_train_rmse)
print("XGB Test RMSE : %.4g" % xgb_test_rmse)


filename = 'model_xgb_gCV.sav'
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
plt.plot(test_data[target].values[0:100], '.-', label = 'test-data')
plt.plot(lin_pred[0:100], '.-', label = 'Predictions')
plt.ylabel('Purchase')
plt.title('Multiple Linear Regression')
plt.grid(True)
plt.legend()
plt.savefig('lin_pred_gCV.png', dpi = 600)


plt.figure()
plt.plot(test_data[target].values[0:100], '.-', label = 'test-data')
plt.plot(dec_tree_pred[0:100], '.-', label = 'Predictions')
plt.ylabel('Purchase')
plt.title('Decision Trees')
plt.grid(True)
plt.legend()
plt.savefig('dt_pred_gCV.png', dpi = 600)


plt.figure()
plt.plot(test_data[target].values[0:100], '.-', label = 'test-data')
plt.plot(rf_pred[0:100], '.-', label = 'Predictions')
plt.ylabel('Purchase')
plt.title('Random Forest')
plt.grid(True)
plt.legend()
plt.savefig('rf_pred_gCV.png', dpi = 600)


plt.figure()
plt.plot(test_data[target].values[0:100], '.-', label = 'test-data')
plt.plot(xgb_test_predictions[0:100], '.-', label = 'Predictions')
plt.ylabel('Purchase')
plt.title('XGBoost')
plt.grid(True)
plt.legend()
plt.savefig('xgboost_pred_gCV.png', dpi = 600)

plt.figure()
plt.plot(test_data[target].values[0:100], '.-', label = 'test-data')
plt.plot(lin_pred[0:100], '.-', label = 'MLR')
plt.plot(dec_tree_pred[0:100], '.-', label = 'DT')
plt.plot(rf_pred[0:100], '.-', label = 'RF')
plt.plot(xgb_test_predictions[0:100], '.-', label = 'XGBoost')
plt.ylabel('Purchase')
plt.title('Test Vs Predictions for various regression models')
plt.grid(True)
plt.legend()
plt.savefig('reg_all_gCV.png', dpi = 600)



xtick_names = ['MLR', 'DT', 'RF', 'XGBoost']
x_tick_id = np.arange(len(xtick_names))



train_rmse_all = [lin_rmse[0], td_train_rmse, rf_train_rmse, xgb_train_rmse]
test_rmse_all = [lin_rmse[1], td_test_rmse, rf_test_rmse, xgb_test_rmse]

plt.figure()
plt.plot(train_rmse_all, '.-', label = 'train')
plt.plot(test_rmse_all, '.-', label = 'test')
plt.xticks(x_tick_id, ('MLR', 'DT', 'RF', 'XGBoost'))
plt.xlabel('Regression algorithm')
plt.ylabel('RMSE')
plt.title('RMSE comparison')
plt.grid(True)
plt.legend()
plt.savefig('rmse_all_gCV.png', dpi = 600)


y_test = test_data[target].values
y_test = y_test.reshape(len(y_test), 1)
lin_pred = lin_pred.reshape(len(lin_pred), 1)
dec_tree_pred = dec_tree_pred.reshape(len(dec_tree_pred), 1)
rf_pred = rf_pred.reshape(len(rf_pred), 1)
xgb_test_predictions = xgb_test_predictions.reshape(len(xgb_test_predictions), 1)

reg_all_results = np.concatenate((y_test, lin_pred, dec_tree_pred, rf_pred, 
                                  xgb_test_predictions), axis = 1)

reg_all_pd = pd.DataFrame(reg_all_results)
reg_all_pd.to_csv('reg_all_predictions_gCV.csv')




xtick_names = ['MLR', 'DT', 'RF', 'XGBoost']
x_tick_id = np.arange(len(xtick_names))



r2_scores_all = [0.240, 0.6993, 0.7075, 0.7472]

plt.figure()
plt.plot(r2_scores_all, '.-')
plt.xticks(x_tick_id, ('MLR', 'DT', 'RF', 'XGBoost'))
plt.xlabel('Regression algorithm')
plt.ylabel('R2-score')
plt.title('R2-score comparison')
plt.grid(True)
plt.savefig('r2score_all_gCV.png', dpi = 600)
