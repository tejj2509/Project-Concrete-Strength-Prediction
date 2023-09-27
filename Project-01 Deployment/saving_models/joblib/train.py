import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
sns.set()
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
sns.set_style('whitegrid')
import joblib

data = "D:\Anaconda_JPNB\ML PROJECTS PROFILE\Project - 3 Concrete Compressive Strength Predictor\concrete_data.csv"

df = pd.read_csv(data)
# print(df.head())

df_model = df.copy()

Q1 = df_model['Blast Furnace Slag'].quantile(0.25)
Q3 = df_model['Blast Furnace Slag'].quantile(0.75)

IQR = Q3 - Q1

upper_limit = Q3 + 1.5*IQR
lower_limit = Q1 - 1.5*IQR

df_model['Blast Furnace Slag'] = np.where(df_model['Blast Furnace Slag'] > upper_limit, upper_limit, 
                                         np.where(df_model['Blast Furnace Slag']<lower_limit, lower_limit, df_model['Blast Furnace Slag']))
Q1 = df_model['Water'].quantile(0.25)
Q3 = df_model['Water'].quantile(0.75)

IQR = Q3 - Q1

upper_limit = Q3 + 1.5*IQR
lower_limit = Q1 - 1.5*IQR

df_model['Water'] = np.where(df_model['Water'] > upper_limit, upper_limit, 
                                         np.where(df_model['Water']<lower_limit, lower_limit, df_model['Water']))

Q1 = df_model['Superplasticizer'].quantile(0.25)
Q3 = df_model['Superplasticizer'].quantile(0.75)

IQR = Q3 - Q1

upper_limit = Q3 + 1.5*IQR
lower_limit = Q1 - 1.5*IQR

df_model['Superplasticizer'] = np.where(df_model['Superplasticizer'] > upper_limit, upper_limit, 
                                         np.where(df_model['Superplasticizer']<lower_limit, lower_limit, df_model['Superplasticizer']))

Q1 = df_model['Fine Aggregate'].quantile(0.25)
Q3 = df_model['Fine Aggregate'].quantile(0.75)

IQR = Q3 - Q1

upper_limit = Q3 + 1.5*IQR
lower_limit = Q1 - 1.5*IQR

df_model['Fine Aggregate'] = np.where(df_model['Fine Aggregate'] > upper_limit, upper_limit, 
                                         np.where(df_model['Fine Aggregate']<lower_limit, lower_limit, df_model['Fine Aggregate']))

df_model = df_model.rename(columns={'Age':'Age_days'})

Q1 = df_model['Age_days'].quantile(0.25)
Q3 = df_model['Age_days'].quantile(0.75)

IQR = Q3 - Q1

upper_limit = int(Q3 + 1.5*IQR - 0.5)
lower_limit = int(Q1 - 1.5*IQR + 0.5)

df_model['Age_days'] = np.where(df_model['Age_days'] > upper_limit, upper_limit, 
                                         np.where(df_model['Age_days']<lower_limit, lower_limit, df_model['Age_days']))

x = df_model.drop(['Strength'],axis=1)
y = df_model['Strength']

from sklearn.model_selection import train_test_split, cross_val_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import PowerTransformer
pt_yj = PowerTransformer(method='yeo-johnson')
x_train_tf = pt_yj.fit_transform(x_train)
x_test_tf = pt_yj.fit_transform(x_test)

# Building a Linear Regression model:

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

linear_model = LinearRegression()
linear_model.fit(x_train_tf, y_train)
y_pred_train = linear_model.predict(x_train_tf)
y_pred_test = linear_model.predict(x_test_tf)

# print('Train Accuracy',r2_score(y_train, y_pred_train))
# print('\n')
# print('Test Accuracy',r2_score(y_test, y_pred_test))
# print('\n')
# print('Mean Absolute Error for test',mean_absolute_error(y_test, y_pred_test))
# print('\n')
# print('Mean Absolute Percent Error for test',mean_absolute_percentage_error(y_test, y_pred_test)*100)
# print('\n')
# print('Root Mean Squared Error for test',np.sqrt(mean_squared_error(y_test, y_pred_test)))
# print('\n')
# print('Cross Validation score using train data',round(np.mean(cross_val_score(linear_model, x_train_tf, y_train, cv=10, scoring='r2')),4))
# print('\n')
# print('Cross Validation score using test data',round(np.mean(cross_val_score(linear_model, x_test_tf, y_test, cv=10, scoring='r2')),4))

# def predict_strength(Cement, Blast_Furnace_Slag, Fly_Ash, Water, Superplasticizer, Coarse_Aggregate, Fine_Aggregate, Age_days):
#     x = []
#     x.append(Cement)
#     x.append(Blast_Furnace_Slag)
#     x.append(Fly_Ash)
#     x.append(Water)
#     x.append(Superplasticizer)
#     x.append(Coarse_Aggregate)
#     x.append(Fine_Aggregate)
#     x.append(Age_days)
#     input_tf = pt_yj.transform([x])
#     prediction = linear_model.predict(input_tf)
#     return prediction

# print(predict_strength(389.9, 189.0, 0.0, 145.9, 22, 944.7, 755.8, 91))

# Model Saving
filename = 'concrete_strength_81.pkl'
joblib.dump(linear_model, filename)

transformer = 'yjtransformer.pkl'
joblib.dump(pt_yj, transformer)



# new_prediction = linear_model.predict(pt_yj.transform([[389.9, 189.0, 0.0, 145.9, 22, 944.7, 755.8, 91]]))
# print("The compressive strength of the concrete structure with the given input features is:",new_prediction)