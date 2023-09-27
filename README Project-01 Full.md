# Project-Concrete-Strength-Prediction

Overview
This is an End to End project on how to build a Machine Learning model and Deploy the same using AWS EC2. First Exploring the data using basic Exploratory Data Analysis and gaining insights about the patterns and behaviour of different features contributing to the compressive strength of the concrete and then building a suitable model (Linear Regression) and predicting the compressive strength given the independent variable values.

Table of Contents
Project Description
Data
Data Preprocessing
Exploratory Data Analysis (EDA)
Model Building
Model Evaluation
Usage

Project Description
This is a model used to predict Concrete Compressive Strength using certain crucial input features. The data set used here involves 1,030 observations and 8 input features and 1 Output feature. Aimed to solve the problem of the limitations seen in the conventional compressive strength procedure which involves high cost, time consumption, Destructive testing and came up with a precise prediction model which by taking certain simple input features can predict the compressive strength of the concrete structure.

Data
Data Type: multivariate
 
Abstract: Concrete is the most important material in civil engineering. The 
concrete compressive strength is a highly nonlinear function of age and 
ingredients. These ingredients include cement, blast furnace slag, fly ash, 
water, superplasticizer, coarse aggregate, and fine aggregate.

---------------------------------

Sources: 

  Original Owner and Donor
  Prof. I-Cheng Yeh
  Department of Information Management 
  Chung-Hua University, 
  Hsin Chu, Taiwan 30067, R.O.C.
  e-mail:icyeh@chu.edu.tw
  TEL:886-3-5186511

  Date Donated: August 3, 2007
 
---------------------------------

Data Characteristics:
    
The actual concrete compressive strength (MPa) for a given mixture under a 
specific age (days) was determined from laboratory. Data is in raw form (not scaled). 

Summary Statistics: 

Number of instances (observations): 1030
Number of Attributes: 9
Attribute breakdown: 8 quantitative input variables, and 1 quantitative output variable
Missing Attribute Values: None

---------------------------------

Variable Information:

Given is the variable name, variable type, the measurement unit and a brief description. 
The concrete compressive strength is the regression problem. The order of this listing 
corresponds to the order of numerals along the rows of the database. 

Name -- Data Type -- Measurement -- Description

Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
Age -- quantitative -- Day (1~365) -- Input Variable
Concrete compressive strength -- quantitative -- MPa -- Output Variable

cement: a substance used for construction that hardens to other materials to bind them together.

slag: Mixture of metal oxides and silicon dioxide.

Flyash: coal combustion product that is composed of the particulates that are driven out of coal-fired boilers together with the flue gases.

Water: It is used to form a thick paste.

Superplasticizer:  used in making high-strength concrete.

Coaseseaggregate: prices of rocks obtain from ground deposits. 

fineaggregate: the size of aggregate small than 4.75mm.

age: Rate of gain of strength is faster to start with and the rate gets reduced with age.

csMPa: Measurement unit of concrete strength.

The dataset can be found within this repository folder Project-01 Full.

Data Preprocessing
Handled Outliers, used Yeo-Johnson power transformation techniques to normalise the distribution of the features.

Exploratory Data Analysis (EDA)
EDA has been performed to understand the distribution of all the input features.

Model Building
Linear Regression algorithm is used to build the machine learning model, also checked the model performance using regularization techniques such as Lasso and Ridge.

Model Evaluation
Model's performance has been evaluated using R squared, Adjusted R squared (using OLS method), MAE, MAPE, RMSE, Cross Validation tests for both test and train datasets.

Usage
The model can be trained as shown in the .ipynb file and relevant input variables can be given to predict the output by using the following example code: #1 new_prediction = linear_model.predict(pt_yj.transform([[389.9, 189.0, 0.0, 145.9, 22, 944.7, 755.8, 91]]))
              #2 print("The compressive strength of the concrete structure with the given input features is:",new_prediction).
