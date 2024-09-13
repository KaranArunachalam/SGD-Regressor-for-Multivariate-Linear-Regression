# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start Step  <br>
2.Data Preparation <br>
3.Hypothesis Definition <br>
4.Cost Function <br>
5.Parameter Update Rule <br>
6.Iterative Training <br>
7.Model Evaluation <br>
8.End

## Program:
```
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Karan A
RegisterNumber:  212223230099
```
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
df.head()
```
![image](https://github.com/user-attachments/assets/c6ce6fb4-721e-478d-8c81-ce7fa7a430a2)
```py
df.info()
```
![image](https://github.com/user-attachments/assets/d371cb6e-a3e8-4ba0-82b1-722c3e9fc3ea)
```py
X=df.drop(columns=['AveOccup','HousingPrice'])
X.info()
```
![image](https://github.com/user-attachments/assets/57f0829d-e6fb-4c62-ab4f-c63499dadc0d)
```py
Y=df[['AveOccup','HousingPrice']]
Y.info()
```
![image](https://github.com/user-attachments/assets/1e7119dc-eb25-432f-be38-5c7c1faf7269)
```py
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
Y_train=scaler_Y.fit_transform(Y_train)
X_test=scaler_X.transform(X_test)
Y_test=scaler_Y.transform(Y_test)
print(X_train)
```
![image](https://github.com/user-attachments/assets/5fa3d3aa-456b-48db-bf47-8994d654c841)
```py
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])
```
![image](https://github.com/user-attachments/assets/d65ca61d-7644-49de-b84a-b954c12f366d)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
