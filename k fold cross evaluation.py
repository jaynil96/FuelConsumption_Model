# -*- coding: utf-8 -*-
"""
Created on Sun May 17 11:53:06 2020

@author: jayni
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/jayni/.spyder-py3/learning/resources/FuelConsumption.csv")

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
from sklearn import linear_model #import regression model
regr = linear_model.LinearRegression()
r = []
for train_index, test_index in kf.split(df):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    train = cdf.iloc[train_index]
    test = cdf.iloc[test_index]
    
    # train the model
    train_x = np.asanyarray(train[['ENGINESIZE']]) #set all train to train x values 
    train_y = np.asanyarray(train[['CO2EMISSIONS']]) #set all train to train y values
    regr.fit (train_x, train_y) # try to regress fit for both train_x and train y values
    
    #test the model
    from sklearn.metrics import r2_score

    test_x = np.asanyarray(test[['ENGINESIZE']]) #cast test array['EngineSize']to asanarray
    test_y = np.asanyarray(test[['CO2EMISSIONS']]) #cast test array['CO2 Emissions']to asanarray
    test_y_ = regr.predict(test_x) #try to predetic the values and see how accurate the model is

    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    r.append(r2_score(test_y_ , test_y))
    print("R2-score: %.2f" % r2_score(test_y_ , test_y) )
    
    """
    #plot the values to ensure that they are seperate data points
    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    print(df)

    """
r = np.asanyarray(r)
print("the average r score is", np.mean(r) )
plt.bar([1,2,3,4,5], r)

g = np.asarray([[2.3], [3.2]])
print("predicted value", regr.predict(g))

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r') #plot the line using regr.coef_[0][0]*train_x + regr.intercept_[0]
plt.xlabel("Engine size")
plt.ylabel("Emission")