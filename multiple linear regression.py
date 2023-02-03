# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 22:17:59 2023

@author: busra
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# import data
df = pd.read_csv("multiple_linear_regression_dataset.csv",sep = ";")

y=df.maas.values.reshape(-1,1)
x=df.iloc[:,[0,2]].values

multiple_linear_reg=LinearRegression()
multiple_linear_reg.fit(x,y)

print("b0: ",multiple_linear_reg.intercept_)

print("b1,b2: ",multiple_linear_reg.coef_)

multiple_linear_reg.predict([np.array[10,35],[5,35]])
