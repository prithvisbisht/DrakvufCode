#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:02:37 2018

@author: prithvi_29
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_excel('process.xls');
data=data.fillna(0);
X = data.iloc[:,1:].values #-1 means exclude last column
Y = data.iloc[:,0].values #taking values from last column 

from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE

svm = LinearSVC()
# create the RFE model for the svm classifier 
# and select attributes
rfe = RFE(svm)
rfe = rfe.fit(X, Y)
# print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)