# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 22:38:00 2021

@author: 91892
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = 'https://github.com/codebasics/py/blob/master/ML/14_naive_bayes/spam.csv'

df = pd.read_csv('spam.csv')
df.head()