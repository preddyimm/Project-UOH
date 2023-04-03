#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 01:37:32 2023

@author: preddy
"""


import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt

for i in ["CNY", 'GBP', 'JPY', 'INR', 'EUR']:
  df = pd.read_excel("Exchange rates.xlsx", sheet_name=1)
  data = list(df[i])

  numb = range(len(data))

  plt.plot(numb, data)
  plt.xlabel('q')
  plt.ylabel('scaling exponent')
  plt.title(i)
  plt.show()
  
  
  
df = pd.read_excel("Exchange rates.xlsx", sheet_name=2)
data = list(df['CNY'])

numb = range(len(data))

plt.plot(numb, data)
plt.xlabel('q')
plt.ylabel('scaling exponent')
plt.title('CNY')
plt.show()