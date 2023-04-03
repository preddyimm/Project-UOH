# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:44:44 2023

@author: rajes
"""
import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt



df = pd.read_excel("Data.xlsx", sheet_name=0)
data = list(df['Returns'])

#Mean of the time series
x_bar = sum(data)/len(data)


#Doing Step 1 of building a profile
y = []
for i in range(len(data)):
    sum = 0
    for k in range(i + 1):
        sum = sum + data[k] - x_bar
    y.append(sum)
        

#Doing Step 2 of method
#Choosing the value of s to be 12
#so Ns to be divided into groups of 186
#As N is 2234, the last list will have 188

Ns = 12
s = int(len(data)/Ns)


y_list = []

for i in range(Ns):
    y_list.append(np.array(y[s*(i):s*(i + 1)]).tolist())
    
x = np.array(list(range(1,s + 1)))    

#print(y_list)


curve_fit = []

for i in range(Ns):
    curve_fit.append(np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y_list[i], rcond=None)[0].tolist())
    
print(curve_fit)

curves = []

for coeff in curve_fit: 
    m = []
    for i in range(1,s + 1):
        m.append(coeff[0]*i + coeff[1])
    curves.append(m)

print(curves)

variance_list = []

for i in range(Ns):
    variance = 0
    for h in range(s):
      
      variance += ((y_list[i][h] - curves[i][h])**2/s)
    variance_list.append(variance)

print(variance_list)

#qth order fluctuation function

def fluc_fun(q, lis):
    f_q_s = 0
    if q != 0:
        for i in lis:
          f_q_s += i**(q/2)/Ns
        return f_q_s**(1/q)
    else:
        abc = 0
        for i in lis:
            abc += np.log(i)/(2*Ns)
        return math.exp(abc)


fluc_fun_list = []
for i in range(len(variance_list)):
    fluc_fun_list.append( fluc_fun(i, variance_list))


print(fluc_fun_list)

numb = list(range(1, Ns + 1))
plt.plot(numb, fluc_fun_list)

plt.xlabel('q')
plt.ylabel('fluctuation function')

plt.title('graph')
plt.show()