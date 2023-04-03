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

# Mean of the time series
x_bar = sum(data)/len(data)


# Step 1 of building a profile
y = []
for i in range(len(data)):
    sum = 0
    for k in range(i + 1):
        sum = sum + data[k] - x_bar
    y.append(sum)
        

# Step 2 of the method
# Choosing the value of s to be 12
# so Ns to be divided into groups of 186
# As N is 2234, the last list will have 188

Ns = 12
s = int(len(data)/Ns)


y_list = []

for i in range(Ns):
    y_list.append(np.array(y[s*(i):s*(i + 1)]).tolist())
    
x = np.array(list(range(1,s + 1)))    

# Fit a line to each y_list
curve_fit = []
for i in range(Ns):
    curve_fit.append(np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y_list[i], rcond=None)[0].tolist())
    
print(curve_fit)

# Generate curves for each fit
curves = []
for coeff in curve_fit: 
    m = []
    for i in range(1,s + 1):
        m.append(coeff[0]*i + coeff[1])
    curves.append(m)

print(curves)

# Calculate variance for each curve
variance_list = []
for i in range(Ns):
    variance = 0
    for h in range(s):
      variance += ((y_list[i][h] - curves[i][h])**2/s)
    variance_list.append(variance)

print(variance_list)

# qth order fluctuation function
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
plt.title('Fluctuation Function Plot')
plt.show()

# Plotting the profile
plt.plot(range(len(y)), y)
plt.xlabel('Index')
plt.ylabel('Profile')
plt.title('Profile of the Time Series')
plt.show()

# Plotting the lines fitted to each segment
for i in range(Ns):
    plt.plot(x, y_list[i], 'o')
    plt.plot(x, curves[i], '-')
plt.xlabel('Index')
plt.ylabel('Profile')
plt.title('Fitted Lines for Each Segment')
plt.show()


from numpy.polynomial import polynomial as P

order = 2

poly_curve_fit = []

for i in range(Ns):
    poly_curve_fit.append(P.polyfit(x, y_list[i], order))

poly_curves = []
for coeff in poly_curve_fit: 
    m = []
    for i in range(1,s + 1):
        m.append(P.polyval(i, coeff))
    poly_curves.append(m)

poly_variance_list = []
for i in range(Ns):
    poly_variance = 0
    for h in range(s):
      poly_variance += ((y_list[i][h] - poly_curves[i][h])**2/s)
    poly_variance_list.append(poly_variance)

poly_fluc_fun_list = []
for i in range(len(poly_variance_list)):
    poly_fluc_fun_list.append(fluc_fun(i, poly_variance_list))

plt.plot(numb, fluc_fun_list, label='Linear Fit')
plt.plot(numb, poly_fluc_fun_list, label=f'{order}-degree Polynomial Fit')
plt.xlabel('q')
plt.ylabel('fluctuation function')
plt.title('Comparison of Linear and Polynomial Fits')
plt.legend()
plt.show()

def dfa_realizations(data, Ns_list):
    results = []
    for Ns in Ns_list:
        s = int(len(data)/Ns)
        y_list = []
        for i in range(Ns):
            y = []
            x_bar = sum(data[s*i:s*(i + 1)])/s
            for k in range(s*i, s*(i + 1)):
                y.append(data[k] - x_bar)
            y_list.append(np.array(y).tolist())
        curve_fit = []
        for i in range(Ns):
            x = np.array(list(range(1,s + 1)))    
            curve_fit.append(np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y_list[i], rcond=None)[0].tolist())
        curves = []
        for coeff in curve_fit: 
            m = []
            for i in range(1,s + 1):
                m.append(coeff[0]*i + coeff[1])
            curves.append(m)
        variance_list = []
        for i in range(Ns):
            variance = 0
            for h in range(s):
                variance += ((y_list[i][h] - curves[i][h])**2/s)
            variance_list.append(variance)
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
            fluc_fun_list.append(fluc_fun(2, variance_list))
        results.append(fluc_fun_list)
    return results

df = pd.read_excel("Data.xlsx", sheet_name=0)
data = list(df['Returns'])
Ns_list = [6, 8, 10, 12, 14]
results = dfa_realizations(data, Ns_list)

# Plot the results for each realization
import matplotlib.pyplot as plt
for i, res in enumerate(results):
    plt.plot(list(range(1, len(res) + 1)), res, label=f"Ns={Ns_list[i]}")


plt.xlabel('q')
plt.ylabel('fluctuation function')
plt.title('Comparison of Linear and Polynomial Fits')
plt.legend()
plt.show()

