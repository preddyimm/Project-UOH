
import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt


df = pd.read_excel("SNP.xlsx", sheet_name=0)
data = list(df['Returns'])


def fluctuation_function(data_list, Ns, q):
    
    # Mean of the time series
    x_bar = sum(data_list)/len(data_list)

    # Doing Step 1 of building a profile
    y = []
    for i in range(len(data_list)):
        summation = 0
        for k in range(i + 1):
            summation += data_list[k] - x_bar
        y.append(summation)
        
    

    # Doing Step 2 of method

    s = int(len(data_list)/Ns)

    y_list = []

    for i in range(Ns):
        y_list.append(np.array(y[s*(i):s*(i + 1)]).tolist())
        
    x = np.array(list(range(1, s + 1)))    

    curve_fit = []

    for i in range(Ns):
        curve_fit.append(np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y_list[i], rcond=None)[0].tolist())

    curves = []

    for coeff in curve_fit: 
        m = []
        for i in range(1, s + 1):
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
              f_q_s += i**(q/2)/(Ns)
            return f_q_s**(1/q)
        else:
            abc = 0
            for i in lis:
                abc += np.log(i)/(2*Ns)
            return math.exp(abc)
        
    return fluc_fun(q, variance_list)

Ns_range = range(4,200)
def q_f_s(lis, q):
    f_f = [(fluctuation_function(data, i, q)) for i in Ns_range]
    return f_f

    
numb = [(len(data)/i) for i in Ns_range]

q_range = range(-10,11)

for f in q_range:
    plt.plot(numb, q_f_s(data, f), label=f"q = {f}")


plt.xlabel('s')
plt.ylabel('fluctuation function')
plt.title('fluctuation function vs s')
plt.legend()
plt.show()


h_q = []
for q in q_range:
  l = np.array([math.log(f) for f in q_f_s(data, q)])
  g = np.array([math.log(i) for i in numb])

  m, n = np.polyfit(g, l, 1)
  h_q.append(m)
plt.plot(q_range, h_q)

plt.xlabel('q')
plt.ylabel('hurst expression')
plt.title('Hurst Expression vs q')
plt.show()


a, b, c, d, e = np.polyfit(q_range, h_q, 4)

def hurst_function(q):
    return a*(q**4) + b*(q**3) + c*(q**2) + d*q + e

def diff_hurst_function(q):
    return (a*(q**3)*4 + b*(q**2)*3 + c*(q)*2 + d)

def scaling_exponent(q):
    return q*hurst_function(q) - 1

q_list = np.arange(-10, 11, 0.1).tolist()

#scaling exponent
t_q = [scaling_exponent(i) for i in q_list]
plt.plot(q_list, t_q)
plt.xlabel('q')
plt.ylabel('scaling exponent')
plt.title('t(q) vs q')
plt.show()

#singularity strength α
def singularity_strength(q):
    return hurst_function(q) + q*diff_hurst_function(q)
    

#Multifractal spectrum f(α)
def multifractal_spectrum(q):
    return q*(singularity_strength(q)) - scaling_exponent(q)

f_a = [multifractal_spectrum(i) for i in q_list]
alpha = [singularity_strength(i) for i in q_list]
plt.scatter(alpha, f_a)
plt.xlabel('α')
plt.ylabel('multifractal_spectrum')
plt.title('f(a) vs a')
plt.show()

print(Ns_range, q_range)



