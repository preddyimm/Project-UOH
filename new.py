import pandas as pd
import math
import matplotlib.pyplot as plt

df = pd.read_excel("Exchange rates.xlsx", sheet_name=1)
data = list(df['EUR'])

def fluctuation_function(data_list, Ns, q):
    fluctuation_function_calls = 0
    
    # Mean of the time series
    x_bar = sum(data_list) / len(data_list)

    # Doing Step 1 of building a profile
    y = []
    for i in range(len(data_list)):
        summation = 0
        for k in range(i + 1):
            summation += data_list[k] - x_bar
        y.append(summation)

    # Doing Step 2 of method
    s = int(len(data_list) / Ns)
    y_list = []
    for i in range(Ns):
        y_list.append([y[s*(i):s*(i+1)]])

    x = [i for i in range(1, s + 1)]
    curve_fit = []

    for i in range(Ns):
        denominator = len(x)
        coefficient1 = 0
        coefficient2 = 0
        for j in range(len(x)):
            coefficient1 += x[j] * y_list[i][j]
            coefficient2 += y_list[i][j]
        denominator_coefficient = denominator * sum([pow(i, 2) for i in x]) - pow(sum(x), 2)
        coefficient1 = (denominator * coefficient1 - sum(x) * coefficient2) / denominator_coefficient
        coefficient2 = (sum([pow(i, 2) for i in x]) * coefficient2 - sum(x) * coefficient1) / denominator_coefficient
        curve_fit.append([coefficient1, coefficient2])

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
            variance += pow(y_list[i][h] - curves[i][h], 2) / s
        variance_list.append(variance)

    def fluc_fun(q, lis):
        f_q_s = 0
        if q != 0:
            for i in lis:
                f_q_s += pow(i, q/2) / Ns
            return pow(f_q_s, 1/q)
        else:
            abc = 0
            for i in lis:
                abc += math.log(i) / (2 * Ns)
            return math.exp(abc)
    
    result = fluc_fun(q, variance_list)
    fluctuation_function_calls += 1
    return result

fluctuation_function.calls = 0

def q_f_s(lis, q):
    fluctuation_function.calls = 0
    f_f = [fluctuation_function(lis, i, q) for i in range(1, 20)]
    print("fluctuation_function was called", fluctuation_function.calls, "times.")
    return f_f

Ns_range = range(1, 20)
numb = [len(data)/i for i in Ns_range]
q_range = range(-20, 21)

for f in q_range:
    plt.loglog(numb, q_f_s(data, f), label=f"q = {f}")

plt.xlabel('s')
plt.ylabel('fluctuation function')
plt.title('fluctuation function vs s')
plt.legend()
plt.show


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

print("The Hurst function at q = 2 is : " + str(hurst_function(2)))

def diff_hurst_function(q):
    return (a*(q**3)*4 + b*(q**2)*3 + c*(q)*2 + d)


#scaling exponent
def scaling_exponent(q):
    return q*hurst_function(q) - 1
q_list = np.arange(-10, 11, 0.1).tolist()

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
