# Project-UOH
The code is a Python script that uses several modules including numpy, pandas, math, and matplotlib.pyplot to perform multifractal analysis on a given data set. It reads a set of data from an Excel file (specified in the code as "Data.xlsx") and stores it in a list named data.

The first major function defined in the code is called fluctuation_function, which takes in three arguments: data_list, Ns, and q. The data_list parameter is a list of data points, Ns is an integer that specifies the number of segments to divide the data into, and q is a real number that is used to compute the fluctuation function.

The function starts by computing the mean of the time series using the formula x_bar = sum(data_list)/len(data_list). Then, it calculates a detrended series y by subtracting the mean from each data point and then cumulatively summing the deviations from the mean. This detrended series y is then divided into Ns segments, with each segment containing s data points.

The code then fits a linear function to each segment using least squares regression and stores the parameters of the fit in a list curve_fit. For each segment, the fitted linear function is subtracted from the data points in that segment, and the variance of the residuals is computed. The fluc_fun function is then called to compute the fluctuation function for the given q value, which is a function of the variances of the residuals for each segment.

The q_f_s function is defined to compute the fluctuation function for different Ns values using a range of Ns values. This function is used in the plotting of the fluctuation function vs. Ns graphs for different q values.

The code then computes the Hurst exponent using the h_q function. The Hurst exponent is a measure of the long-term memory of the data series. For each q value in a range, the function computes the logarithms of the fluctuation functions at different Ns values and then fits a straight line to the data using least squares regression. The slope of the fitted line is the Hurst exponent H(q) for that q value. The hurst_function function is defined to compute the Hurst exponent for a given q value using the coefficients obtained from the polynomial fit.

Next, the scaling exponent t(q) is computed using the scaling_exponent function, which is a function of the Hurst exponent and the q value. The singularity strength α is computed using the singularity_strength function, which is a function of the Hurst exponent and its derivative with respect to q. The multifractal spectrum f(α) is computed using the multifractal_spectrum function, which is a function of the singularity strength and q.

The code then plots the t(q) vs. q graph and the f(α) vs. α graph using matplotlib.pyplot. Finally, the code prints the ranges of the Ns and q values used in the analysis.

In summary, the code performs multifractal analysis on a given data set using the detrended fluctuation analysis (DFA) method. It computes the fluctuation function, Hurst exponent, scaling exponent, singularity strength, and multifractal spectrum, and then plots various graphs related to these measures.
