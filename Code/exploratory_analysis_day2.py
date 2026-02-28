#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


#%%
# Load the data
data = pd.read_csv('/Users/tomas/CompBME/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

day_number = data['day']
active_cases = data['active reported daily cases']
#%%
# We have day number, date, and active cases. We can use the day number and active cases to fit an exponential growth curve to estimate R0.
# Let's define the exponential growth function
def exponential_growth(t, r):
    return np.exp(r * t)

# Fit the exponential growth model to the data. 
# We'll use a handy function from scipy called CURVE_FIT that allows us to fit any given function to our data. 
# We will fit the exponential growth function to the active cases data. HINT: Look up the documentation for curve_fit to see how to use it.


t = np.array(day_number)
t = t - t[0]
y = np.array(active_cases)

y_norm = y / y[0]

popt, pcov = curve_fit(exponential_growth, t, y_norm)
r = popt[0]


# Approximate R0 using this fit

D = 7
R0 = 1 + r * D


# Add the fit as a line on top of your scatterplot.