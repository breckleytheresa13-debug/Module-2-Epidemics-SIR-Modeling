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

print(f"Fitted growth rate r: {r:.4f} per day")
print(f"Doubling time: {np.log(2)/r:.1f} days")
print(f"Estimated R0 (generation time = {D} days): {R0:.2f}")

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(t, y, color='steelblue', zorder=5, label='Active Cases (observed)')

t_smooth = np.linspace(t[0], t[-1], 300)
y_fit = y[0] * exponential_growth(t_smooth, r)
ax.plot(t_smooth, y_fit, color='tomato', linewidth=2,
        label=f'Exponential fit (r={r:.4f}, R0≈{R0:.2f})')

ax.set_xlabel('Day Number')
ax.set_ylabel('Active Cases')
ax.set_title('Exponential Growth Fit to Active Cases')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Add the fit as a line on top of your scatterplot.