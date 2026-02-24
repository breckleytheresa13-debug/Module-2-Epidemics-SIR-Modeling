#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data
data = pd.read_csv('../Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)

#%%
# Make a plot of the active cases over time

plt.plot(data['day'], data['active reported daily cases'], marker='o', linestyle='-')
plt.xlabel('Day')
plt.ylabel('Active Infections')
plt.title('Day vs Active Infections')
plt.grid(True)
plt.show()

# approximaetely exponential growth, but not exactly. We can see that the number of active infections is increasing over time, but the rate of increase is not constant. It appears to be accelerating, which is characteristic of exponential growth, but there are some fluctuations in the data that suggest it may not be a perfect exponential curve.


# Day 2: Estimate R_0 for the mystrey virus data using the fit for exponential growth in I.
# Plot estimated exponential curve over your scatterplot from last class.

# What viruses ahve a similar R_0?
# How accurate do you this R_o estuiamte is?