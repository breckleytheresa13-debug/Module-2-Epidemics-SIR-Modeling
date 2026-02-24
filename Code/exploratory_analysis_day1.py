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