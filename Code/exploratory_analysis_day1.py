#%%
import pandas as pd
import matplotlib.pyplot as plt
import csv

#%%
# Load the data
data = pd.read_csv('/Users/tomas/CompBME/Breckley_Daniel - Module_2_Epidemics_SIR_Modeling/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv', parse_dates=['date'], header=0, index_col=None)


days = []
cases = []

with open("/Users/tomas/CompBME/Breckley_Daniel - Module_2_Epidemics_SIR_Modeling/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#1.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        days.append(int(row["day"]))
        cases.append(int(row["active reported daily cases"]))

plt.plot(days, cases)
plt.xlabel("# of Days")
plt.ylabel("Active Infections")
plt.title("Mystery Virus - Active Infections")
plt.show()