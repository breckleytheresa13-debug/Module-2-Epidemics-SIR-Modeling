import matplotlib.pyplot as plt
import numpy as np
import csv

# Graph that I made

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

# NEEDED AI TO HELP ME CREATE THIS PLOT
# WANTED A NICER GRAPH AND IT MADE EVERYTHING BELOW



days = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]

cases = [1,1,1,1,1,1,2,2,2,3,3,4,4,4,5,6,7,9,9,10,
         13,14,16,17,20,24,25,31,33,38,43,54,56,60,75,76,93,94,110,134,155,170,189,211,223]


fit_days = np.array(days[19:])
fit_cases = np.array(cases[19:])
coeffs = np.polyfit(fit_days, np.log(fit_cases), 1)
r, log_a = coeffs
a = np.exp(log_a)
fit_line = a * np.exp(r * np.array(days))


fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#0a0d0f')
ax.set_facecolor('#0f1417')

ax.plot(days, cases, color='#ff3b3b', linewidth=2, marker='o',
        markersize=4, markerfacecolor='#ff3b3b', label='Active Infections (reported)')
ax.fill_between(days, cases, alpha=0.08, color='#ff3b3b')
ax.plot(days, fit_line, color='#ffb347', linewidth=1.5,
        linestyle='--', label=f'Exponential fit (r={r:.4f}/day)')

ax.set_title('Mystery Virus — Active Infections Over Time\nData Release #1',
             color='white', fontsize=14, pad=16)
ax.set_xlabel('Day', color='#4a6070', fontsize=11)
ax.set_ylabel('Active Infections', color='#4a6070', fontsize=11)
ax.tick_params(colors='#4a6070')
for spine in ax.spines.values():
    spine.set_edgecolor('#1e2830')
ax.grid(color='#1e2830', linewidth=0.7)
ax.legend(facecolor='#0f1417', edgecolor='#1e2830', labelcolor='white', fontsize=10)

doubling_time = np.log(2) / r
ax.annotate(f'Doubling time ≈ {doubling_time:.1f} days\nDay 45 cases: 223',
            xy=(40, 134), xytext=(28, 160),
            color='#ffb347', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#ffb347'))

plt.tight_layout()
plt.savefig('virus_plot.png', dpi=150, facecolor=fig.get_facecolor())
plt.show()
print(f"Growth rate r = {r:.4f} per day")
print(f"Doubling time = {doubling_time:.1f} days")