#%%

#========================================
##Data Release #1
#========================================
import matplotlib.pyplot as plt
import numpy as np
import csv

# Graph that I made

days = []
cases = []

with open("C:\\Users\\15712\\OneDrive - University of Virginia\\Comp Mod 2\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#1.csv") as f:
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








#%%
# ============================================================
#Data Release #2 Analysis
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Load Data Release #2
# ---------------------------
data2 = pd.read_csv("C:\\Users\\15712\\OneDrive - University of Virginia\\Comp Mod 2\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#2.csv")

# Use the correct columns:
# day = first column
# active infections = third column
days_data = pd.to_numeric(data2.iloc[:, 0], errors="coerce")
infected_data = pd.to_numeric(data2.iloc[:, 2], errors="coerce")

# Keep only rows where both are valid numbers
clean_data = pd.DataFrame({
    "day": days_data,
    "infected": infected_data
}).dropna()

days_data = clean_data["day"].to_numpy(dtype=float)
infected_data = clean_data["infected"].to_numpy(dtype=float)

print("First few rows of Release #2 data:")
print(data2.head())

print("\nCleaned data preview:")
print(clean_data.head())

print("\nNumber of usable data points:", len(days_data))
print("First few days:", days_data[:5])
print("First few active infection values:", infected_data[:5])


# ---------------------------
# Euler's Method for SEIR
# ---------------------------
def euler_seir(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N):
    dt = timepoints[1] - timepoints[0]

    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))

    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    for n in range(len(timepoints) - 1):
        dSdt = -beta * S[n] * I[n] / N
        dEdt = beta * S[n] * I[n] / N - sigma * E[n]
        dIdt = sigma * E[n] - gamma * I[n]
        dRdt = gamma * I[n]

        S[n + 1] = S[n] + dSdt * dt
        E[n + 1] = E[n] + dEdt * dt
        I[n + 1] = I[n] + dIdt * dt
        R[n + 1] = R[n] + dRdt * dt

    return S, E, I, R


# ---------------------------
# SSE Function
# ---------------------------
def calculate_sse(observed, predicted):
    return np.sum((observed - predicted) ** 2)


# ---------------------------
# Modified SEIR to predict daily new cases
# ---------------------------
def seir_daily_new_cases(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N):
    dt = timepoints[1] - timepoints[0]

    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))
    new_cases = np.zeros(len(timepoints))  # Daily new infections

    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    for n in range(len(timepoints) - 1):
        # SEIR equations
        dSdt = -beta * S[n] * I[n] / N
        dEdt = beta * S[n] * I[n] / N - sigma * E[n]
        dIdt = sigma * E[n] - gamma * I[n]
        dRdt = gamma * I[n]

        S[n + 1] = S[n] + dSdt * dt
        E[n + 1] = E[n] + dEdt * dt
        I[n + 1] = I[n] + dIdt * dt
        R[n + 1] = R[n] + dRdt * dt

        # Daily new cases = rate of new infections
        new_cases[n + 1] = beta * S[n] * I[n] / N * dt

    return S, E, I, R, new_cases


# ---------------------------
# Initial Conditions
# ---------------------------
# Population size should be much larger - peak active cases are ~2363
N = 100000  # Increased further to accommodate the epidemic scale
# I0 should be estimated, not set to the first daily case count
I0 = 10  # Estimated initial infectious individuals
E0 = 20  # Estimated initial exposed individuals
R0_init = 0
S0 = N - E0 - I0 - R0_init

timepoints = days_data


# ---------------------------
# Initial Parameter Guess
# ---------------------------
beta_guess = 0.4   # Transmission rate
sigma_guess = 0.25  # 1/latent period (4 days)
gamma_guess = 0.1  # 1/infectious period (10 days)

S_guess, E_guess, I_guess, R_guess, new_cases_guess = seir_daily_new_cases(
    beta_guess, sigma_guess, gamma_guess,
    S0, E0, I0, R0_init,
    timepoints, N
)

sse_guess = calculate_sse(infected_data, new_cases_guess)

print("\nInitial parameter guess results:")
print("beta =", beta_guess)
print("sigma =", sigma_guess)
print("gamma =", gamma_guess)
print("SSE =", sse_guess)
print("Peak predicted daily cases =", np.max(new_cases_guess))
print("Data peak daily cases =", np.max(infected_data))


# ---------------------------
# Plot Initial Guess
# ---------------------------
plt.figure(figsize=(8, 5))
plt.scatter(days_data, infected_data, label="Observed daily new cases")
plt.plot(timepoints, new_cases_guess, label="SEIR model daily new cases")
plt.xlabel("Day")
plt.ylabel("Daily new cases")
plt.title("Release #2: Data vs Initial SEIR Model")
plt.legend()
plt.show()

#%%
# ---------------------------
# Grid Search for Best Fit
# ---------------------------
beta_vals = np.linspace(0.2, 0.8, 16)    # Transmission rate range
sigma_vals = np.linspace(0.15, 0.4, 15)   # Latent period 2.5-6.7 days
gamma_vals = np.linspace(0.04, 0.15, 15)  # Infectious period 6.7-20 days

best_sse = np.inf
best_beta = None
best_sigma = None
best_gamma = None
best_new_cases = None

for beta in beta_vals:
    for sigma in sigma_vals:
        for gamma in gamma_vals:
            S_try, E_try, I_try, R_try, new_cases_try = seir_daily_new_cases(
                beta, sigma, gamma,
                S0, E0, I0, R0_init,
                timepoints, N
            )

            sse = calculate_sse(infected_data, new_cases_try)

            if sse < best_sse:
                best_sse = sse
                best_beta = beta
                best_sigma = sigma
                best_gamma = gamma
                best_new_cases = new_cases_try

print("\nBest-fit parameters from grid search:")
print("Best beta =", best_beta)
print("Best sigma =", best_sigma)
print("Best gamma =", best_gamma)
print("Lowest SSE =", best_sse)
print("Peak predicted daily cases =", np.max(best_new_cases))
print("Data peak daily cases =", np.max(infected_data))



# ---------------------------
# Plot Best-Fit Model
# ---------------------------
plt.figure(figsize=(8, 5))
plt.scatter(days_data, infected_data, label="Observed daily new cases")
plt.plot(timepoints, best_new_cases, label="Best-fit SEIR model daily new cases")
plt.xlabel("Day")
plt.ylabel("Daily new cases")
plt.title("Best-Fit SEIR Model for Release #2")
plt.legend()
plt.show()


# ---------------------------
# Print Model Interpretation
# ---------------------------
R0_model = best_beta / best_gamma
latent_period = 1 / best_sigma
infectious_period = 1 / best_gamma

print("\nModel interpretation:")
print("Estimated model R0 =", R0_model)
print("Estimated latent period (days) =", latent_period)
print("Estimated infectious period (days) =", infectious_period)


# ---------------------------
# Predict Future Peak
# ---------------------------
future_timepoints = np.arange(1, 121, 1)

S_future, E_future, I_future, R_future, new_cases_future = seir_daily_new_cases(
    best_beta, best_sigma, best_gamma,
    S0, E0, I0, R0_init,
    future_timepoints, N
)

peak_index = np.argmax(new_cases_future)
peak_day = future_timepoints[peak_index]
peak_value = new_cases_future[peak_index]

print("\nFuture outbreak prediction:")
print("Predicted peak day =", peak_day)
print("Predicted peak daily new cases =", peak_value)


# ---------------------------
# Plot Future Projection
# ---------------------------
plt.figure(figsize=(8, 5))
plt.scatter(days_data, infected_data, label="Observed daily new cases")
plt.plot(future_timepoints, new_cases_future, label="Projected SEIR daily new cases")
plt.axvline(peak_day, linestyle="--", label=f"Peak day = {peak_day}")
plt.xlabel("Day")
plt.ylabel("Daily new cases")
plt.title("SEIR Projection and Predicted Peak")
plt.legend()
plt.show() 






#%%

##################################
# Data Release #3
##################################
data3 = pd.read_csv("C:\\Users\\15712\\OneDrive - University of Virginia\\Comp Mod 2\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#3.csv")

days_data = pd.to_numeric(data3.iloc[:, 0], errors="coerce")
infected_data = pd.to_numeric(data3.iloc[:, 2], errors="coerce")

# Keep only rows where both are valid numbers
clean_data = pd.DataFrame({
    "day": days_data,
    "infected": infected_data
}).dropna()

days_data = clean_data["day"].to_numpy(dtype=float)
infected_data = clean_data["infected"].to_numpy(dtype=float)

print("First few rows of Release #3 data:")
print(data3.head())

print("\nCleaned data preview:")
print(clean_data.head())

print("\nNumber of usable data points:", len(days_data))
print("First few days:", days_data[:5])
print("First few active infection values:", infected_data[:5])

# Find actual peak from Release #3
actual_peak_index = np.argmax(infected_data)
actual_peak_day = days_data[actual_peak_index]
actual_peak_value = infected_data[actual_peak_index]

# ---------------------------
# Plot Release #3 data vs model prediction
# ---------------------------

plt.figure(figsize=(8,5))

# Plot full dataset (Release #3)
plt.scatter(days_data, infected_data, 
            color="black", 
            label="Release #3 Observed Data")

# Plot prediction from Release #2 model
plt.plot(future_timepoints, new_cases_future, 
         color="red", 
         linewidth=2,
         label="Prediction from Release #2 Model")

# Labels and title
plt.xlabel("Day")
plt.ylabel("Daily New Cases")
plt.title("Full Dataset (Release #3) vs SEIR Prediction from Release #2")
plt.axvline(peak_day, linestyle="--", label="Predicted Peak Day")
plt.axvline(actual_peak_day, linestyle=":", label="Actual Peak Day")

plt.legend()
plt.show()

print("\nActual peak from Release #3:")
print("Actual peak day =", actual_peak_day)
print("Actual peak daily new cases =", actual_peak_value)

# Compute relative errors
error_y = abs(peak_value - actual_peak_value) / actual_peak_value * 100
error_x = abs(peak_day - actual_peak_day) / actual_peak_day * 100

print("\nRelative errors:")
print("Relative error in peak cases (y) =", error_y, "%")
print("Relative error in peak day (x) =", error_x, "%")


# %%
#==========================================
##Intervention Methods
#==========================================

