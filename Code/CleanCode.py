

#%%
# Helper Functions
#==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loads day values and active-case values from a CSV file
def load_active_case_data(filename):
    data = pd.read_csv(filename)

    clean_data = pd.DataFrame({
        "day": pd.to_numeric(data.iloc[:, 0], errors="coerce"),
        "active_cases": pd.to_numeric(data.iloc[:, 2], errors="coerce")
    }).dropna()

    # arrays to be used in the model and plots
    days = clean_data["day"].to_numpy(dtype=float)
    active_cases = clean_data["active_cases"].to_numpy(dtype=float)

    return days, active_cases, data

def calculate_sse(observed, predicted):
    return np.sum((observed - predicted) ** 2)

# SEIR model using Euler's Method
def seir_model(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N):
    dt = timepoints[1] - timepoints[0]

    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))

    # starting values
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

# %%
# Data Release #1
# ================================

days1, active1, raw1 = load_active_case_data("C:\\Users\\15712\\OneDrive - University of Virginia\\Comp Mod 2\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#1.csv")

plt.figure(figsize=(8, 5))
plt.plot(days1, active1, marker="o")
plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("Mystery Virus — Data Release #1")
plt.grid(True)
plt.show()

fit_days = days1[19:]
fit_cases = active1[19:]

# This finds the exponential growth rate from the log-transformed data
coeffs = np.polyfit(fit_days, np.log(fit_cases), 1)
r, log_a = coeffs
a = np.exp(log_a)
fit_line = a * np.exp(r * days1)

plt.figure(figsize=(8, 5))
plt.plot(days1, active1, "o-", label="Observed active cases")
plt.plot(days1, fit_line, "--", label=f"Exponential fit (r = {r:.4f})")
plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("Release #1 with Exponential Fit")
plt.legend()
plt.grid(True)
plt.show()

doubling_time = np.log(2) / r
print("Growth rate r =", r)
print("Doubling time =", doubling_time, "days")

# %%
# Data Release #2
# ================================

days2, active2, raw2 = load_active_case_data("C:\\Users\\15712\\OneDrive - University of Virginia\\Comp Mod 2\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#2.csv")

print("First few rows of Release #2:")
print(raw2.head())

print("\nNumber of usable data points:", len(days2))
print("First few days:", days2[:5])
print("First few active-case values:", active2[:5])

# starting assumptions for  population and compartments
N = 10000
I0 = active2[0]
E0 = I0
R0_initial = 0
S0 = N - E0 - I0 - R0_initial

timepoints2 = days2

# initial parameter guesses before optimization
beta_guess = 0.40
sigma_guess = 0.25
gamma_guess = 0.10

S_guess, E_guess, I_guess, R_guess = seir_model(
    beta_guess, sigma_guess, gamma_guess,
    S0, E0, I0, R0_initial,
    timepoints2, N
)

sse_guess = calculate_sse(active2, I_guess)

print("Initial parameter guess results:")
print("beta =", beta_guess)
print("sigma =", sigma_guess)
print("gamma =", gamma_guess)
print("SSE =", sse_guess)
print("Peak predicted active cases =", np.max(I_guess))
print("Peak observed active cases =", np.max(active2))

plt.figure(figsize=(8, 5))
plt.scatter(days2, active2, label="Observed active cases")
plt.plot(timepoints2, I_guess, label="SEIR predicted active cases")
plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("Release #2: Data vs Initial SEIR Model")
plt.legend()
plt.grid(True)
plt.show()

# Data Release #2 — Grid Search Optimization
# ================================

beta_vals = np.linspace(0.15, 0.90, 40)
sigma_vals = np.linspace(0.08, 0.50, 35)
gamma_vals = np.linspace(0.03, 0.20, 35)
E0_multipliers = [1, 1.5, 2, 3, 4]

# These variables keep track of the best fit found so far
best_sse = np.inf
best_beta = None
best_sigma = None
best_gamma = None
best_E0 = None
best_I = None

for E_mult in E0_multipliers:
    E0_try = E_mult * I0
    S0_try = N - E0_try - I0 - R0_initial

    for beta in beta_vals:
        for sigma in sigma_vals:
            for gamma in gamma_vals:
                S_try, E_try, I_try, R_try = seir_model(
                    beta, sigma, gamma,
                    S0_try, E0_try, I0, R0_initial,
                    timepoints2, N
                )

                sse = calculate_sse(active2, I_try)

                if sse < best_sse:
                    best_sse = sse
                    best_beta = beta
                    best_sigma = sigma
                    best_gamma = gamma
                    best_E0 = E0_try
                    best_I = I_try

print("Best-fit parameters:")
print("Best beta =", best_beta)
print("Best sigma =", best_sigma)
print("Best gamma =", best_gamma)
print("Best E0 =", best_E0)
print("Lowest SSE =", best_sse)
print("Peak predicted active cases =", np.max(best_I))
print("Peak observed active cases =", np.max(active2))

plt.figure(figsize=(8, 5))
plt.scatter(days2, active2, label="Observed active cases")
plt.plot(timepoints2, best_I, label="Best-fit SEIR predicted active cases")
plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("Best-Fit SEIR Model for Release #2")
plt.legend()
plt.grid(True)
plt.show()

# These values help interpret the meaning of the best-fit parameters
R0_model = best_beta / best_gamma
latent_period = 1 / best_sigma
infectious_period = 1 / best_gamma

print("Model interpretation:")
print("Estimated R0 =", R0_model)
print("Estimated latent period (days) =", latent_period)
print("Estimated infectious period (days) =", infectious_period)

# This rebuilds the starting susceptible population using the best E0 value
S0_best = N - best_E0 - I0 - R0_initial

future_timepoints = np.arange(1, 180, 1)

S_future, E_future, I_future, R_future = seir_model(
    best_beta, best_sigma, best_gamma,
    S0_best, best_E0, I0, R0_initial,
    future_timepoints, N
)

# This finds the predicted peak from the future simulation
peak_index = np.argmax(I_future)
peak_day = future_timepoints[peak_index]
peak_value = I_future[peak_index]

print("Future outbreak prediction:")
print("Predicted peak day =", peak_day)
print("Predicted peak active cases =", peak_value)

# This plots the future prediction and marks the predicted peak day
plt.figure(figsize=(8, 5))
plt.scatter(days2, active2, label="Observed active cases")
plt.plot(future_timepoints, I_future, label="Projected active cases")
plt.axvline(peak_day, linestyle="--", label=f"Peak day = {peak_day}")
plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("SEIR Projection and Predicted Peak")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Data Release #3
# ================================

# This loads the third release of active-case data
days3, active3, raw3 = load_active_case_data("C:\\Users\\15712\\OneDrive - University of Virginia\\Comp Mod 2\\Module-2-Epidemics-SIR-Modeling\\Data\\mystery_virus_daily_active_counts_RELEASE#3.csv")

# These print statements help verify the data
print("First few rows of Release #3:")
print(raw3.head())

print("\nNumber of usable data points:", len(days3))
print("First few days:", days3[:5])
print("First few active-case values:", active3[:5])

# This finds the actual peak in Release #3
actual_peak_index = np.argmax(active3)
actual_peak_day = days3[actual_peak_index]
actual_peak_value = active3[actual_peak_index]

# This plots the Release #3 data against the prediction made from Release #2
plt.figure(figsize=(8, 5))
plt.scatter(days3, active3, color="black", label="Release #3 observed active cases")
plt.plot(future_timepoints, I_future, color="red", linewidth=2, label="Prediction from Release #2 model")
plt.axvline(peak_day, linestyle="--", label="Predicted peak day")
plt.axvline(actual_peak_day, linestyle=":", label="Actual peak day")
plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("Release #3 vs Prediction from Release #2")
plt.legend()
plt.grid(True)
plt.show()

# This prints the actual peak values from Release #3
print("Actual peak from Release #3:")
print("Actual peak day =", actual_peak_day)
print("Actual peak active cases =", actual_peak_value)

# These calculate the relative errors in peak size and peak timing
error_y = abs(peak_value - actual_peak_value) / actual_peak_value * 100
error_x = abs(peak_day - actual_peak_day) / actual_peak_day * 100

print("\nRelative errors:")
print("Relative error in peak active cases =", error_y, "%")
print("Relative error in peak day =", error_x, "%")


# %%
# Data Release #3 — Curve of Best Fit
# ================================

from scipy.optimize import curve_fit

# This defines a Gaussian function to model the epidemic curve
def gaussian(x, a, b, c):
    return a * np.exp(-((x - b)**2) / (2 * c**2))

# Initial guesses for the Gaussian parameters
a_guess = np.max(active3)
b_guess = days3[np.argmax(active3)]
c_guess = 15

initial_guess = [a_guess, b_guess, c_guess]

# Fit the Gaussian model to the Release #3 data
params, covariance = curve_fit(gaussian, days3, active3, p0=initial_guess)

a_fit, b_fit, c_fit = params

print("Gaussian fit parameters:")
print("Peak height =", a_fit)
print("Peak day =", b_fit)
print("Spread parameter =", c_fit)

# Generate smooth curve for plotting
smooth_days = np.linspace(min(days3), max(days3), 300)
gaussian_fit = gaussian(smooth_days, a_fit, b_fit, c_fit)

# Plot the observed data and Gaussian best fit
plt.figure(figsize=(8,5))

plt.scatter(days3, active3, color="black", label="Observed active cases")
plt.plot(smooth_days, gaussian_fit, color="blue", linewidth=2, label="Gaussian best-fit curve")

plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("Best-Fit Curve for Data Release #3")
plt.legend()
plt.grid(True)

plt.show()

# %%
# Intervention Strategy 1 — Mandated Masking
# ==========================================

# Masking reduces transmission rate beta by 40% after day 70
beta_mask = best_beta * 0.6

# Custom SEIR model with intervention starting at day 70
def seir_with_masking(beta, beta_mask, sigma, gamma, S0, E0, I0, R0, timepoints, N):

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

        # Switch transmission rate after day 70
        if timepoints[n] >= 70:
            beta_current = beta_mask
        else:
            beta_current = beta

        dSdt = -beta_current * S[n] * I[n] / N
        dEdt = beta_current * S[n] * I[n] / N - sigma * E[n]
        dIdt = sigma * E[n] - gamma * I[n]
        dRdt = gamma * I[n]

        S[n+1] = S[n] + dSdt * dt
        E[n+1] = E[n] + dEdt * dt
        I[n+1] = I[n] + dIdt * dt
        R[n+1] = R[n] + dRdt * dt

    return S, E, I, R


# Run simulation with masking intervention
S_mask, E_mask, I_mask, R_mask = seir_with_masking(
    best_beta, beta_mask, best_sigma, best_gamma,
    S0_best, best_E0, I0, R0_initial,
    future_timepoints, N
)


# Plot comparison: actual outbreak vs masking intervention
plt.figure(figsize=(8,5))

plt.scatter(days3, active3, color="black", label="Observed Release #3 Data")

plt.plot(future_timepoints, I_future,
         color="red", linewidth=2,
         label="Original Model (No Intervention)")

plt.plot(future_timepoints, I_mask,
         color="blue", linewidth=2,
         label="Mandated Masking at Day 70")

plt.axvline(70, linestyle="--", color="gray", label="Intervention Start (Day 70)")

plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("Effect of Mandated Masking on Epidemic Curve")

plt.legend()
plt.grid(True)

plt.show()