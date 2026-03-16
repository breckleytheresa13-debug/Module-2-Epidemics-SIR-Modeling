

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
# Mandated Masking Intervention at Day 70
# ================================================================

# This creates a best-fit SEIR model using Release #3 data
beta_vals_3 = np.linspace(0.2, 0.8, 15)
sigma_vals_3 = np.linspace(0.1, 0.4, 15)
gamma_vals_3 = np.linspace(0.04, 0.15, 15)
E0_multipliers_3 = [1, 2, 3]

# These variables store the best Release #3 fit
best_sse_3 = np.inf
best_beta_3 = None
best_sigma_3 = None
best_gamma_3 = None
best_E0_3 = None
best_I_3 = None

# These are the initial conditions based on Release #3
I0_3 = active3[0]
R0_3 = 0
N_3 = N

# This searches for the best-fit SEIR curve for Release #3
for E_mult in E0_multipliers_3:
    E0_try_3 = E_mult * I0_3
    S0_try_3 = N_3 - E0_try_3 - I0_3 - R0_3

    for beta in beta_vals_3:
        for sigma in sigma_vals_3:
            for gamma in gamma_vals_3:
                S_try_3, E_try_3, I_try_3, R_try_3 = seir_model(
                    beta, sigma, gamma,
                    S0_try_3, E0_try_3, I0_3, R0_3,
                    days3, N_3
                )

                sse_3 = calculate_sse(active3, I_try_3)

                if sse_3 < best_sse_3:
                    best_sse_3 = sse_3
                    best_beta_3 = beta
                    best_sigma_3 = sigma
                    best_gamma_3 = gamma
                    best_E0_3 = E0_try_3
                    best_I_3 = I_try_3

# This prints the best-fit Release #3 model parameters
print("Best-fit Release #3 parameters:")
print("Best beta =", best_beta_3)
print("Best sigma =", best_sigma_3)
print("Best gamma =", best_gamma_3)
print("Best E0 =", best_E0_3)
print("Lowest SSE =", best_sse_3)

# This rebuilds the starting susceptible population for the Release #3 model
S0_best_3 = N_3 - best_E0_3 - I0_3 - R0_3

# This creates a longer time range so the intervention curve can be seen clearly
future_timepoints_3 = np.arange(1, 161, 1)

# This generates the original best-fit Release #3 model with no intervention
S_base_3, E_base_3, I_base_3, R_base_3 = seir_model(
    best_beta_3, best_sigma_3, best_gamma_3,
    S0_best_3, best_E0_3, I0_3, R0_3,
    future_timepoints_3, N_3
)

# Masking reduces transmission by 40 percent, so beta becomes 60 percent of its original value
beta_mask_3 = 0.6 * best_beta_3

# This function applies masking starting at day 70
def seir_with_masking(beta_before, beta_after, sigma, gamma, S0, E0, I0, R0, timepoints, N):
    dt = timepoints[1] - timepoints[0]

    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))

    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    # This updates the model one day at a time and switches beta at day 70
    for n in range(len(timepoints) - 1):
        if timepoints[n] >= 70:
            beta_current = beta_after
        else:
            beta_current = beta_before

        dSdt = -beta_current * S[n] * I[n] / N
        dEdt = beta_current * S[n] * I[n] / N - sigma * E[n]
        dIdt = sigma * E[n] - gamma * I[n]
        dRdt = gamma * I[n]

        S[n + 1] = S[n] + dSdt * dt
        E[n + 1] = E[n] + dEdt * dt
        I[n + 1] = I[n] + dIdt * dt
        R[n + 1] = R[n] + dRdt * dt

    return S, E, I, R

# This generates the masking intervention curve based on the Release #3 best-fit model
S_mask_3, E_mask_3, I_mask_3, R_mask_3 = seir_with_masking(
    best_beta_3, beta_mask_3, best_sigma_3, best_gamma_3,
    S0_best_3, best_E0_3, I0_3, R0_3,
    future_timepoints_3, N_3
)

# This plots the Release #3 best-fit curve and the masking intervention curve
plt.figure(figsize=(8, 5))

plt.plot(future_timepoints_3, I_base_3, linewidth=2, label="Release #3 Best-Fit Curve")
plt.plot(future_timepoints_3, I_mask_3, linewidth=2, label="Mandated Masking at Day 70")
plt.axvline(70, linestyle="--", color="gray", label="Intervention Start (Day 70)")

plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("Effect of Mandated Masking on Release #3 Best-Fit Curve")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Vaccine Campaign Intervention
# ================================================================

# This vaccine campaign moves vaccinated susceptible people into the recovered group on day 70
vaccinated_students = 2000
vaccine_efficacy = 0.90
effective_vaccinated = vaccinated_students * vaccine_efficacy

# This function applies a one-time vaccination event on day 70
def seir_with_vaccination(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N):
    dt = timepoints[1] - timepoints[0]

    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))

    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    vaccination_applied = False

    # This updates the model one step at a time and applies vaccination once at day 70
    for n in range(len(timepoints) - 1):
        dSdt = -beta * S[n] * I[n] / N
        dEdt = beta * S[n] * I[n] / N - sigma * E[n]
        dIdt = sigma * E[n] - gamma * I[n]
        dRdt = gamma * I[n]

        S[n + 1] = S[n] + dSdt * dt
        E[n + 1] = E[n] + dEdt * dt
        I[n + 1] = I[n] + dIdt * dt
        R[n + 1] = R[n] + dRdt * dt

        # This applies the vaccine campaign one time on day 70
        if timepoints[n + 1] >= 70 and not vaccination_applied:
            vaccinated_now = min(effective_vaccinated, S[n + 1])
            S[n + 1] = S[n + 1] - vaccinated_now
            R[n + 1] = R[n + 1] + vaccinated_now
            vaccination_applied = True

    return S, E, I, R

# This generates the vaccine intervention curve based on the Release #3 best-fit model
S_vax_3, E_vax_3, I_vax_3, R_vax_3 = seir_with_vaccination(
    best_beta_3, best_sigma_3, best_gamma_3,
    S0_best_3, best_E0_3, I0_3, R0_3,
    future_timepoints_3, N_3
)

# This plots the Release #3 best-fit curve and the vaccine intervention curve
plt.figure(figsize=(8, 5))

plt.plot(future_timepoints_3, I_base_3, linewidth=2, label="Release #3 Best-Fit Curve")
plt.plot(future_timepoints_3, I_vax_3, linewidth=2, label="Vaccine Campaign at Day 70")
plt.axvline(70, linestyle="--", color="gray", label="Intervention Start (Day 70)")

plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("Effect of Vaccine Campaign on Release #3 Best-Fit Curve")
plt.legend()
plt.grid(True)
plt.show()

# %%
# School Closure Intervention
# ===============================================================

# School closure reduces contact rate to 20 percent of normal for 2 weeks
beta_closure = 0.2 * best_beta_3

# This function applies a temporary school closure from day 70 to day 83
def seir_with_school_closure(beta_before, beta_during, sigma, gamma, S0, E0, I0, R0, timepoints, N):
    dt = timepoints[1] - timepoints[0]

    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))

    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    # This updates the model one day at a time and changes beta only during the closure period
    for n in range(len(timepoints) - 1):
        if 70 <= timepoints[n] < 84:
            beta_current = beta_during
        else:
            beta_current = beta_before

        dSdt = -beta_current * S[n] * I[n] / N
        dEdt = beta_current * S[n] * I[n] / N - sigma * E[n]
        dIdt = sigma * E[n] - gamma * I[n]
        dRdt = gamma * I[n]

        S[n + 1] = S[n] + dSdt * dt
        E[n + 1] = E[n] + dEdt * dt
        I[n + 1] = I[n] + dIdt * dt
        R[n + 1] = R[n] + dRdt * dt

    return S, E, I, R

# This generates the school closure intervention curve based on the Release #3 best-fit model
S_close_3, E_close_3, I_close_3, R_close_3 = seir_with_school_closure(
    best_beta_3, beta_closure, best_sigma_3, best_gamma_3,
    S0_best_3, best_E0_3, I0_3, R0_3,
    future_timepoints_3, N_3
)

# This plots the Release #3 best-fit curve and the school closure intervention curve
plt.figure(figsize=(8, 5))

plt.plot(future_timepoints_3, I_base_3, linewidth=2, label="Release #3 Best-Fit Curve")
plt.plot(future_timepoints_3, I_close_3, linewidth=2, label="2-Week School Closure at Day 70")
plt.axvline(70, linestyle="--", color="gray", label="Closure Starts (Day 70)")
plt.axvline(84, linestyle=":", color="gray", label="Closure Ends (Day 84)")

plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("Effect of School Closure on Release #3 Best-Fit Curve")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Combined Intervention - Mandated Masking + Vaccine Campaign + School Closure
# =======================================

# This combines masking, vaccination, and school closure starting at day 70
vaccinated_students_combo = 2000
vaccine_efficacy_combo = 0.90
effective_vaccinated_combo = vaccinated_students_combo * vaccine_efficacy_combo

# This sets the reduced transmission rates for the intervention periods
beta_mask_only = 0.6 * best_beta_3
beta_mask_and_closure = 0.12 * best_beta_3

# This function applies all three interventions together
def seir_with_combined_intervention(beta_normal, beta_mask_after, beta_mask_closure, sigma, gamma, S0, E0, I0, R0, timepoints, N):
    dt = timepoints[1] - timepoints[0]

    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))

    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    vaccination_applied = False

    # This updates the model one day at a time and applies all interventions starting at day 70
    for n in range(len(timepoints) - 1):
        if 70 <= timepoints[n] < 84:
            beta_current = beta_mask_closure
        elif timepoints[n] >= 84:
            beta_current = beta_mask_after
        else:
            beta_current = beta_normal

        dSdt = -beta_current * S[n] * I[n] / N
        dEdt = beta_current * S[n] * I[n] / N - sigma * E[n]
        dIdt = sigma * E[n] - gamma * I[n]
        dRdt = gamma * I[n]

        S[n + 1] = S[n] + dSdt * dt
        E[n + 1] = E[n] + dEdt * dt
        I[n + 1] = I[n] + dIdt * dt
        R[n + 1] = R[n] + dRdt * dt

        # This applies the vaccine campaign one time on day 70
        if timepoints[n + 1] >= 70 and not vaccination_applied:
            vaccinated_now = min(effective_vaccinated_combo, S[n + 1])
            S[n + 1] = S[n + 1] - vaccinated_now
            R[n + 1] = R[n + 1] + vaccinated_now
            vaccination_applied = True

    return S, E, I, R

# This generates the combined intervention curve based on the Release #3 best-fit model
S_combo_3, E_combo_3, I_combo_3, R_combo_3 = seir_with_combined_intervention(
    best_beta_3, beta_mask_only, beta_mask_and_closure,
    best_sigma_3, best_gamma_3,
    S0_best_3, best_E0_3, I0_3, R0_3,
    future_timepoints_3, N_3
)

# This plots the Release #3 best-fit curve and the combined intervention curve
plt.figure(figsize=(8, 5))

plt.plot(future_timepoints_3, I_base_3, linewidth=2, label="Release #3 Best-Fit Curve")
plt.plot(future_timepoints_3, I_combo_3, linewidth=2, label="Combined Intervention at Day 70")
plt.axvline(70, linestyle="--", color="gray", label="Intervention Start (Day 70)")
plt.axvline(84, linestyle=":", color="gray", label="School Reopens (Day 84)")

plt.xlabel("Day")
plt.ylabel("Active Cases")
plt.title("Effect of Combined Intervention on Release #3 Best-Fit Curve")
plt.legend()
plt.grid(True)
plt.show()
# %%
