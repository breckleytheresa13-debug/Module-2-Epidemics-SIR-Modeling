#%%

# ============================================================
# Check-in 2: SEIR Model with Euler's Method + SSE + Grid Search
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
# Initial Conditions
# ---------------------------
N = 1000
I0 = infected_data[0]
E0 = 5
R0_init = 0
S0 = N - E0 - I0 - R0_init

timepoints = days_data


# ---------------------------
# Initial Parameter Guess
# ---------------------------
beta_guess = 0.45
sigma_guess = 0.20
gamma_guess = 0.12

S_guess, E_guess, I_guess, R_guess = euler_seir(
    beta_guess, sigma_guess, gamma_guess,
    S0, E0, I0, R0_init,
    timepoints, N
)

sse_guess = calculate_sse(infected_data, I_guess)

print("\nInitial parameter guess results:")
print("beta =", beta_guess)
print("sigma =", sigma_guess)
print("gamma =", gamma_guess)
print("SSE =", sse_guess)


# ---------------------------
# Plot Initial Guess
# ---------------------------
plt.figure(figsize=(8, 5))
plt.scatter(days_data, infected_data, label="Observed active infections")
plt.plot(timepoints, I_guess, label="SEIR model I(t)")
plt.xlabel("Day")
plt.ylabel("Active infections")
plt.title("Release #2: Data vs Initial SEIR Model")
plt.legend()
plt.show()


# ---------------------------
# Grid Search for Best Fit
# ---------------------------
beta_vals = np.linspace(0.2, 0.8, 25)
sigma_vals = np.linspace(0.05, 0.4, 25)
gamma_vals = np.linspace(0.05, 0.3, 25)

best_sse = np.inf
best_beta = None
best_sigma = None
best_gamma = None
best_I = None

for beta in beta_vals:
    for sigma in sigma_vals:
        for gamma in gamma_vals:
            S_try, E_try, I_try, R_try = euler_seir(
                beta, sigma, gamma,
                S0, E0, I0, R0_init,
                timepoints, N
            )

            sse = calculate_sse(infected_data, I_try)

            if sse < best_sse:
                best_sse = sse
                best_beta = beta
                best_sigma = sigma
                best_gamma = gamma
                best_I = I_try

print("\nBest-fit parameters from grid search:")
print("Best beta =", best_beta)
print("Best sigma =", best_sigma)
print("Best gamma =", best_gamma)
print("Lowest SSE =", best_sse)


# ---------------------------
# Plot Best-Fit Model
# ---------------------------
plt.figure(figsize=(8, 5))
plt.scatter(days_data, infected_data, label="Observed active infections")
plt.plot(timepoints, best_I, label="Best-fit SEIR model I(t)")
plt.xlabel("Day")
plt.ylabel("Active infections")
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

S_future, E_future, I_future, R_future = euler_seir(
    best_beta, best_sigma, best_gamma,
    S0, E0, I0, R0_init,
    future_timepoints, N
)

peak_index = np.argmax(I_future)
peak_day = future_timepoints[peak_index]
peak_value = I_future[peak_index]

print("\nFuture outbreak prediction:")
print("Predicted peak day =", peak_day)
print("Predicted peak active infections =", peak_value)


# ---------------------------
# Plot Future Projection
# ---------------------------
plt.figure(figsize=(8, 5))
plt.scatter(days_data, infected_data, label="Observed active infections")
plt.plot(future_timepoints, I_future, label="Projected SEIR I(t)")
plt.axvline(peak_day, linestyle="--", label=f"Peak day = {peak_day}")
plt.xlabel("Day")
plt.ylabel("Active infections")
plt.title("SEIR Projection and Predicted Peak")
plt.legend()
plt.show()  