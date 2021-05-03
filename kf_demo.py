import time

import matplotlib.pyplot as plt
import numpy as np

# Sampling Time
dt = 0.1

# Initial Values
x_0 = 0
v_0 = 0
a_base = 0.3
a_bias = 0.05

# Noises
sigma_x = 0.01
sigma_v = 0.01
sigma_a = 0.01
sigma_obs_x = 3.0
sigma_obs_x_real = 1.0

# Variables
I = np.eye(2)

F = np.matrix(
    [
        [1, dt],
        [0, 1],
    ]
)

B = np.matrix(
    [
        [(dt ** 2) / 2],
        [dt],
    ]
)

G = I

w = np.matrix(
    [
        [sigma_x],
        [sigma_v],
    ]
)

W = np.matrix(
    [
        [sigma_x ** 2, 0],
        [0, sigma_v ** 2],
    ]
)

Q = np.matrix(
    [
        [sigma_a ** 2],
    ]
)

H = np.matrix(
    [
        [1, 0],
    ]
)

R = np.matrix(
    [
        [sigma_obs_x ** 2],
    ]
)

# Functions
def get_true_x(t):
    x_true = x_0 + v_0 * t + a_base / 2 * (t ** 2)
    v_true = v_0 + a_base * t

    return np.matrix(
        [
            [x_true],
            [v_true],
        ]
    )


def measurement(x_true):
    [x, v] = np.array(x_true)[:, 0]

    x_obs = x + np.random.normal(0, sigma_obs_x_real)

    return np.matrix(
        [
            [x_obs],
        ]
    )


def get_input():
    return np.matrix(
        [
            [a_base + a_bias + np.random.normal(0, sigma_a)],
        ]
    )


def predict(x_prev, P, u, dt):
    x_predict = F @ x_prev + B @ u
    P_predict = F @ P @ F.T + B @ Q @ B.T + G @ W @ G.T

    return x_predict, P_predict


def update(x_predict, P_predict, z):
    e = z - H @ x_predict
    S = R + H @ P_predict @ H.T
    K = P_predict @ H.T @ np.linalg.inv(S)
    x_est = x_predict + K @ e
    P_est = (I - K @ H) @ P_predict

    return x_est, P_est


def demo():
    # Initialize data
    x = np.matrix(
        [
            [x_0],
            [v_0],
        ]
    )

    P = np.matrix(
        [
            [0.5, 0],
            [0, 0.5],
        ]
    )

    # Dead Reckoning data only for testing
    x_dr = x
    P_dr = P

    # Buffer
    ts = []
    xs_true = []
    xs_est = []
    xs_obs = []
    xs_dr = []

    # Initialize figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([0, 10])
    ax.set_ylim([-5, 20])
    plot_x_true = ax.plot([], [], label="x_true")[0]
    plot_x_est = ax.plot([], [], label="x_est")[0]
    plot_x_obs = ax.plot([], [], label="x_obs")[0]
    plot_x_dr = ax.plot([], [], label="x_dr")[0]
    ax.legend()

    # Show figure in interactive mode
    plt.ion()
    plt.show()

    # Loop
    for i in range(100):
        # Calculate ground truth
        t = i * dt
        x_true = get_true_x(t)

        # Predict
        u = get_input()
        [x_predict, P_predict] = predict(x, P, u, dt)
        [x_dr, P_dr] = predict(x_dr, P_dr, u, dt)

        # Update
        z = measurement(x_true)
        [x, P] = update(x_predict, P_predict, z)

        # Add data for plot
        ts.append(t)
        xs_true.append(x_true[0].item())
        xs_est.append(x[0].item())
        xs_dr.append(x_dr[0].item())
        xs_obs.append(z[0].item())

        # Plot
        plot_x_true.set_data(ts, xs_true)
        plot_x_est.set_data(ts, xs_est)
        plot_x_obs.set_data(ts, xs_obs)
        plot_x_dr.set_data(ts, xs_dr)

        # Update figure
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.01)

    # Wait for window close
    plt.ioff()
    plt.show()


demo()
