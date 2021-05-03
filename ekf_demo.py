import time

import matplotlib.pyplot as plt
import numpy as np

# Sampling Time
dt = 0.1

# Initial Values
x_0 = 10
y_0 = 0
theta_0 = np.deg2rad(90)

r = 10

v = 5
v_bias = 0.5
v_obs = v + v_bias

omega = v / r
omega_bias = np.deg2rad(0.5)
omega_obs = omega + omega_bias

# Noises
sigma_x = 0.05
sigma_y = 0.05
sigma_theta = 0.05
sigma_gnss = 3.0
sigma_gnss_real = 1.0
sigma_lidar_r = 1.0
sigma_lidar_phi = np.deg2rad(1.0)
sigma_lidar_r_real = 0.1
sigma_lidar_phi_real = np.deg2rad(0.1)

# Variables
I = np.eye(3)

G = I

w = np.matrix(
    [
        [sigma_x],
        [sigma_y],
        [sigma_theta],
    ]
)

Q = np.diag(
    [
        sigma_x ** 2,
        sigma_y ** 2,
        sigma_theta ** 2,
    ]
)

R_gnss = np.diag(
    [
        sigma_gnss ** 2,
        sigma_gnss ** 2,
    ]
)

R_lidar = np.diag(
    [
        sigma_lidar_r ** 2,
        sigma_lidar_phi ** 2,
    ]
)

# Functions
def normalize_radian(rad):
    while rad < -np.pi:
        rad += 2 * np.pi
    while rad >= np.pi:
        rad -= 2 * np.pi
    return rad


def get_true_x(t):
    theta_true = omega * t
    r = v / omega
    x_true = r * np.cos(theta_true)
    y_true = r * np.sin(theta_true)

    return np.matrix(
        [
            [x_true],
            [y_true],
            [theta_true],
        ]
    )


def measurement_gnss(x_true, x_predict):
    [x, y, theta] = np.array(x_true)[:, 0]

    H = np.matrix(
        [
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

    x_obs = x + np.random.normal(0, sigma_gnss_real)
    y_obs = y + np.random.normal(0, sigma_gnss_real)
    z = np.matrix(
        [
            [x_obs],
            [y_obs],
        ]
    )

    e = z - H @ x_predict

    return e, H, z


def measurement_lidar(x_true, x_predict):
    [x, y, theta] = np.array(x_true)[:, 0]

    [x_, y_, theta_] = np.array(x_predict)[:, 0]
    H = np.matrix(
        [
            [x_ / np.hypot(x_, y_), y_ / np.hypot(x_, y_), 0],
            [-y_ / (x_ ** 2 + y_ ** 2), x_ / (x_ ** 2 + y_ ** 2), 0],
        ]
    )

    r_obs = np.hypot(x, y) + np.random.normal(0, sigma_lidar_r_real)
    phi_obs = np.arctan2(y, x) + np.random.normal(0, sigma_lidar_phi_real)
    z = np.matrix(
        [
            [r_obs],
            [phi_obs],
        ]
    )

    e = z - np.matrix(
        [
            [np.hypot(x_, y_)],
            [np.arctan2(y_, x_)],
        ]
    )

    # Normalize angle
    e[1, 0] = normalize_radian(e[1, 0])

    return e, H, z


def predict(x_prev, P, dt):
    theta = x_prev[2].item()

    F = np.matrix(
        [
            [1, 0, -v_obs * np.sin(theta) * dt],
            [0, 1, v_obs * np.cos(theta) * dt],
            [0, 0, 1],
        ]
    )

    x_predict = x_prev + np.matrix(
        [
            [v_obs * np.cos(theta) * dt],
            [v_obs * np.sin(theta) * dt],
            [omega_obs * dt],
        ]
    )

    # Normalize angle
    x_predict[2, 0] = normalize_radian(x_predict[2, 0])

    P_predict = F @ P @ F.T + G @ Q @ G.T

    return x_predict, P_predict


def update(x_predict, P_predict, e, H, z, R):
    S = H @ P_predict @ H.T + R
    K = P_predict @ H.T @ np.linalg.inv(S)
    x_est = x_predict + K @ e
    P_est = (I - K @ H) @ P_predict

    # Normalize angle
    x_est[2, 0] = normalize_radian(x_est[2, 0])

    return x_est, P_est


def demo():
    # Initialize data
    x = np.matrix(
        [
            [x_0],
            [y_0],
            [theta_0],
        ]
    )

    P = np.matrix(
        [
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5],
        ]
    )

    # Dead Reckoning data only for testing
    x_dr = x
    P_dr = P

    # Initialize figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    plot_marker_true = ax.plot([], [], "^", label="true")[0]
    plot_marker_est = ax.plot([], [], "^", label="est")[0]
    plot_marker_gnss = ax.plot([], [], "^", label="gnss")[0]
    plot_marker_lidar = ax.plot([], [], "^", label="lidar")[0]
    plot_marker_dr = ax.plot([], [], "^", label="dr")[0]
    plot_ellipse_est = ax.plot([], [], linestyle="dashed")[0]
    ax.legend()

    # Plot reference circle
    ts_circle = np.linspace(0, 2.0 * np.pi, 100)
    ax.plot(r * np.cos(ts_circle), r * np.sin(ts_circle), linestyle="dashed")

    # Show figure in interactive mode
    plt.ion()
    plt.show()

    # Loop
    for i in range(500):
        # Calculate ground truth
        t = i * dt
        x_true = get_true_x(t)

        # Predict
        [x_predict, P_predict] = predict(x, P, dt)
        [x_dr, P_dr] = predict(x_dr, P_dr, dt)
        [x, P] = x_predict, P_predict

        # Update by GNSS
        z_gnss = None
        use_gnss = True
        if use_gnss:
            if (i % 2) == 0:
                [e_gnss, H_gnss, z_gnss] = measurement_gnss(x_true, x_predict)
                [x, P] = update(x_predict, P_predict, e_gnss, H_gnss, z_gnss, R_gnss)

        # Update by LiDAR
        z_lidar = None
        use_lidar = True
        if use_lidar:
            if (i % 5) == 0:
                [e_lidar, H_lidar, z_lidar] = measurement_lidar(x_true, x_predict)
                [x, P] = update(x_predict, P_predict, e_lidar, H_lidar, z_lidar, R_lidar)

        # Prepare plot data
        if z_gnss is not None:
            x_gnss = z_gnss[0, 0]
            y_gnss = z_gnss[1, 0]
        else:
            x_gnss = []
            y_gnss = []

        if z_lidar is not None:
            # Convert LiDAR data to xy for debug
            [r_obs, phi_obs] = np.array(z_lidar[:, 0])
            x_lidar = r_obs * np.cos(phi_obs)
            y_lidar = r_obs * np.sin(phi_obs)
        else:
            x_lidar = []
            y_lidar = []

        # Plot markers
        plot_marker_true.set_data(x_true[0, 0], x_true[1, 0])
        plot_marker_est.set_data(x[0, 0], x[1, 0])
        plot_marker_gnss.set_data(x_gnss, y_gnss)
        plot_marker_lidar.set_data(x_lidar, y_lidar)
        plot_marker_dr.set_data(x_dr[0, 0], x_dr[1, 0])

        # Plot error ellipses
        x_ellipse_center = x[0, 0]
        y_ellipse_center = x[1, 0]
        sigma_x = np.sqrt(P[0, 0])
        sigma_y = np.sqrt(P[1, 1])
        larger_sigma = np.max([sigma_x, sigma_y])
        ts_ellipse = np.linspace(0, 2.0 * np.pi, 100)
        xs_ellipse = x_ellipse_center + 3 * larger_sigma * np.cos(ts_ellipse)
        ys_ellipse = y_ellipse_center + 3 * larger_sigma * np.sin(ts_ellipse)
        plot_ellipse_est.set_data(xs_ellipse, ys_ellipse)

        # Update figure
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)

    # Wait for window close
    plt.ioff()
    plt.show()


demo()
