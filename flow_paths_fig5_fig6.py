"""
Recreates Figures 5 and 6 from Holderrieth & Erives,
"An Introduction to Flow Matching and Diffusion Models" (MIT 6.S184).

Figure 5: conditional and marginal Gaussian probability paths
          p_t(x|z) = N(alpha_t z, beta_t^2 I) with alpha_t = t, beta_t = 1 - t,
          going from p_init = N(0, I) to a chessboard p_data in 2D.

Figure 6: simulate the ODE dX_t = u_t(X_t) dt with Euler's method
          (top) using the conditional vector field u_t(x|z), Eq. (20)
          (bottom) using the marginal vector field u_t(x), Eq. (18)
          and check the samples follow the paths from Figure 5.

No neural network is trained. Because we represent p_data by a finite set of
samples z_1..z_N, the integral in Eq. (18) becomes a weighted sum, so the
marginal vector field can be computed exactly.

Requirements: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Noise schedule: alpha_t = t, beta_t = 1 - t (the "CondOT" path)
# ---------------------------------------------------------------------------
def alpha(t):
    return t


def beta(t):
    return 1.0 - t


def alpha_dot(t):
    return np.ones_like(np.asarray(t, dtype=float))


def beta_dot(t):
    return -np.ones_like(np.asarray(t, dtype=float))


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
def sample_p_init(n):
    """Standard Gaussian N(0, I) in 2D."""
    return rng.standard_normal((n, 2))


def sample_chessboard(n, half_width=4.0, square=2.0):
    """Uniform samples on the 'dark' squares of a chessboard on [-4, 4]^2."""
    out = []
    while len(out) < n:
        x = rng.uniform(-half_width, half_width, size=(2 * n, 2))
        i = np.floor(x[:, 0] / square)
        j = np.floor(x[:, 1] / square)
        keep = (i + j) % 2 == 0
        out.extend(x[keep])
    return np.array(out[:n])


# ---------------------------------------------------------------------------
# Probability paths (sampling), Eq. (16)
# ---------------------------------------------------------------------------
def sample_conditional_path(z, t, n):
    """x ~ p_t(.|z) = N(alpha_t z, beta_t^2 I) for a single data point z."""
    eps = sample_p_init(n)
    return alpha(t) * z + beta(t) * eps


def sample_marginal_path(t, n):
    """x ~ p_t: draw z ~ p_data, then x ~ p_t(.|z)."""
    z = sample_chessboard(n)
    eps = sample_p_init(n)
    return alpha(t) * z + beta(t) * eps


# ---------------------------------------------------------------------------
# Vector fields
# ---------------------------------------------------------------------------
def conditional_vector_field(x, z, t):
    """Eq. (20): u_t(x|z) = (alpha_dot - beta_dot/beta * alpha) z + beta_dot/beta * x."""
    a, b = alpha(t), beta(t)
    ad, bd = alpha_dot(t), beta_dot(t)
    return (ad - bd / b * a) * z + (bd / b) * x


def marginal_vector_field(x, z_data, t):
    """
    Eq. (18) with p_data replaced by the empirical distribution of z_data:
        u_t(x) = sum_i w_i(x) u_t(x|z_i),
        w_i(x) = p_t(x|z_i) / sum_j p_t(x|z_j)   (posterior over data points).
    x: (M, 2) particles, z_data: (N, 2) data points.
    """
    a, b = alpha(t), beta(t)
    # squared distances between each particle and each mean alpha_t z_i -> (M, N)
    diff = x[:, None, :] - a * z_data[None, :, :]
    log_w = -np.sum(diff ** 2, axis=-1) / (2 * b ** 2)
    log_w -= log_w.max(axis=1, keepdims=True)          # for numerical stability
    w = np.exp(log_w)
    w /= w.sum(axis=1, keepdims=True)
    # conditional vector field for every (particle, data point) pair -> (M, N, 2)
    u_cond = conditional_vector_field(x[:, None, :], z_data[None, :, :], t)
    return np.sum(w[..., None] * u_cond, axis=1)


# ---------------------------------------------------------------------------
# Euler simulation, Algorithm 1
# ---------------------------------------------------------------------------
def simulate_ode(x0, vector_field, n_steps=200):
    """Returns the full trajectory, shape (n_steps + 1, M, 2), and the time grid."""
    h = 1.0 / n_steps
    ts = np.linspace(0.0, 1.0, n_steps + 1)
    xs = [x0]
    x = x0.copy()
    for k in range(n_steps):          # evaluates u at t = 0, h, ..., 1 - h
        x = x + h * vector_field(x, ts[k])
        xs.append(x)
    return np.stack(xs), ts


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
LIM = 5.0
BINS = 120


def hist2d(ax, pts, title):
    ax.hist2d(pts[:, 0], pts[:, 1], bins=BINS,
              range=[[-LIM, LIM], [-LIM, LIM]], cmap="viridis")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def background(ax, data_pts, init_pts):
    """Blue = p_data, red = p_init, drawn faintly behind the samples."""
    ax.scatter(init_pts[:, 0], init_pts[:, 1], s=2, c="tab:red", alpha=0.08)
    ax.scatter(data_pts[:, 0], data_pts[:, 1], s=2, c="tab:blue", alpha=0.08)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Figure 5
# ---------------------------------------------------------------------------
def figure_5(z, n=50_000, times=(0.0, 0.25, 0.5, 0.75, 1.0)):
    fig, axes = plt.subplots(2, len(times), figsize=(3 * len(times), 7))
    for k, t in enumerate(times):
        hist2d(axes[0, k], sample_conditional_path(z, t, n), f"t = {t}")
        hist2d(axes[1, k], sample_marginal_path(t, n), f"t = {t}")
    axes[0, 0].set_ylabel("conditional  $p_t(x|z)$")
    axes[1, 0].set_ylabel("marginal  $p_t(x)$")
    axes[0, -1].scatter(*z, c="red", s=25, marker="*")   # mark the data point z
    fig.suptitle("Figure 5: Gaussian probability path, "
                 r"$\alpha_t = t,\ \beta_t = 1-t$")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6
# ---------------------------------------------------------------------------
def figure_6(z, n_particles=1000, n_data=2000, n_steps=200,
             times=(0.0, 0.25, 0.5, 0.75, 1.0)):
    data_bg = sample_chessboard(3000)
    init_bg = sample_p_init(3000)
    z_data = sample_chessboard(n_data)
    colours = plt.cm.plasma(np.linspace(0, 0.9, len(times)))

    rows = {
        "conditional": dict(
            truth=lambda t: sample_conditional_path(z, t, n_particles),
            field=lambda x, t: conditional_vector_field(x, z, t)),
        "marginal": dict(
            truth=lambda t: sample_marginal_path(t, n_particles),
            field=lambda x, t: marginal_vector_field(x, z_data, t)),
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for r, (name, row) in enumerate(rows.items()):
        traj, ts = simulate_ode(sample_p_init(n_particles), row["field"], n_steps)
        idx = [int(round(t * n_steps)) for t in times]

        # Left: ground truth samples from the path
        ax = axes[r, 0]
        background(ax, data_bg, init_bg)
        for c, t in zip(colours, times):
            p = row["truth"](t)
            ax.scatter(p[:, 0], p[:, 1], s=3, color=c, label=f"t={t}")
        ax.set_title(f"Ground truth: {name} path")

        # Middle: ODE samples at the same times
        ax = axes[r, 1]
        background(ax, data_bg, init_bg)
        for c, t, i in zip(colours, times, idx):
            ax.scatter(traj[i, :, 0], traj[i, :, 1], s=3, color=c, label=f"t={t}")
        ax.set_title(f"ODE samples over time ({name} $u_t$)")

        # Right: a handful of full trajectories
        ax = axes[r, 2]
        background(ax, data_bg, init_bg)
        for m in range(60):
            ax.plot(traj[:, m, 0], traj[:, m, 1], lw=0.7, color="k", alpha=0.6)
        ax.scatter(traj[0, :60, 0], traj[0, :60, 1], s=8, c="tab:red", zorder=3)
        ax.scatter(traj[-1, :60, 0], traj[-1, :60, 1], s=8, c="tab:blue", zorder=3)
        ax.set_title(f"ODE trajectories ({name} $u_t$)")

    axes[0, 0].legend(loc="lower left", fontsize=7, markerscale=3)
    fig.suptitle("Figure 6: simulating the probability path with an ODE "
                 "(red = $p_{init}$, blue = $p_{data}$)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    z = np.array([2.5, 2.5])     # the single data point for the conditional path
    figure_5(z).savefig("figure5.png", dpi=130)
    figure_6(z).savefig("figure6.png", dpi=130)
    plt.show()
