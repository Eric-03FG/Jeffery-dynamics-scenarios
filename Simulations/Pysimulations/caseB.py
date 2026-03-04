# - Case B: Simple shear in xz plane: u = γ z e_x
#   Integrate Jeffery equation in (θ, φ) with coupled ODEs.

import numpy as np
import matplotlib.pyplot as plt

import src_Jeffery as jf


def main():
    # Parameters
    gamma = 1.0
    c = 6.0
    B = jf.lambda_ar(c)

    # Case B flow: u = γ z e_x
    G = jf.grad_u_simple_shear(gamma, plane="xz")
    E, W = jf.decompose_grad_u(G)

    # Initial condition (theta0, phi0)
    theta0 = np.deg2rad(60.0)
    phi0   = np.deg2rad(20.0)
    y0 = np.array([theta0, phi0], dtype=float)

    # Time
    t0, tf = 0.0, 40.0
    t_eval = np.linspace(t0, tf, 4000)

    # Integrate
    t, Y = jf.integrate_theta_phi(y0, (t0, tf), t_eval, E, W, B)
    theta = Y[:, 0]
    phi = np.unwrap(Y[:, 1])

    # Reconstruct d(t)
    D = np.column_stack((
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ))
    D = jf.normalize_rows(D)

    d1, d2, d3 = D[:, 0], D[:, 1], D[:, 2]
    norm = np.linalg.norm(D, axis=1)

    # Plot 2x2
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    axs[0, 0].plot(t, d1, label="d1")
    axs[0, 0].plot(t, d2, label="d2")
    axs[0, 0].plot(t, d3, label="d3")
    axs[0, 0].set_xlabel("t")
    axs[0, 0].set_ylabel("d(t)")
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[0, 0].set_title(rf"Case B: $u=\gamma z\,e_x$  ($\gamma={gamma}$, $B={B:.4f}$)")

    axs[0, 1].plot(t, theta, label=r"$\theta(t)$")
    axs[0, 1].set_xlabel("t")
    axs[0, 1].set_ylabel(r"$\theta$ [rad]")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(t, phi, label=r"$\phi(t)$")
    axs[1, 0].set_xlabel("t")
    axs[1, 0].set_ylabel(r"$\phi$ [rad]")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(t, norm, label=r"$\|d\|$")
    axs[1, 1].set_xlabel("t")
    axs[1, 1].set_ylabel(r"$\|d\|$")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()