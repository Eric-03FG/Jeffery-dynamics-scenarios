# - Case A: Simple shear in xy plane: u = γ y e_x
#   Integrate Jeffery equation in vector form for director p(t) on S².
import numpy as np
import matplotlib.pyplot as plt

import src_Jeffery as jf

def main():
    # Parameters
    gamma = 1.0       # shear rate (γ)
    c = 6.0           # aspect ratio of the spheroid
    lam = jf.lambda_ar(c)   # B

    #Case A: u = γ y e_x  -> plane="xy" 
    G = jf.grad_u_simple_shear(gamma, plane="xy")
    E, W = jf.decompose_grad_u(G)

    # Initial condition
    theta0 = np.deg2rad(60.0)
    phi0   = np.deg2rad(20.0)
    d0 = jf.sph_to_vec(theta0, phi0)

    # Time
    t0, tf = 0.0, 40.0
    t_eval = np.linspace(t0, tf, 4000)

    # Jeffery int
    t, D = jf.integrate_director_vector(d0, (t0, tf), t_eval, E, W, lam)

    d1, d2, d3 = D[:, 0], D[:, 1], D[:, 2]

    # Angles
    theta = np.arccos(np.clip(d3, -1.0, 1.0))
    phi = np.unwrap(np.arctan2(d2, d1))

   
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    axs[0, 0].plot(t, d1, label="d1")
    axs[0, 0].plot(t, d2, label="d2")
    axs[0, 0].plot(t, d3, label="d3")
    axs[0, 0].set_xlabel("t")
    axs[0, 0].set_ylabel("d(t)")
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[0, 0].set_title(rf"Case A: $u=\gamma y\,e_x$  ($\gamma={gamma}$, $B={lam:.4f}$)")

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

    # Norma (después de normalize_rows será ~1; sirve como check)
    norm = np.linalg.norm(D, axis=1)
    axs[1, 1].plot(t, norm, label=r"$\|d\|$")
    axs[1, 1].set_xlabel("t")
    axs[1, 1].set_ylabel(r"$\|d\|$")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()