# =============================================================================
# JEFFERY DETERMINISTA + ESTOCÁSTICO (Langevin / Euler-Maruyama)
# -----------------------------------------------------------------------------
# Este script:
#
#   1) Resuelve la ecuación de Jeffery en forma vectorial:
#
#        p_dot = W p + beta * (E p - (p^T E p) p)
#
#      donde:
#        E = (A + A^T)/2
#        W = (A - A^T)/2
#        A = grad(u) = ∇u
#
#   2) Permite usar campos de flujo predefinidos:
#        - shear_xy
#        - shear_xz
#        - extensional_xy
#        - rotation_z
#        - mixed_shear_stretch
#        - custom
#
#   3) Recibe condiciones iniciales en coordenadas esféricas (theta0, phi0),
#      las convierte a cartesianas y resuelve para p(t).
#
#   4) Calcula beta a partir del aspect ratio r:
#
#        beta = (r^2 - 1)/(r^2 + 1)
#
#   5) Grafica en una sola figura con 4 subplots:
#       - p_x, p_y, p_z vs t
#       - theta vs t [grados]
#       - phi vs t [grados]
#       - ||p|| vs t
#
#   6) Genera una figura interactiva 3D NATIVA con PyVista
#      con la esfera unitaria y la trayectoria.
#
# -----------------------------------------------------------------------------
#   - MODE = "deterministic" o "stochastic"
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pyvista as pv
import os
from pathlib import Path
from datetime import datetime


# =============================================================================
# 1. FUNCIONES AUXILIARES
# =============================================================================

def bretherton_parameter(r):
    """
    Calcula el parámetro de Bretherton beta a partir del aspect ratio r = a/b.

    Para un esferoide prolato:
        beta = (r^2 - 1)/(r^2 + 1)

    Casos importantes:
        r = 1  -> beta = 0   (esfera)
        r > 1  -> 0 < beta < 1 (partícula alargada)
    """
    r = float(r)
    return (r**2 - 1.0) / (r**2 + 1.0)


def p_from_angles(theta, phi):
    """
    Convierte coordenadas esféricas (theta, phi) a un vector unitario cartesiano.

    Convención:
      - theta: ángulo polar desde +z, theta en [0, pi]
      - phi  : ángulo azimutal en el plano xy desde +x
    """
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], dtype=float)


def angles_from_p(p):
    """
    Convierte un vector p = [px, py, pz] a coordenadas esféricas (theta, phi).

    Se asume que p es aproximadamente unitario.
    """
    px, py, pz = p
    theta = np.arccos(np.clip(pz, -1.0, 1.0))
    phi = np.arctan2(py, px)
    return theta, phi


def vector_norm(v):
    """Norma euclidiana."""
    return np.sqrt(np.sum(v**2))


def normalize_vector(v):
    """
    Normaliza un vector 3D.
    """
    n = vector_norm(v)
    if n < 1e-15:
        raise ValueError("No se puede normalizar un vector de norma ~ 0.")
    return v / n


def unwrap_angle_deg(angle_deg):
    """
    Desenvuelve un ángulo en grados para evitar saltos artificiales de ±180°.
    """
    return np.rad2deg(np.unwrap(np.deg2rad(angle_deg)))


def wrap_angle_deg(angle_deg):
    """
    Lleva un ángulo en grados al intervalo [-180, 180].
    """
    return (angle_deg + 180.0) % 360.0 - 180.0


def get_velocity_plane(case):
    if case in ["shear_xy", "extensional_xy", "rotation_z"]:
        return "xy"
    elif case in ["shear_xz", "mixed_shear_stretch"]:
        return "xz"
    else:
        return "xy"


def build_velocity_field_on_plane(A, plane="xy", n=10, lim=1.0):
    grid = np.linspace(-lim, lim, n)

    points = []
    vectors = []

    for a in grid:
        for b in grid:
            if plane == "xy":
                X = np.array([a, b, 0.0])
            elif plane == "xz":
                X = np.array([a, 0.0, b])
            elif plane == "yz":
                X = np.array([0.0, a, b])
            else:
                raise ValueError("plane debe ser 'xy', 'xz' o 'yz'")

            u = A @ X
            points.append(X)
            vectors.append(u)

    points = np.array(points)
    vectors = np.array(vectors)
    return points, vectors


def get_flow_latex(case, flow_params):
    if case == "shear_xy":
        gamma = flow_params.get("gamma", 1.0)
        return rf"$\mathbf{{u}} = ({gamma} y,\; 0,\; 0)$"

    elif case == "shear_xz":
        gamma = flow_params.get("gamma", 1.0)
        return rf"$\mathbf{{u}} = ({gamma} z,\; 0,\; 0)$"

    elif case == "extensional_xy":
        eps = flow_params.get("epsilon_dot", 1.0)
        return rf"$\mathbf{{u}} = ({eps} x,\; -{eps} y,\; 0)$"

    elif case == "rotation_z":
        omega = flow_params.get("omega", 1.0)
        return rf"$\mathbf{{u}} = (-{omega} y,\; {omega} x,\; 0)$"

    elif case == "mixed_shear_stretch":
        gamma = flow_params.get("gamma", 1.0)
        s = flow_params.get("s", 0.1)
        return rf"$\mathbf{{u}} = ({s} x + {gamma} z,\; -{s} y,\; 0)$"

    elif case == "custom":
        return r"$\mathbf{u} = \mathbf{A}\mathbf{x}$"

    else:
        return r"$\mathbf{u} = \mathbf{A}\mathbf{x}$"


# =============================================================================
# 2. CAMPOS DE FLUJO PREDEFINIDOS
# =============================================================================

def grad_u_shear_xy(gamma=1.0):
    """
    Shear simple en el plano xy:
        u = gamma * y * e_x
    """
    return np.array([
        [0.0, gamma, 0.0],
        [0.0, 0.0,   0.0],
        [0.0, 0.0,   0.0]
    ], dtype=float)


def grad_u_shear_xz(gamma=1.0):
    """
    Shear simple en el plano xz:
        u = gamma * z * e_x
    """
    return np.array([
        [0.0, 0.0, gamma],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=float)


def grad_u_extensional_xy(epsilon_dot=1.0):
    """
    Flujo extensional planar:
        u = (epsilon_dot * x, -epsilon_dot * y, 0)
    """
    return np.array([
        [epsilon_dot, 0.0, 0.0],
        [0.0, -epsilon_dot, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=float)


def grad_u_rotation_z(omega=1.0):
    """
    Flujo rotacional rígido alrededor de z:
        u = (-omega y, omega x, 0)
    """
    return np.array([
        [0.0, -omega, 0.0],
        [omega, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=float)


def grad_u_mixed_shear_stretch(gamma=1.0, s=0.1):
    """
    Flujo mixto shear + stretching:
        u = A x
    con
        A = [[ s, 0, gamma],
             [ 0,-s,   0 ],
             [ 0, 0,   0 ]]
    """
    return np.array([
        [s, 0.0, gamma],
        [0.0, -s, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=float)


def build_grad_u(case, flow_params=None, grad_u_custom=None):
    """
    Devuelve la matriz A = ∇u correspondiente al caso elegido.
    """
    if flow_params is None:
        flow_params = {}

    if case == "shear_xy":
        gamma = flow_params.get("gamma", 1.0)
        return grad_u_shear_xy(gamma=gamma)

    elif case == "shear_xz":
        gamma = flow_params.get("gamma", 1.0)
        return grad_u_shear_xz(gamma=gamma)

    elif case == "extensional_xy":
        epsilon_dot = flow_params.get("epsilon_dot", 1.0)
        return grad_u_extensional_xy(epsilon_dot=epsilon_dot)

    elif case == "rotation_z":
        omega = flow_params.get("omega", 1.0)
        return grad_u_rotation_z(omega=omega)

    elif case == "mixed_shear_stretch":
        gamma = flow_params.get("gamma", 1.0)
        s = flow_params.get("s", 0.1)
        return grad_u_mixed_shear_stretch(gamma=gamma, s=s)

    elif case == "custom":
        if grad_u_custom is None:
            raise ValueError("Para case='custom' debes proporcionar grad_u_custom.")
        A = np.asarray(grad_u_custom, dtype=float)
        if A.shape != (3, 3):
            raise ValueError("grad_u_custom debe ser una matriz 3x3.")
        return A

    else:
        raise ValueError(
            f"Caso '{case}' no reconocido. "
            "Usa: shear_xy, shear_xz, extensional_xy, rotation_z, "
            "mixed_shear_stretch o custom."
        )


# =============================================================================
# 3. ECUACIÓN DE JEFFERY
# =============================================================================

def jeffery_rhs(t, p, beta, A):
    """
    Lado derecho de la ecuación de Jeffery:

        p_dot = W p + beta * (E p - (p^T E p) p)

    donde:
        E = (A + A^T)/2
        W = (A - A^T)/2
    """
    p = normalize_vector(np.asarray(p, dtype=float))

    E = 0.5 * (A + A.T)
    W = 0.5 * (A - A.T)

    Ep = E @ p
    pEp = float(p @ Ep)

    return W @ p + beta * (Ep - pEp * p)


def jeffery_drift(p, beta, A):
    """
    Drift determinista de Jeffery en forma vectorial.
    """
    p = normalize_vector(np.asarray(p, dtype=float))

    E = 0.5 * (A + A.T)
    W = 0.5 * (A - A.T)

    Ep = E @ p
    pEp = float(p @ Ep)

    return W @ p + beta * (Ep - pEp * p)

# =============================================================================
# 3B. TORQUE ÓPTICO / ALINEACIÓN ÓPTICA
# =============================================================================

def normalize_or_default(v, default=None):
    """
    Normaliza un vector. Si v es None, usa default.
    """
    if v is None:
        if default is None:
            raise ValueError("Se requiere un vector o un default.")
        v = np.asarray(default, dtype=float)
    return normalize_vector(np.asarray(v, dtype=float))


def optical_alignment_drift(p, e_pol=np.array([1.0, 0.0, 0.0]), lambda_opt=0.0):
    """
    Drift orientacional inducido por una pinza óptica linealmente polarizada.

    Modelo:
        U(p) = -(1/2) * Delta_alpha * E0^2 * (p · e_pol)^2

    Tras absorber Delta_alpha * E0^2 / zeta_r en un parámetro efectivo
    lambda_opt [1/s], la contribución a p_dot queda:

        p_dot_opt = lambda_opt * (p·e) * [ e - (p·e) p ]

    donde:
        - p es el vector unitario de orientación
        - e_pol es la dirección de polarización lineal
        - lambda_opt controla la fuerza de alineación óptica

    Si lambda_opt = 0, no hay efecto de la pinza.
    """
    p = normalize_vector(np.asarray(p, dtype=float))
    e_pol = normalize_vector(np.asarray(e_pol, dtype=float))

    c = float(np.dot(p, e_pol))
    return lambda_opt * c * (e_pol - c * p)


def total_orientation_drift(p, beta, A, e_pol=None, lambda_opt=0.0):
    """
    Drift total:
        Jeffery + alineación óptica
    """
    drift_j = jeffery_drift(p, beta, A)
    drift_opt = optical_alignment_drift(
        p,
        e_pol=np.array([1.0, 0.0, 0.0]) if e_pol is None else e_pol,
        lambda_opt=lambda_opt
    )
    return drift_j + drift_opt


# =============================================================================
# 4. INTERFACES DE SIMULACIÓN
# =============================================================================

def run_case(
    case="shear_xy",
    r=5.0,
    theta0_deg=63.0,
    phi0_deg=17.0,
    t_final=40.0,
    n_points=4000,
    flow_params=None,
    grad_u_custom=None,
    method="RK45",
    rtol=1e-10,
    atol=1e-10,
    optical_params=None
):
    """
    Interfaz principal para correr un caso determinista de Jeffery
    con opción de alineación óptica.
    """
    if flow_params is None:
        flow_params = {}

    if optical_params is None:
        optical_params = {}

    theta0 = np.deg2rad(theta0_deg)
    phi0 = np.deg2rad(phi0_deg)

    beta = bretherton_parameter(r)
    p0 = p_from_angles(theta0, phi0)
    A = build_grad_u(case, flow_params=flow_params, grad_u_custom=grad_u_custom)

    # Parámetros ópticos lumped
    lambda_opt = optical_params.get("lambda_opt", 0.0)
    e_pol = optical_params.get("e_pol", np.array([1.0, 0.0, 0.0], dtype=float))
    e_pol = normalize_vector(np.asarray(e_pol, dtype=float))

    t_span = (0.0, t_final)
    t_eval = np.linspace(0.0, t_final, n_points)

    def rhs(t, p):
        p = normalize_vector(p)
        return total_orientation_drift(
            p,
            beta=beta,
            A=A,
            e_pol=e_pol,
            lambda_opt=lambda_opt
        )

    sol = solve_ivp(
        rhs,
        t_span,
        p0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol
    )

    if not sol.success:
        raise RuntimeError(f"La integración falló: {sol.message}")

    P = sol.y.T
    norms = np.linalg.norm(P, axis=1)

    theta = np.zeros_like(sol.t)
    phi = np.zeros_like(sol.t)

    for i in range(len(sol.t)):
        p_i = normalize_vector(P[i])
        theta[i], phi[i] = angles_from_p(p_i)

    result = {
        "mode": "deterministic",
        "case": case,
        "A": A,
        "t": sol.t,
        "P": P,
        "px": P[:, 0],
        "py": P[:, 1],
        "pz": P[:, 2],
        "theta_rad": theta,
        "phi_rad": phi,
        "theta_deg": np.rad2deg(theta),
        "phi_deg_wrapped": np.rad2deg(phi),
        "phi_deg_unwrapped": unwrap_angle_deg(np.rad2deg(phi)),
        "norms": norms,
        "beta": beta,
        "aspect_ratio": r,
        "theta0_rad": theta0,
        "phi0_rad": phi0,
        "theta0_deg": theta0_deg,
        "phi0_deg": phi0_deg,
        "t_final": t_final,
        "flow_params": flow_params,
        "optical_params": {
            "lambda_opt": lambda_opt,
            "e_pol": e_pol
        },
        "solver_result": sol
    }

    return result


def run_case_stochastic(
    case="shear_xy",
    r=5.0,
    theta0_deg=63.0,
    phi0_deg=17.0,
    t_final=40.0,
    dt=1e-3,
    D=1e-2,
    flow_params=None,
    grad_u_custom=None,
    random_seed=1234,
    renormalize_every_step=True,
    optical_params=None
):
    """
    Simulación estocástica de Jeffery en 3D usando Euler-Maruyama
    sobre el vector unitario p, con opción de alineación óptica.

    Interpretación de Itô:
        dp = [ drift_total(p) - 2 D p ] dt + sqrt(2D) (p x dW)

    donde:
        drift_total = Jeffery + alineación óptica
        dW ~ N(0, dt I).
    """
    if flow_params is None:
        flow_params = {}

    if optical_params is None:
        optical_params = {}

    rng = np.random.default_rng(random_seed)

    theta0 = np.deg2rad(theta0_deg)
    phi0 = np.deg2rad(phi0_deg)

    beta = bretherton_parameter(r)
    p0 = p_from_angles(theta0, phi0)
    A = build_grad_u(case, flow_params=flow_params, grad_u_custom=grad_u_custom)

    lambda_opt = optical_params.get("lambda_opt", 0.0)
    e_pol = optical_params.get("e_pol", np.array([1.0, 0.0, 0.0], dtype=float))
    e_pol = normalize_vector(np.asarray(e_pol, dtype=float))

    n_steps = int(np.floor(t_final / dt)) + 1
    t = np.linspace(0.0, dt * (n_steps - 1), n_steps)

    P = np.zeros((n_steps, 3), dtype=float)
    P[0] = normalize_vector(p0)

    for n in range(n_steps - 1):
        p = normalize_vector(P[n])

        # Drift total: Jeffery + alineación óptica
        drift_det = total_orientation_drift(
            p,
            beta=beta,
            A=A,
            e_pol=e_pol,
            lambda_opt=lambda_opt
        )

        # Drift geométrico de Itô
        drift_ito = -2.0 * D * p

        # Incremento Wiener vectorial
        dW = np.sqrt(dt) * rng.normal(size=3)

        # Término estocástico
        noise = np.sqrt(2.0 * D) * np.cross(p, dW)

        # Euler-Maruyama
        p_new = p + (drift_det + drift_ito) * dt + noise

        if renormalize_every_step:
            p_new = normalize_vector(p_new)

        P[n + 1] = p_new

    norms = np.linalg.norm(P, axis=1)

    theta = np.zeros(n_steps)
    phi = np.zeros(n_steps)

    for i in range(n_steps):
        p_i = normalize_vector(P[i])
        theta[i], phi[i] = angles_from_p(p_i)

    result = {
        "mode": "stochastic",
        "case": case,
        "A": A,
        "t": t,
        "P": P,
        "px": P[:, 0],
        "py": P[:, 1],
        "pz": P[:, 2],
        "theta_rad": theta,
        "phi_rad": phi,
        "theta_deg": np.rad2deg(theta),
        "phi_deg_wrapped": np.rad2deg(phi),
        "phi_deg_unwrapped": unwrap_angle_deg(np.rad2deg(phi)),
        "norms": norms,
        "beta": beta,
        "aspect_ratio": r,
        "theta0_rad": theta0,
        "phi0_rad": phi0,
        "theta0_deg": theta0_deg,
        "phi0_deg": phi0_deg,
        "t_final": t_final,
        "dt": dt,
        "D": D,
        "flow_params": flow_params,
        "optical_params": {
            "lambda_opt": lambda_opt,
            "e_pol": e_pol
        },
        "random_seed": random_seed,
        "renormalize_every_step": renormalize_every_step
    }

    return result


# =============================================================================
# 5. GRÁFICA 2x2
# =============================================================================

def plot_jeffery_4subplots(
    result,
    flow_params=None,
    use_unwrapped_phi=True,
    figsize=(12, 8),
    save=True,
    folder="ResultsG",
    filename=None
):
    t = result["t"]
    theta_deg = result["theta_deg"]
    phi_deg = result["phi_deg_unwrapped"] if use_unwrapped_phi else result["phi_deg_wrapped"]
    beta = result["beta"]
    r = result["aspect_ratio"]
    case = result["case"]
    mode = result.get("mode", "deterministic")
    D = result.get("D", None)
    optical_params = result.get("optical_params", {})
    lambda_opt = optical_params.get("lambda_opt", 0.0)
    e_pol = optical_params.get("e_pol", np.array([1.0, 0.0, 0.0]))

    flow_latex = get_flow_latex(case, flow_params or result.get("flow_params", {}))

    if save:
        os.makedirs(folder, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    if mode == "stochastic" and D is not None:
        suptitle = (
            rf"$d\mathbf{{p}}$ | "
            rf"$r = {r:.3f}$, $\beta = {beta:.3f}$, "
            rf"$D = {D:.3e}$, $\lambda_{{opt}} = {lambda_opt:.3e}$" "\n"
            rf"{flow_latex}"
        )
    else:
        suptitle = (
            rf"$\dot{{\mathbf{{p}}}}$ | "
            rf"$r = {r:.3f}$, $\beta = {beta:.3f}$, "
            rf"$\lambda_{{opt}} = {lambda_opt:.3e}$" "\n"
            rf"{flow_latex}"
        )

    fig.suptitle(suptitle, fontsize=14)

    axs[0, 0].plot(t, result["px"], label=r"$p_x$", linewidth=2)
    axs[0, 0].plot(t, result["py"], label=r"$p_y$", linewidth=2)
    axs[0, 0].plot(t, result["pz"], label=r"$p_z$", linewidth=2)
    axs[0, 0].set_xlabel(r"$t$")
    axs[0, 0].set_ylabel(r"$\mathbf{p}(t)$")
    axs[0, 0].set_title(r"Components of $\mathbf{p}(t)$")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(t, theta_deg, label=r"$\theta(t)$", linewidth=2)
    axs[0, 1].set_xlabel(r"$t$")
    axs[0, 1].set_ylabel(r"$\theta\ [^\circ]$")
    axs[0, 1].set_title(r"$\theta(t)$")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(t, phi_deg, label=r"$\phi(t)$", linewidth=2)
    axs[1, 0].set_xlabel(r"$t$")
    axs[1, 0].set_ylabel(r"$\phi\ [^\circ]$")
    axs[1, 0].set_title(r"$\phi(t)$")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(t, result["norms"], label=r"$\|\mathbf{p}\|$", linewidth=2)
    axs[1, 1].set_xlabel(r"$t$")
    axs[1, 1].set_ylabel(r"$\|\mathbf{p}\|$")
    axs[1, 1].set_title(r"$\|\mathbf{p}(t)\|$")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()

    if save:
        if filename is None:
            if mode == "stochastic" and D is not None:
                filename = f"jeffery_stochastic_{case}_r{r:.3f}_beta{beta:.3f}_D{D:.3e}.png"
            else:
                filename = f"jeffery_{case}_r{r:.3f}_beta{beta:.3f}.png"

        path = os.path.join(folder, filename)

        fig.savefig(
            path,
            dpi=500,
            bbox_inches="tight"
        )

        print(f"\nFigura guardada en: {path}")

    return fig, axs


# =============================================================================
# 6. VISUALIZACIÓN 3D INTERACTIVA NATIVA CON PYVISTA
# =============================================================================

def get_velocity_planes(case):
    """
    Devuelve una lista de planos donde se visualizará el campo de velocidad.
    Solo afecta la figura 3D.
    """
    if case == "mixed_shear_stretch":
        return ["xz", "xy"]   # plano principal + plano extra
    elif case in ["shear_xy", "extensional_xy", "rotation_z"]:
        return ["xy"]
    elif case == "shear_xz":
        return ["xz"]
    else:
        return ["xy"]

def plot_trajectory_on_sphere_interactive(
    result,
    title=None,
    screenshot_path=None,
    save_screenshot=False,
    show_grid=True,
    sphere_opacity=0.22
):
    def format_vector(v):
        return f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})"

    def print_state_block(name, p):
        p = normalize_vector(p)
        theta, phi = angles_from_p(p)
        rnorm = vector_norm(p)

        theta_deg = np.degrees(theta)
        phi_deg = np.degrees(phi)

        if phi_deg < 0:
            phi_deg += 360.0

        print(f"  {name}")
        print("  " + "─" * 52)
        print(f"    Vector cartesiano : {format_vector(p)}")
        print(f"    Norma             : {rnorm:.4f}")
        print(f"    θ (polar)         : {theta_deg:.4f}°")
        print(f"    φ (azimutal)      : {phi_deg:.4f}°")
        print()

        return theta_deg, phi_deg

    P = result["P"]
    case = result["case"]
    mode = result.get("mode", "deterministic")
    r_val = result["aspect_ratio"]
    beta_val = result["beta"]
    D = result.get("D", None)
    optical_params = result.get("optical_params", {})
    lambda_opt = optical_params.get("lambda_opt", 0.0)
    e_pol = optical_params.get("e_pol", np.array([1.0, 0.0, 0.0]))

    if title is None:
        flow_latex = get_flow_latex(case, result["flow_params"])

        if mode == "stochastic" and D is not None:
            title = (
                rf"$\mathbf{{p}}(t)\ \mathrm{{on}}\ S^2$"
                "\n"
                + flow_latex
                + "\n"
                + rf"$r = {r_val:.3f},\ \beta = {beta_val:.3f},\ D = {D:.3e},\ \lambda_{{opt}} = {lambda_opt:.3e}$"
            )
        else:
            title = (
                rf"$\mathbf{{p}}(t)\ \mathrm{{on}}\ S^2$"
                "\n"
                + flow_latex
                + "\n"
                + rf"$r = {r_val:.3f},\ \beta = {beta_val:.3f},\ \lambda_{{opt}} = {lambda_opt:.3e}$"
            )

    plotter = pv.Plotter(window_size=(1000, 780))
    plotter.set_background("white")

    results_dir = Path("ResultsG")
    results_dir.mkdir(parents=True, exist_ok=True)

    def build_filename():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if mode == "stochastic" and D is not None:
            return results_dir / f"{case}_D{D:.3e}_{timestamp}.png"
        return results_dir / f"{case}_{timestamp}.png"

    def save_current_screenshot():
        filepath = Path(screenshot_path) if screenshot_path else build_filename()
        plotter.screenshot(str(filepath))
        print(f"\n📸 Screenshot guardado en: {filepath}")

    sphere = pv.Sphere(radius=1.0, theta_resolution=80, phi_resolution=80)
    plotter.add_mesh(
        sphere,
        color="lightgray",
        opacity=sphere_opacity,
        smooth_shading=True,
        specular=0.15,
        show_edges=False
    )

    polyline = pv.lines_from_points(P)
    plotter.add_mesh(
        polyline,
        color="red",
        line_width=4,
        label="p(t)"
    )

    p_start = normalize_vector(P[0])
    p_end = normalize_vector(P[-1])

    print()
    print("╔" + "═" * 58 + "╗")
    print("║{:^58s}║".format("JEFFERY ORBIT — ORIENTACIONES"))
    print("╚" + "═" * 58 + "╝")
    print(f"  Caso: {case}")
    print(f"  Modo: {mode}")

    if mode == "stochastic" and D is not None:
        print(f"  D   : {D:.6e}")
    print()

    print("  Parámetros de la partícula")
    print("  " + "─" * 52)
    print(f"    Aspect ratio (r = a/b) : {r_val:.4f}")
    print(f"    Bretherton (β)         : {beta_val:.6f}")
    print(f"  lambda_opt : {lambda_opt:.6e}  [1/s]")
    print(f"  e_pol      : ({e_pol[0]:.3f}, {e_pol[1]:.3f}, {e_pol[2]:.3f})")

    if np.isclose(beta_val, 0.0):
        tipo = "Esfera (sin órbitas de Jeffery)"
    elif beta_val < 0.5:
        tipo = "Débilmente alargada"
    elif beta_val < 0.9:
        tipo = "Elipsoide prolato"
    else:
        tipo = "Altamente alargada (casi fibra)"

    print(f"    Tipo de partícula      : {tipo}")
    print()

    theta0_deg, phi0_deg = print_state_block("t0  (estado inicial)", p_start)
    thetaf_deg, phif_deg = print_state_block("tf  (estado final)", p_end)

    delta_theta = thetaf_deg - theta0_deg
    phi_series = np.array([phi0_deg, phif_deg])
    phi_unwrapped = unwrap_angle_deg(phi_series)
    delta_phi = phi_unwrapped[1] - phi_unwrapped[0]

    print("  Cambios angulares")
    print("  " + "─" * 52)
    print(f"    Δθ (polar)         : {delta_theta:.4f}°")
    print(f"    Δφ (azimutal)      : {delta_phi:.4f}°")
    print()

    start = pv.PolyData(p_start.reshape(1, 3))
    end = pv.PolyData(p_end.reshape(1, 3))

    plotter.add_mesh(start, color="green", point_size=8, render_points_as_spheres=True)
    plotter.add_mesh(end, color="blue", point_size=8, render_points_as_spheres=True)

    arrow_length = 0.98

    arrow_t0 = pv.Arrow(
        start=(0.0, 0.0, 0.0),
        direction=p_start,
        tip_length=0.22,
        tip_radius=0.05,
        shaft_radius=0.018,
        scale=arrow_length
    )

    arrow_tf = pv.Arrow(
        start=(0.0, 0.0, 0.0),
        direction=p_end,
        tip_length=0.22,
        tip_radius=0.05,
        shaft_radius=0.018,
        scale=arrow_length
    )

    plotter.add_mesh(arrow_t0, color="green")
    plotter.add_mesh(arrow_tf, color="blue")

    plotter.add_axes(
        line_width=3,
        cone_radius=0.08,
        shaft_length=0.75,
        tip_length=0.25,
        ambient=0.5,
        xlabel="x",
        ylabel="y",
        zlabel="z"
    )

    if show_grid:
        plotter.show_bounds(
            grid="front",
            location="outer",
            all_edges=True,
            color="black",
            xtitle="x",
            ytitle="y",
            ztitle="z",
            font_size=18
        )

    plotter.add_text(title, font_size=15, position="upper_left", color="black")

    plotter.add_legend(
        labels=[
            [r"$\mathbf{p}(t)$", "red"],
            [r"$t_0$", "green"],
            [r"$t_f$", "blue"]
        ],
        bcolor="white",
        border=False,
        face="circle",
        size=(0.14, 0.12)
    )

    A = result["A"]
    planes = get_velocity_planes(case)

    plane_colors = {
        "xy": "black",
        "xz": "black",
        "yz": "black"
    }

    for plane in planes:
        points, vectors = build_velocity_field_on_plane(A, plane=plane, n=11, lim=1.0)

        mesh = pv.PolyData(points)
        mesh["velocity"] = vectors

        glyphs = mesh.glyph(
            orient="velocity",
            scale="velocity",
            factor=0.3
        )

        plotter.add_mesh(
            glyphs,
            color=plane_colors.get(plane, "black"),
            opacity=0.85
        )

    plotter.camera_position = "iso"
    plotter.camera.zoom(1.15)

    plotter.add_key_event("s", save_current_screenshot)
    plotter.add_key_event("q", lambda: plotter.close())

    print("  Controles interactivos")
    print("  " + "─" * 52)
    print("    Presiona 's' para guardar screenshot en ResultsG/")
    print("    Presiona 'q' para cerrar la ventana")
    print()

    if save_screenshot:
        save_current_screenshot()

    plotter.show(auto_close=True)


# =============================================================================
# 7. SWITCHES
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # MODO DE SIMULACIÓN
    # -------------------------------------------------------------------------
    MODE = "deterministic"      # "deterministic" o "stochastic"

    # -------------------------------------------------------------------------
    # CASO DE FLUJO
    # -------------------------------------------------------------------------
    CASE = "shear_xy"
    # CASE = "shear_xz"
    # CASE = "extensional_xy"
    # CASE = "rotation_z"
    # CASE = "mixed_shear_stretch"
    # CASE = "custom"

    # -------------------------------------------------------------------------
    # PARÁMETROS DE LA PARTÍCULA
    # -------------------------------------------------------------------------
    r = 50/7 # Monoraphidium griffithii
    theta0_deg = 60
    phi0_deg = 60

    # -------------------------------------------------------------------------
    # TIEMPO
    # -------------------------------------------------------------------------
    t_final = 200.0
    n_points = 4000   # solo determinista

    # Para el caso estocástico
    dt = 1e-3
    D = 1e-5
    random_seed = 42

    # -------------------------------------------------------------------------
    # PARÁMETROS DEL FLUJO
    # -------------------------------------------------------------------------
    flow_params = {
        "gamma": 1,
        "epsilon_dot": 1.0,
        "omega": 1.0,
        "s": 0.1
    }

    grad_u_custom = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=float)

    # -------------------------------------------------------------------------
    # PARÁMETROS ÓPTICOS (modelo lumped)
    # -------------------------------------------------------------------------
    optical_params = {
        # Fuerza de alineación óptica efectiva [1/s]
        "lambda_opt": 0.005,

        # Dirección de polarización lineal de la pinza
        # Ejemplo típico: polarización en x
        "e_pol": np.array([1, 0, 0], dtype=float),
    }

    # -------------------------------------------------------------------------
    # EJECUCIÓN
    # -------------------------------------------------------------------------
    if MODE == "deterministic":
        result = run_case(
            case=CASE,
            r=r,
            theta0_deg=theta0_deg,
            phi0_deg=phi0_deg,
            t_final=t_final,
            n_points=n_points,
            flow_params=flow_params,
            grad_u_custom=grad_u_custom,
            optical_params=optical_params
        )

    elif MODE == "stochastic":
        result = run_case_stochastic(
            case=CASE,
            r=r,
            theta0_deg=theta0_deg,
            phi0_deg=phi0_deg,
            t_final=t_final,
            dt=dt,
            D=D,
            flow_params=flow_params,
            grad_u_custom=grad_u_custom,
            random_seed=random_seed,
            renormalize_every_step=True,
            optical_params=optical_params
        )

    else:
        raise ValueError("MODE debe ser 'deterministic' o 'stochastic'.")

    fig, axs = plot_jeffery_4subplots(
        result,
        flow_params=flow_params,
        use_unwrapped_phi=True,
        figsize=(12, 8),
        save=True
    )
    plt.show()

    plot_trajectory_on_sphere_interactive(
        result,
        save_screenshot=False
    )