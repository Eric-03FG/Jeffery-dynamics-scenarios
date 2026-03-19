# =============================================================================
# JEFFERY DETERMINISTA GENERALIZADO EN PYTHON
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
#   2) Permite usar campos de flujo deterministas predefinidos:
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
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pyvista as pv


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


# =============================================================================
# 4. INTERFAZ ÚNICA DE SIMULACIÓN
# =============================================================================

def run_case(
    case="shear_xy",
    r=5.0,
    theta0=1.1,
    phi0=0.3,
    t_final=40.0,
    n_points=4000,
    flow_params=None,
    grad_u_custom=None,
    method="RK45",
    rtol=1e-10,
    atol=1e-10
):
    """
    Interfaz principal para correr un caso determinista de Jeffery.
    """
    if flow_params is None:
        flow_params = {}

    beta = bretherton_parameter(r)
    p0 = p_from_angles(theta0, phi0)
    A = build_grad_u(case, flow_params=flow_params, grad_u_custom=grad_u_custom)

    t_span = (0.0, t_final)
    t_eval = np.linspace(0.0, t_final, n_points)

    def rhs(t, p):
        return jeffery_rhs(t, p, beta, A)

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
        "theta0": theta0,
        "phi0": phi0,
        "t_final": t_final,
        "flow_params": flow_params,
        "solver_result": sol
    }

    return result


# =============================================================================
# 5. GRÁFICA 2x2
# =============================================================================

def plot_jeffery_4subplots(result, use_unwrapped_phi=True, figsize=(12, 8)):
    """
    Genera una figura con 4 subplots:
      1) p_x, p_y, p_z vs tiempo
      2) theta vs tiempo [grados]
      3) phi vs tiempo [grados]
      4) ||p|| vs tiempo
    """
    t = result["t"]
    theta_deg = result["theta_deg"]
    phi_deg = result["phi_deg_unwrapped"] if use_unwrapped_phi else result["phi_deg_wrapped"]
    beta = result["beta"]
    r = result["aspect_ratio"]
    case = result["case"]

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f"Jeffery determinista | case = {case} | r = {r:.4f} | beta = {beta:.6f}",
        fontsize=14
    )

    axs[0, 0].plot(t, result["px"], label=r"$p_x$", linewidth=2)
    axs[0, 0].plot(t, result["py"], label=r"$p_y$", linewidth=2)
    axs[0, 0].plot(t, result["pz"], label=r"$p_z$", linewidth=2)
    axs[0, 0].set_xlabel(r"$t$")
    axs[0, 0].set_ylabel(r"$\mathbf{p}(t)$")
    axs[0, 0].set_title("Componentes de p vs tiempo")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(t, theta_deg, label=r"$\theta(t)$", linewidth=2)
    axs[0, 1].set_xlabel(r"$t$")
    axs[0, 1].set_ylabel(r"$\theta\ [^\circ]$")
    axs[0, 1].set_title(r"$\theta$ vs tiempo")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(t, phi_deg, label=r"$\phi(t)$", linewidth=2)
    axs[1, 0].set_xlabel(r"$t$")
    axs[1, 0].set_ylabel(r"$\phi\ [^\circ]$")
    axs[1, 0].set_title(r"$\phi$ vs tiempo")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(t, result["norms"], label=r"$\|\mathbf{p}\|$", linewidth=2)
    axs[1, 1].set_xlabel(r"$t$")
    axs[1, 1].set_ylabel(r"$\|\mathbf{p}\|$")
    axs[1, 1].set_title(r"Norma de $\mathbf{p}$ vs tiempo")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    return fig, axs


# =============================================================================
# 6. VISUALIZACIÓN 3D INTERACTIVA NATIVA CON PYVISTA
# =============================================================================

def plot_trajectory_on_sphere_interactive(
    result,
    title=None,
    screenshot_path="jeffery_sphere.png",
    save_screenshot=True,
    show_grid=True,
    sphere_opacity=0.22
):
    """
    Abre una ventana 3D interactiva nativa con PyVista, sin navegador.

    Parámetros
    ----------
    result : dict
        Resultado devuelto por run_case(...)

    title : str o None
        Título mostrado en la ventana.

    screenshot_path : str
        Ruta del archivo PNG para guardar screenshot.

    save_screenshot : bool
        Si es True, guarda una captura automáticamente.

    show_grid : bool
        Si es True, muestra una caja/ejes de referencia elegantes.

    sphere_opacity : float
        Opacidad de la esfera unitaria.
    """
    P = result["P"]
    case = result["case"]

    if title is None:
        title = rf"$\mathbf{{p}}(t)$ orbit on $S^2$  |  {case}"

    # -------------------------------------------------------------------------
    # Crear ventana
    # -------------------------------------------------------------------------
    plotter = pv.Plotter(window_size=(1000, 780))
    plotter.set_background("white")

    # -------------------------------------------------------------------------
    # Esfera unitaria
    # -------------------------------------------------------------------------
    sphere = pv.Sphere(radius=1.0, theta_resolution=80, phi_resolution=80)
    plotter.add_mesh(
        sphere,
        color="lightgray",
        opacity=sphere_opacity,
        smooth_shading=True,
        specular=0.15,
        show_edges=False
    )

    # -------------------------------------------------------------------------
    # Trayectoria de p(t)
    # -------------------------------------------------------------------------
    polyline = pv.lines_from_points(P)
    plotter.add_mesh(
        polyline,
        color="red",
        line_width=4,
        label="p(t)"
    )

    # -------------------------------------------------------------------------
    # Punto inicial y final
    # -------------------------------------------------------------------------
    p_start = P[0]
    p_end = P[-1]

    start = pv.PolyData(p_start.reshape(1, 3))
    end = pv.PolyData(p_end.reshape(1, 3))

    plotter.add_mesh(
        start,
        color="green",
        point_size=16,
        render_points_as_spheres=True,
        label="t0"
    )

    plotter.add_mesh(
        end,
        color="blue",
        point_size=16,
        render_points_as_spheres=True,
        label="tf"
    )

    # Etiquetas visuales junto a los puntos
    plotter.add_point_labels(
        np.array([p_start, p_end]),
        ["t0", "tf"],
        font_size=18,
        point_size=0,
        text_color="black",
        shape=None,
        margin=0
    )

    # -------------------------------------------------------------------------
    # Ejes más bonitos
    # -------------------------------------------------------------------------
    plotter.add_axes(
        line_width=3,
        cone_radius=0.08,
        shaft_length=0.75,
        tip_length=0.25,
        ambient=0.5,
        xlabel="x",
        ylabel="y",
        zlabel="z",
        labels_off=False
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
            font_size=12
        )

    # -------------------------------------------------------------------------
    # Texto
    # -------------------------------------------------------------------------
    plotter.add_text(title, font_size=12, position="upper_left", color="black")

    # Leyenda visual
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

    # -------------------------------------------------------------------------
    # Cámara
    # -------------------------------------------------------------------------
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.15)

    # -------------------------------------------------------------------------
    # Guardar screenshot automáticamente (antes o durante show)
    # -------------------------------------------------------------------------
    if save_screenshot:
        # render previo para asegurar captura correcta
        plotter.show(auto_close=False, interactive_update=False)
        plotter.screenshot(screenshot_path)
        print(f"Screenshot guardado en: {screenshot_path}")

        # mantener abierta la ventana interactiva
        plotter.show()
    else:
        plotter.show()


# =============================================================================
# 7. SWITCHES
# =============================================================================

if __name__ == "__main__":

    # CASE = "shear_xy"
    CASE = "shear_xz"
    # CASE = "extensional_xy"
    # CASE = "rotation_z"
    # CASE = "mixed_shear_stretch"
    # CASE = "custom"

    r = 50/7 # Monoraphidium griffithii
    theta0 = 15
    phi0 = 60
    t_final = 80.0
    n_points = 4000

    flow_params = {
        "gamma": 1,
        "epsilon_dot": 1.0,
        "omega": 1.0,
        "s": 1
    }

    grad_u_custom = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=float)

    result = run_case(
        case=CASE,
        r=r,
        theta0=theta0,
        phi0=phi0,
        t_final=t_final,
        n_points=n_points,
        flow_params=flow_params,
        grad_u_custom=grad_u_custom
    )

    fig, axs = plot_jeffery_4subplots(result, use_unwrapped_phi=True, figsize=(12, 8))
    plt.show()

    # Esto abre una ventana 3D nativa
    plot_trajectory_on_sphere_interactive(result)