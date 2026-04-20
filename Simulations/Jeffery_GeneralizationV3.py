# =============================================================================
# JEFFERY EXTENDIDO EN CANAL CILÍNDRICO CON PRESIÓN VARIABLE
# -----------------------------------------------------------------------------
# Este script resuelve la dinámica acoplada de una partícula/nadador alargado
# en un canal cilíndrico con perfil axial de Poiseuille:
#
#   x_dot = u(x,t) + v_s p
#   p_dot = W p + beta * (E p - (p^T E p) p)
#
# donde:
#   - x(t) = posición del centro de la partícula
#   - p(t) = vector unitario de orientación
#   - u(x,t) = campo de velocidad local del flujo
#   - v_s = velocidad propia de nado de la partícula/célula
#           (si v_s = 0, la partícula es pasiva;
#            si v_s > 0, hay autopropulsión en la dirección p)
#   - E = (A + A^T)/2 = tensor de tasa de deformación
#   - W = (A - A^T)/2 = tensor de rotación local del flujo
#   - A = grad(u)
#   - beta = parámetro de Bretherton
#
# El caso considerado en esta versión es exclusivamente:
#
#   CASE = "poiseuille_cylindrical"
#
# con:
#   - canal cilíndrico de radio R = H/2
#   - velocidad axial Umax(t) con dependencia temporal opcional
#   - restricción lateral mediante proyección a la pared y deslizamiento tangencial
#
# Se generan las siguientes salidas:
#   1) Panel 2x3: componentes de orientación, ángulos y posición
#   2) Proyecciones espaciales: y(x), z(x), z(y)
#   3) Variación temporal de presión/caudal: g(t), Umax(t)
#   4) Visualización interactiva 3D de la trayectoria de orientación sobre S^2
# =============================================================================

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


# =============================================================================
# 1. FUNCIONES AUXILIARES BÁSICAS
# =============================================================================

def bretherton_parameter(r):
    """
    Calcula el parámetro de Bretherton para un esferoide prolato:

        beta = (r^2 - 1) / (r^2 + 1)

    donde r = a/b es la razón de aspecto.
    """
    r = float(r)
    return (r**2 - 1.0) / (r**2 + 1.0)


def p_from_angles(theta, phi):
    """
    Convierte coordenadas esféricas (theta, phi) a un vector unitario cartesiano.

    Convención:
      - theta: ángulo polar medido desde +z
      - phi  : ángulo azimutal medido desde +x en el plano xy
    """
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], dtype=float)


def angles_from_p(p):
    """
    Convierte un vector cartesiano p = [px, py, pz] a ángulos (theta, phi).
    """
    px, py, pz = p
    theta = np.arccos(np.clip(pz, -1.0, 1.0))
    phi = np.arctan2(py, px)
    return theta, phi


def vector_norm(v):
    """Devuelve la norma euclidiana de un vector."""
    return np.sqrt(np.sum(v**2))


def normalize_vector(v):
    """
    Normaliza un vector 3D.
    """
    v = np.asarray(v, dtype=float)
    n = vector_norm(v)
    if n < 1e-15:
        raise ValueError("No se puede normalizar un vector de norma cercana a cero.")
    return v / n


def unwrap_angle_deg(angle_deg):
    """
    Desenvuelve un ángulo en grados para eliminar saltos artificiales de ±180°.
    """
    return np.rad2deg(np.unwrap(np.deg2rad(angle_deg)))


def wrap_angle_deg(angle_deg):
    """
    Lleva un ángulo en grados al intervalo [-180, 180].
    """
    return (angle_deg + 180.0) % 360.0 - 180.0


# =============================================================================
# 2. DESCRIPCIÓN DEL PERFIL TEMPORAL DE PRESIÓN / CAUDAL
# =============================================================================

def smooth_square_wave(t, omega, sharpness=20.0):
    """
    Aproximación suave de una onda cuadrada basada en tanh(sin()).
    """
    return np.tanh(sharpness * np.sin(omega * t))


def ramp_wave(t, period):
    """
    Onda tipo ramp periódica en el intervalo [0, 1).
    """
    if period <= 0:
        raise ValueError("period debe ser positivo.")
    return (t % period) / period


def pulse_train(t, period, duty=0.3):
    """
    Tren de pulsos periódicos.

    Devuelve 1 durante una fracción 'duty' del periodo y 0 el resto del tiempo.
    """
    if period <= 0:
        raise ValueError("period debe ser positivo.")
    if not (0.0 < duty <= 1.0):
        raise ValueError("duty debe pertenecer al intervalo (0,1].")

    phase = (t % period) / period
    return 1.0 if phase < duty else 0.0


def pressure_gradient_time_function(t, flow_params):
    """
    Devuelve una función adimensional g(t) tal que:

        -dp/dx(t) = G0 * g(t)

    y por tanto:

        Umax(t) = Umax0 * g(t)

    Perfiles soportados:
      - constant
      - sinusoidal
      - cosine
      - square
      - ramp
      - pulse
      - custom
    """
    profile = flow_params.get("time_profile", "constant")

    eps = flow_params.get("modulation_amplitude", 0.0)
    omega = flow_params.get("omega_t", 1.0)
    phase = flow_params.get("phase_t", 0.0)

    period = flow_params.get(
        "period",
        2.0 * np.pi / omega if abs(omega) > 1e-15 else 1.0
    )

    sharpness = flow_params.get("square_sharpness", 20.0)
    duty = flow_params.get("duty_cycle", 0.3)

    if profile == "constant":
        return 1.0

    elif profile == "sinusoidal":
        return 1.0 + eps * np.sin(omega * t + phase)

    elif profile == "cosine":
        return 1.0 + eps * np.cos(omega * t + phase)

    elif profile == "square":
        t_shift = t + phase / max(abs(omega), 1e-15)
        return 1.0 + eps * smooth_square_wave(t_shift, omega, sharpness=sharpness)

    elif profile == "ramp":
        t_shift = t + phase / max(abs(omega), 1e-15)
        return 1.0 + eps * (2.0 * ramp_wave(t_shift, period) - 1.0)

    elif profile == "pulse":
        t_shift = t + phase / max(abs(omega), 1e-15)
        return 1.0 + eps * pulse_train(t_shift, period, duty=duty)

    elif profile == "custom":
        custom_fun = flow_params.get("custom_time_function", None)
        if custom_fun is None or not callable(custom_fun):
            raise ValueError(
                "Para time_profile='custom' debes proporcionar "
                "flow_params['custom_time_function'] = f(t)."
            )
        return float(custom_fun(t))

    else:
        raise ValueError(
            f"time_profile='{profile}' no reconocido. "
            "Usa: constant, sinusoidal, cosine, square, ramp, pulse o custom."
        )


def Umax_of_t(t, flow_params):
    """
    Devuelve la velocidad axial máxima instantánea:

        Umax(t) = Umax0 * g(t)

    con g(t) dada por pressure_gradient_time_function(...).

    Si allow_flow_reversal=False, el valor se recorta a cero.
    """
    Umax0 = flow_params.get("Umax", 1.0)
    g_t = pressure_gradient_time_function(t, flow_params)
    U_t = Umax0 * g_t

    allow_reversal = flow_params.get("allow_flow_reversal", True)
    if not allow_reversal:
        U_t = max(U_t, 0.0)

    return U_t


def pressure_gradient_of_t(t, flow_params):
    """
    Devuelve el gradiente de presión axial efectivo:

        -dp/dx(t) = G0 * g(t)

    En este modelo reducido, se toma proporcional a Umax(t).
    """
    G0 = flow_params.get("pressure_gradient_ref", 1.0)
    g_t = pressure_gradient_time_function(t, flow_params)
    G_t = G0 * g_t

    allow_reversal = flow_params.get("allow_flow_reversal", True)
    if not allow_reversal:
        G_t = max(G_t, 0.0)

    return G_t


# =============================================================================
# 3. DESCRIPCIÓN DEL FLUJO EN EL CANAL
# =============================================================================

def get_flow_latex(case, flow_params):
    """
    Devuelve una representación LaTeX simple del campo de velocidad.
    """
    if case != "poiseuille_cylindrical":
        return r"$\mathbf{u}=\mathbf{u}(\mathbf{x},t)$"

    H = flow_params.get("H", 2.0)
    R = 0.5 * H
    profile = flow_params.get("time_profile", "constant")

    if profile == "constant":
        Umax = flow_params.get("Umax", 1.0)
        return (
            rf"$\mathbf{{u}}=\left("
            rf"{Umax}\left[1-\frac{{y^2+z^2}}{{({R})^2}}\right],\,0,\,0"
            rf"\right)$"
        )
    else:
        return (
            rf"$\mathbf{{u}}=\left("
            rf"U_{{\max}}(t)\left[1-\frac{{y^2+z^2}}{{({R})^2}}\right],\,0,\,0"
            rf"\right)$"
        )


def velocity_field(x, t, case, flow_params=None):
    """
    Devuelve la velocidad local u(x,t).

    Para el caso implementado:
        u = ( Umax(t) * [1 - (y^2+z^2)/R^2], 0, 0 )
    """
    if flow_params is None:
        flow_params = {}

    if case != "poiseuille_cylindrical":
        raise ValueError("Esta versión del código solo soporta case='poiseuille_cylindrical'.")

    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("x debe ser un vector de tamaño 3.")

    H = flow_params.get("H", 2.0)
    R = 0.5 * H

    Umax_t = Umax_of_t(t, flow_params)

    y, z = x[1], x[2]
    rho2 = y**2 + z**2

    # Se limita el radio efectivo para evitar velocidades no físicas
    rho2_eff = min(rho2, R**2)

    ux = Umax_t * (1.0 - rho2_eff / R**2)

    return np.array([ux, 0.0, 0.0], dtype=float)


def grad_u_local(x, t, case, flow_params=None):
    """
    Devuelve grad(u)(x,t).

    Para Poiseuille cilíndrico:
        u_x = Umax(t) * (1 - (y^2 + z^2)/R^2)

    por tanto:
        dux/dy = -2 Umax(t) y / R^2
        dux/dz = -2 Umax(t) z / R^2
    """
    if flow_params is None:
        flow_params = {}

    if case != "poiseuille_cylindrical":
        raise ValueError("Esta versión del código solo soporta case='poiseuille_cylindrical'.")

    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("x debe ser un vector de tamaño 3.")

    H = flow_params.get("H", 2.0)
    R = 0.5 * H

    Umax_t = Umax_of_t(t, flow_params)

    y, z = x[1], x[2]

    dux_dy = -2.0 * Umax_t * y / (R**2)
    dux_dz = -2.0 * Umax_t * z / (R**2)

    return np.array([
        [0.0, dux_dy, dux_dz],
        [0.0, 0.0,    0.0   ],
        [0.0, 0.0,    0.0   ]
    ], dtype=float)


# =============================================================================
# 4. ECUACIÓN DE JEFFERY
# =============================================================================

def jeffery_drift(p, beta, A):
    """
    Evalúa el término determinista de Jeffery:

        p_dot = W p + beta * (E p - (p^T E p) p)
    """
    p = normalize_vector(np.asarray(p, dtype=float))

    E = 0.5 * (A + A.T)
    W = 0.5 * (A - A.T)

    Ep = E @ p
    pEp = float(p @ Ep)

    return W @ p + beta * (Ep - pEp * p)


# =============================================================================
# 5. CONDICIONES DE BORDE DEL CANAL
# =============================================================================

def project_to_cylinder_and_slide(x, p, flow_params):
    """
    Proyecta la posición al interior del cilindro y elimina la componente normal
    saliente de la orientación si la trayectoria alcanza la pared.

    Canal cilíndrico:
        radio efectivo R = H/2 - wall_margin
    """
    x = np.asarray(x, dtype=float).copy()
    p = normalize_vector(np.asarray(p, dtype=float))

    H = flow_params.get("H", 2.0)
    wall_margin = flow_params.get("wall_margin", 0.0)

    R = 0.5 * H - wall_margin
    if R <= 0:
        raise ValueError("El radio efectivo del canal debe ser positivo.")

    yz = x[1:3]
    rho = np.linalg.norm(yz)

    if rho > R:
        n = yz / rho

        # Proyección geométrica a la pared cilíndrica
        x[1:3] = R * n

        # Deslizamiento tangencial: se elimina componente normal saliente
        p_n = p[1] * n[0] + p[2] * n[1]
        if p_n > 0.0:
            p[1] -= p_n * n[0]
            p[2] -= p_n * n[1]

            if np.linalg.norm(p) < 1e-14:
                p = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                p = normalize_vector(p)

    return x, p


def enforce_channel_bc(x, p, flow_params):
    """
    Aplica la restricción lateral del canal cilíndrico.
    """
    x = np.asarray(x, dtype=float).copy()
    p = normalize_vector(np.asarray(p, dtype=float))
    return project_to_cylinder_and_slide(x, p, flow_params)


# =============================================================================
# 6. INTEGRADOR DETERMINISTA EXTENDIDO
# =============================================================================

def run_case_extended(
    case="poiseuille_cylindrical",
    r=5.0,
    theta0_deg=63.0,
    phi0_deg=17.0,
    x0=(0.0, 0.0, 0.0),
    t_final=40.0,
    n_points=4000,
    flow_params=None,
    vs=0.0,
    n_substeps=5,
    renormalize_each_substep=True,
):
    """
    Resuelve la dinámica acoplada extendida mediante RK4 con substepping interno:

        x_dot = u(x,t) + vs p
        p_dot = Jeffery(p, grad(u)(x,t))

    El estado está dado por:
        y = [x, y, z, px, py, pz]

    Aspectos numéricos:
      - integración RK4 libre en cada subpaso
      - aplicación de restricción del canal al final de cada subpaso
      - normalización periódica del vector de orientación
    """
    if flow_params is None:
        flow_params = {}

    if case != "poiseuille_cylindrical":
        raise ValueError("Esta versión solo conserva case='poiseuille_cylindrical'.")

    if n_points < 2:
        raise ValueError("n_points debe ser al menos 2.")
    if n_substeps < 1:
        raise ValueError("n_substeps debe ser al menos 1.")

    theta0 = np.deg2rad(theta0_deg)
    phi0 = np.deg2rad(phi0_deg)

    beta = bretherton_parameter(r)
    p0 = p_from_angles(theta0, phi0)
    x0 = np.asarray(x0, dtype=float)

    if x0.shape != (3,):
        raise ValueError("x0 debe ser un vector/lista de longitud 3.")

    # Aplicación inicial de la restricción geométrica
    x0, p0 = enforce_channel_bc(x0, p0, flow_params)

    t = np.linspace(0.0, t_final, n_points)
    dt_output = t[1] - t[0]
    h = dt_output / n_substeps

    X = np.zeros((n_points, 3), dtype=float)
    P = np.zeros((n_points, 3), dtype=float)

    X[0] = x0
    P[0] = normalize_vector(p0)

    def rhs_extended(ti, x, p):
        p = normalize_vector(p)

        u = velocity_field(x, ti, case, flow_params=flow_params)
        A = grad_u_local(x, ti, case, flow_params=flow_params)

        x_dot = u + vs * p
        p_dot = jeffery_drift(p, beta, A)

        return x_dot, p_dot

    def rk4_free_step(ti, x, p, h_local):
        """
        Ejecuta un paso RK4 sin imponer restricciones dentro de las etapas internas.
        """
        k1x, k1p = rhs_extended(ti, x, p)

        k2x, k2p = rhs_extended(
            ti + 0.5 * h_local,
            x + 0.5 * h_local * k1x,
            p + 0.5 * h_local * k1p
        )

        k3x, k3p = rhs_extended(
            ti + 0.5 * h_local,
            x + 0.5 * h_local * k2x,
            p + 0.5 * h_local * k2p
        )

        k4x, k4p = rhs_extended(
            ti + h_local,
            x + h_local * k3x,
            p + h_local * k3p
        )

        x_new = x + (h_local / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        p_new = p + (h_local / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)

        return x_new, p_new

    for i in range(n_points - 1):
        x_curr = X[i].copy()
        p_curr = normalize_vector(P[i])

        ti = t[i]

        for m in range(n_substeps):
            t_sub = ti + m * h

            x_new, p_new = rk4_free_step(t_sub, x_curr, p_curr, h)

            if renormalize_each_substep:
                p_new = normalize_vector(p_new)

            x_new, p_new = enforce_channel_bc(x_new, p_new, flow_params)

            x_curr = x_new
            p_curr = p_new

        X[i + 1] = x_curr
        P[i + 1] = normalize_vector(p_curr)

    theta = np.zeros(n_points)
    phi = np.zeros(n_points)
    norms = np.linalg.norm(P, axis=1)

    for i in range(n_points):
        theta[i], phi[i] = angles_from_p(P[i])

    # Registro adicional de Umax(t), g(t) y gradiente de presión
    g_t = np.array([pressure_gradient_time_function(ti, flow_params) for ti in t])
    Umax_t = np.array([Umax_of_t(ti, flow_params) for ti in t])

    result = {
        "mode": "deterministic_extended",
        "case": case,
        "t": t,
        "X": X,
        "x": X[:, 0],
        "y": X[:, 1],
        "z": X[:, 2],
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
        "theta0_deg": theta0_deg,
        "phi0_deg": phi0_deg,
        "x0": x0,
        "t_final": t_final,
        "flow_params": flow_params,
        "vs": vs,
        "dt_output": dt_output,
        "dt_internal": h,
        "n_substeps": n_substeps,
        "g_t": g_t,
        "Umax_t": Umax_t,
    }

    return result


# =============================================================================
# 7. FIGURAS 2x3: ORIENTACIÓN Y POSICIÓN
# =============================================================================

def plot_jeffery_panels(
    result,
    flow_params=None,
    use_unwrapped_phi=True,
    figsize=(14, 8.5),
    save=True,
    folder="ResultsG",
    filename=None
):
    """
    Genera un panel 2x3 con:
      - p_x, p_y, p_z
      - theta(t)
      - phi(t)
      - x(t)
      - y(t)
      - z(t)
    """
    t = result["t"]
    theta_deg = result["theta_deg"]
    phi_deg = result["phi_deg_unwrapped"] if use_unwrapped_phi else result["phi_deg_wrapped"]

    beta = result["beta"]
    r = result["aspect_ratio"]
    case = result["case"]
    vs = result.get("vs", None)

    flow_latex = get_flow_latex(case, flow_params or result.get("flow_params", {}))

    if save:
        os.makedirs(folder, exist_ok=True)

    fig, axs = plt.subplots(2, 3, figsize=figsize)

    suptitle = (
        rf"$\dot{{\mathbf{{x}}}}=\mathbf{{u}}+v_s\mathbf{{p}},\ "
        rf"\dot{{\mathbf{{p}}}}=\mathrm{{Jeffery}}$"
        "\n"
        rf"$r={r:.3f}$, $\beta={beta:.3f}$, $v_s={vs:.3f}$"
        "\n"
        rf"{flow_latex}"
    )
    fig.suptitle(suptitle, fontsize=13)

    # Componentes de p
    axs[0, 0].plot(t, result["px"], label=r"$p_x$", linewidth=2)
    axs[0, 0].plot(t, result["py"], label=r"$p_y$", linewidth=2)
    axs[0, 0].plot(t, result["pz"], label=r"$p_z$", linewidth=2)
    axs[0, 0].set_xlabel(r"$t$")
    axs[0, 0].set_ylabel(r"$\mathbf{p}(t)$")
    axs[0, 0].set_title(r"Components of $\mathbf{p}(t)$")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # theta
    axs[0, 1].plot(t, theta_deg, label=r"$\theta(t)$", linewidth=2)
    axs[0, 1].set_xlabel(r"$t$")
    axs[0, 1].set_ylabel(r"$\theta\ [^\circ]$")
    axs[0, 1].set_title(r"$\theta(t)$")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    # phi
    axs[0, 2].plot(t, phi_deg, label=r"$\phi(t)$", linewidth=2)
    axs[0, 2].set_xlabel(r"$t$")
    axs[0, 2].set_ylabel(r"$\phi\ [^\circ]$")
    axs[0, 2].set_title(r"$\phi(t)$")
    axs[0, 2].grid(True, alpha=0.3)
    axs[0, 2].legend()

    # x(t)
    axs[1, 0].plot(t, result["x"], label=r"$x(t)$", linewidth=2)
    axs[1, 0].set_xlabel(r"$t$")
    axs[1, 0].set_ylabel(r"$x$")
    axs[1, 0].set_title(r"$x(t)$")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # y(t)
    axs[1, 1].plot(t, result["y"], label=r"$y(t)$", linewidth=2)
    axs[1, 1].set_xlabel(r"$t$")
    axs[1, 1].set_ylabel(r"$y$")
    axs[1, 1].set_title(r"$y(t)$")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    # z(t)
    axs[1, 2].plot(t, result["z"], label=r"$z(t)$", linewidth=2)
    axs[1, 2].set_xlabel(r"$t$")
    axs[1, 2].set_ylabel(r"$z$")
    axs[1, 2].set_title(r"$z(t)$")
    axs[1, 2].grid(True, alpha=0.3)
    axs[1, 2].legend()

    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"jeffery_extended_{case}_r{r:.3f}_beta{beta:.3f}.png"

        path = os.path.join(folder, filename)
        fig.savefig(path, dpi=500, bbox_inches="tight")
        print(f"\nFigura guardada en: {path}")

    return fig, axs


# =============================================================================
# 8. FIGURA DE PROYECCIONES ESPACIALES
# =============================================================================

def plot_spatial_projections(
    result,
    figsize=(13.5, 4.6),
    save=True,
    folder="ResultsG",
    filename=None,
    lw=2.0,
    border_lw=1.4,
    start_ms=5.5,
    end_ms=5.5,
    print_summary=True
):
    """
    Proyecciones espaciales de la trayectoria:

      (1) y vs x
      (2) z vs x
      (3) z vs y

    donde:
      - x es la coordenada axial
      - y y z son las coordenadas transversales
      - la frontera transversal del canal satisface: y^2 + z^2 = R^2

    En las dos primeras gráficas se muestran los límites ±R como referencia.
    En la tercera gráfica se muestra la sección circular del canal.
    """
    if "X" not in result:
        raise ValueError("Este resultado no contiene trayectoria espacial X(t).")

    x = result["x"]
    y = result["y"]
    z = result["z"]

    case = result["case"]
    r = result["aspect_ratio"]
    beta = result["beta"]
    vs = result.get("vs", 0.0)
    flow_params = result.get("flow_params", {})

    H = flow_params.get("H", 2.0)
    wall_margin = flow_params.get("wall_margin", 0.0)
    R = 0.5 * H - wall_margin

    def format_position(vec):
        return f"({vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f})"

    def radial_distance(vec):
        return np.sqrt(vec[1]**2 + vec[2]**2)

    x_start = np.array([x[0], y[0], z[0]], dtype=float)
    x_end = np.array([x[-1], y[-1], z[-1]], dtype=float)

    if print_summary:
        print()
        print("╔" + "═" * 58 + "╗")
        print("║{:^58s}║".format("TRAYECTORIA ESPACIAL"))
        print("╚" + "═" * 58 + "╝")
        print(f"  Caso: {case}")
        print()

        print("  Parámetros geométricos del canal")
        print("  " + "─" * 52)
        print(f"    H (diámetro)       : {H:.4f}")
        print(f"    wall_margin        : {wall_margin:.4f}")
        print(f"    R efectivo         : {R:.4f}")
        print()

        print("  Estado inicial")
        print("  " + "─" * 52)
        print(f"    Posición cartesiana: {format_position(x_start)}")
        print(f"    Distancia radial   : {radial_distance(x_start):.4f}")
        print()

        print("  Estado final")
        print("  " + "─" * 52)
        print(f"    Posición cartesiana: {format_position(x_end)}")
        print(f"    Distancia radial   : {radial_distance(x_end):.4f}")
        print()

        print("  Extensión de la trayectoria")
        print("  " + "─" * 52)
        print(f"    x_min, x_max       : ({np.min(x):.4f}, {np.max(x):.4f})")
        print(f"    y_min, y_max       : ({np.min(y):.4f}, {np.max(y):.4f})")
        print(f"    z_min, z_max       : ({np.min(z):.4f}, {np.max(z):.4f})")
        print()

    if save:
        os.makedirs(folder, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # (1) y vs x
    axs[0].plot(x, y, linewidth=lw, label="trajectory")
    axs[0].plot(x[0], y[0], "o", ms=start_ms, label=r"$t_0$")
    axs[0].plot(x[-1], y[-1], "s", ms=end_ms, label=r"$t_f$")
    axs[0].axhline(+R, linestyle="--", linewidth=border_lw, color="k")
    axs[0].axhline(-R, linestyle="--", linewidth=border_lw, color="k")
    axs[0].set_xlabel(r"$x$")
    axs[0].set_ylabel(r"$y$")
    axs[0].set_title(r"$y$ vs $x$")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # (2) z vs x
    axs[1].plot(x, z, linewidth=lw, label="trajectory")
    axs[1].plot(x[0], z[0], "o", ms=start_ms, label=r"$t_0$")
    axs[1].plot(x[-1], z[-1], "s", ms=end_ms, label=r"$t_f$")
    axs[1].axhline(+R, linestyle="--", linewidth=border_lw, color="k")
    axs[1].axhline(-R, linestyle="--", linewidth=border_lw, color="k")
    axs[1].set_xlabel(r"$x$")
    axs[1].set_ylabel(r"$z$")
    axs[1].set_title(r"$z$ vs $x$")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    # (3) z vs y
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    y_circ = R * np.cos(theta)
    z_circ = R * np.sin(theta)

    axs[2].plot(y_circ, z_circ, linestyle="--", linewidth=border_lw, color="k", label="channel boundary")
    axs[2].plot(y, z, linewidth=lw, label="trajectory")
    axs[2].plot(y[0], z[0], "o", ms=start_ms, label=r"$t_0$")
    axs[2].plot(y[-1], z[-1], "s", ms=end_ms, label=r"$t_f$")
    axs[2].set_xlabel(r"$y$")
    axs[2].set_ylabel(r"$z$")
    axs[2].set_title(r"$z$ vs $y$")
    axs[2].grid(True, alpha=0.3)
    axs[2].axis("equal")
    axs[2].legend()

    fig.suptitle(
        rf"Spatial projections | {case}, $r={r:.3f}$, $\beta={beta:.3f}$, $v_s={vs:.3f}$",
        fontsize=13
    )

    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"spatial_projections_{case}_r{r:.3f}_beta{beta:.3f}_vs{vs:.3f}.png"

        path = os.path.join(folder, filename)
        fig.savefig(path, dpi=500, bbox_inches="tight")
        print(f"\nFigura guardada en: {path}")

    return fig, axs

# =============================================================================
# 9. FIGURA DE VARIACIÓN DE PRESIÓN / CAUDAL
# =============================================================================

def plot_flow_modulation(
    result,
    figsize=(7.5, 4.5),
    save=True,
    folder="ResultsG",
    filename=None,
    lw=2.0
):
    """
    Grafica una sola señal temporal:

        Umax(t) = U0 * g(t)

    donde:
      - U0 = flow_params["Umax"]
      - g(t) depende de time_profile
    """
    t = result["t"]
    Umax_t = result["Umax_t"]

    flow_params = result.get("flow_params", {})
    profile = flow_params.get("time_profile", "constant")

    if save:
        os.makedirs(folder, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(t, Umax_t, linewidth=lw)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$U_{\max}(t)$")
    ax.set_title(r"Flow modulation")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf"$U_{{\max}}(t)$ | profile = {profile}",
        fontsize=13
    )

    plt.tight_layout()

    if save:
        if filename is None:
            filename = f"flow_modulation_{profile}.png"

        path = os.path.join(folder, filename)
        fig.savefig(path, dpi=500, bbox_inches="tight")
        print(f"\nFigura guardada en: {path}")

    return fig, ax


# =============================================================================
# 10. VISUALIZACIÓN 3D INTERACTIVA DE LA ORIENTACIÓN
# =============================================================================

def plot_trajectory_on_sphere_interactive(
    result,
    title=None,
    screenshot_path=None,
    save_screenshot=False,
    show_grid=True,
    sphere_opacity=0.22
):
    """
    Visualiza la trayectoria del vector de orientación p(t) sobre la esfera unitaria.

    La figura muestra:
      - la esfera unitaria S^2
      - la trayectoria de orientación p(t)
      - el estado inicial t0
      - el estado final tf

    En esta versión no se dibuja el campo vectorial del flujo sobre la esfera,
    ya que la orientación está acoplada a una trayectoria espacial y el flujo
    depende de la posición.
    """
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
    r_val = result["aspect_ratio"]
    beta_val = result["beta"]

    if title is None:
        title = (
            rf"$\mathbf{{p}}(t)\ \mathrm{{on}}\ S^2$"
            "\n"
            + rf"$\mathrm{{orientation\ coupled\ to\ spatial\ trajectory}}$"
            + "\n"
            + rf"$r = {r_val:.3f},\ \beta = {beta_val:.3f}$"
        )

    plotter = pv.Plotter(window_size=(1000, 780))
    plotter.set_background("white")

    results_dir = Path("ResultsG")
    results_dir.mkdir(parents=True, exist_ok=True)

    def build_filename():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return results_dir / f"{case}_{timestamp}.png"

    def save_current_screenshot():
        filepath = Path(screenshot_path) if screenshot_path else build_filename()
        plotter.screenshot(str(filepath))
        print(f"\n📸 Screenshot guardado en: {filepath}")

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
    # Trayectoria de orientación
    # -------------------------------------------------------------------------
    polyline = pv.lines_from_points(P)
    plotter.add_mesh(polyline, color="red", line_width=4, label="p(t)")

    p_start = normalize_vector(P[0])
    p_end = normalize_vector(P[-1])

    print()
    print("╔" + "═" * 58 + "╗")
    print("║{:^58s}║".format("JEFFERY ORBIT — ORIENTACIONES"))
    print("╚" + "═" * 58 + "╝")
    print(f"  Caso: {case}")
    print("  Modo: deterministic_extended")
    print()

    print("  Parámetros de la partícula")
    print("  " + "─" * 52)
    print(f"    Aspect ratio (r = a/b) : {r_val:.4f}")
    print(f"    Bretherton (β)         : {beta_val:.6f}")

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

    # -------------------------------------------------------------------------
    # Marcadores inicial y final
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Ejes y anotaciones
    # -------------------------------------------------------------------------
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
# 11. EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    CASE = "poiseuille_cylindrical"

    # -------------------------------------------------------------------------
    # PARÁMETROS DE LA PARTÍCULA
    # -------------------------------------------------------------------------
    r = 50 / 7  # Relación de aspecto
    # Condiciones iniciales
    theta0_deg = 5.0
    phi0_deg = 5.0
    x0 = np.array([0.0, 0.5, 0.5], dtype=float)

    # -------------------------------------------------------------------------
    # PARÁMETROS TEMPORALES
    # -------------------------------------------------------------------------
    t_final = 1000.0
    n_points = 20000
    n_substeps = 5

    # -------------------------------------------------------------------------
    # PARÁMETROS DEL NADADOR
    # -------------------------------------------------------------------------
    vs = 0.05  # Propulsión

    # -------------------------------------------------------------------------
    # PARÁMETROS DEL FLUJO
    # -------------------------------------------------------------------------
    flow_params = {
        # Canal cilíndrico:
        #   R = H/2 - wall_margin
        "Umax": 1.0,          # Umax(t) = Umax * g(t)
        "H": 2.0,             # diámetro del canal
        "wall_margin": 0.0,   # radio efectivo: R = H/2 - wall_margin

        # Perfil temporal g(t):
        #   Umax(t) = Umax * g(t)
        "time_profile": "constant",   # constant, sinusoidal, cosine, square, ramp, pulse, custom
        "modulation_amplitude": 1.0,    # epsilon

        # sinusoidal / cosine / square:
        #   g(t) = 1 + epsilon * f(omega_t * t + phase_t)
        "omega_t": 0.05,
        "phase_t": 0.0,

        # ramp / pulse:
        "period": 8.0 * np.pi,

        # square:
        "square_sharpness": 20.0,

        # pulse:
        "duty_cycle": 0.2,

        # si False: Umax(t) = max(Umax(t), 0)
        "allow_flow_reversal": True,

        # custom:
        #   g(t) = custom_time_function(t)
        "custom_time_function": None,
    }

    # -------------------------------------------------------------------------
    # EJECUCIÓN DE LA SIMULACIÓN
    # -------------------------------------------------------------------------
    result = run_case_extended(
        case=CASE,
        r=r,
        theta0_deg=theta0_deg,
        phi0_deg=phi0_deg,
        x0=x0,
        t_final=t_final,
        n_points=n_points,
        flow_params=flow_params,
        vs=vs,
        n_substeps=n_substeps,
        renormalize_each_substep=True,
    )

    # -------------------------------------------------------------------------
    # FIGURA 1: ORIENTACIÓN Y POSICIÓN
    # -------------------------------------------------------------------------
    fig1, axs1 = plot_jeffery_panels(
        result,
        flow_params=flow_params,
        use_unwrapped_phi=True,
        figsize=(14, 8.5),
        save=True,
        filename=f"jeffery_extended_{CASE}.png"
    )
    plt.show()

    # -------------------------------------------------------------------------
    # FIGURA 2: PROYECCIONES ESPACIALES
    # -------------------------------------------------------------------------
    fig2, axs2 = plot_spatial_projections(
        result,
        figsize=(13.5, 4.6),
        save=True,
        filename=f"spatial_projections_{CASE}.png"
    )
    plt.show()

    # -------------------------------------------------------------------------
    # FIGURA 3: VARIACIÓN DE PRESIÓN / CAUDAL
    # -------------------------------------------------------------------------
    fig3, ax3 = plot_flow_modulation(
        result,
        figsize=(7.5, 4.5),
        save=True,
        filename="flow_modulation.png"
    )
    plt.show()

    # -------------------------------------------------------------------------
    # VISUALIZACIÓN INTERACTIVA 3D DE LA ORIENTACIÓN
    # -------------------------------------------------------------------------
    plot_trajectory_on_sphere_interactive(
        result,
        save_screenshot=False
    )