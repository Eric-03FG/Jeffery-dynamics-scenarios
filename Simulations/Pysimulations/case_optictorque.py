import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pyvista as pv

import src_Jeffery as jf


# ============================================================
# 1) shear parameter
# ============================================================
gamma = 1.0
plane = "xy"              # "xy" o "xz"

# ============================================================
# 2) Particle parameters
# ============================================================

r = 50 / 7                # Monoraphidium griffithii ~ 50 x 7 um
lam = jf.lambda_ar(r)

# ============================================================
# 3) Initial condition
# ============================================================
theta0_deg = 25.0
phi0_deg = 25.0

theta0 = np.deg2rad(theta0_deg)
phi0 = np.deg2rad(phi0_deg)
d0 = jf.sph_to_vec(theta0, phi0)

# ============================================================
# 4) Time
# ============================================================
t0, tf = 0.0, 2100.0
n_points = 4000
t_eval = np.linspace(t0, tf, n_points)

# ============================================================
# 5) PSEUDO Optic torque
# ============================================================
USE_OPTICAL = False        # Apagar o prender el torque xd (TRUE or FALSE)
lambda_opt = 5.0e-3
e_pol = np.array([1.0, 0.0, 0.0], dtype=float)

# ============================================================
# 6) Shear
# ============================================================
G = jf.grad_u_simple_shear(gamma, plane=plane)
E, W = jf.decompose_grad_u(G)

# ============================================================
# 7) Integration
# ============================================================
if USE_OPTICAL:
    sol = solve_ivp(
        fun=lambda t, d: jf.jeffery_rhs_vector_optical(
            t, d, E, W, lam,
            e_pol=e_pol,
            lambda_opt=lambda_opt
        ),
        t_span=(t0, tf),
        y0=d0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12
    )
else:
    sol = solve_ivp(
        fun=lambda t, d: jf.jeffery_rhs_vector(
            t, d, E, W, lam
        ),
        t_span=(t0, tf),
        y0=d0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12
    )

if not sol.success:
    raise RuntimeError(f"La integración falló: {sol.message}")

P_raw = sol.y.T
norm_raw = np.linalg.norm(P_raw, axis=1)
P = jf.normalize_rows(P_raw)

d1 = P[:, 0]
d2 = P[:, 1]
d3 = P[:, 2]

theta_rad, phi_rad_wrapped, phi_rad_unwrapped = jf.directors_to_angles(P)
theta_deg = np.rad2deg(theta_rad)
phi_deg = np.rad2deg(phi_rad_unwrapped)
norm_error = np.abs(norm_raw - 1.0)

# ============================================================
# 8) Graphics 2x2
# ============================================================
if plane == "xy":
    flow_label = f"u = ({gamma} y, 0, 0)"
elif plane == "xz":
    flow_label = f"u = ({gamma} z, 0, 0)"
else:
    flow_label = "simple shear"

optical_label = "ON" if USE_OPTICAL else "OFF"

title = (
    f"Simple shear\n"
    f"{flow_label}\n"
    f"r = {r:.3f}, beta = {lam:.3f}, optical = {optical_label}, "
    f"lambda_opt = {lambda_opt:.3e}"
)

fig, axs = plt.subplots(2, 2, figsize=(11, 7))

axs[0, 0].plot(sol.t, d1, label="d1")
axs[0, 0].plot(sol.t, d2, label="d2")
axs[0, 0].plot(sol.t, d3, label="d3")
axs[0, 0].set_xlabel("t")
axs[0, 0].set_ylabel("d(t)")
axs[0, 0].grid(True)
axs[0, 0].legend()
axs[0, 0].set_title(title)

axs[0, 1].plot(sol.t, theta_deg, label=r"$\theta(t)$")
axs[0, 1].set_xlabel("t")
axs[0, 1].set_ylabel(r"$\theta$ [deg]")
axs[0, 1].grid(True)
axs[0, 1].legend()

axs[1, 0].plot(sol.t, phi_deg, label=r"$\phi(t)$")
axs[1, 0].set_xlabel("t")
axs[1, 0].set_ylabel(r"$\phi$ [deg]")
axs[1, 0].grid(True)
axs[1, 0].legend()

axs[1, 1].plot(sol.t, np.maximum(norm_error, 1e-16), label=r"$|\|d\|-1|$")
axs[1, 1].set_yscale("log")
axs[1, 1].set_xlabel("t")
axs[1, 1].set_ylabel(r"$|\|d\|-1|$")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# ============================================================
# 9) 3D 
# ============================================================
plotter = pv.Plotter(window_size=(1100, 800))
plotter.set_background("white")

# Esfera unitaria
sphere = pv.Sphere(radius=1.0, theta_resolution=80, phi_resolution=80)
plotter.add_mesh(
    sphere,
    color="lightgray",
    opacity=0.22,
    smooth_shading=True,
    show_edges=False
)

# Trayectoria p(t)
polyline = pv.lines_from_points(P)
plotter.add_mesh(
    polyline,
    color="red",
    line_width=4
)

# Vector inicial y final
p0 = jf.normalize_vector(P[0])
pf = jf.normalize_vector(P[-1])

arr0 = pv.Arrow(
    start=(0.0, 0.0, 0.0),
    direction=p0,
    scale=0.9,
    shaft_radius=0.02,
    tip_radius=0.05,
    tip_length=0.12
)
plotter.add_mesh(arr0, color="green")

arrf = pv.Arrow(
    start=(0.0, 0.0, 0.0),
    direction=pf,
    scale=0.9,
    shaft_radius=0.02,
    tip_radius=0.05,
    tip_length=0.12
)
plotter.add_mesh(arrf, color="blue")

# Puntos inicial y final
start_pt = pv.PolyData(np.array([p0]))
end_pt = pv.PolyData(np.array([pf]))
plotter.add_mesh(start_pt, color="green", point_size=14, render_points_as_spheres=True)
plotter.add_mesh(end_pt, color="blue", point_size=14, render_points_as_spheres=True)

# Campo de velocidad en un plano
grid = np.linspace(-1.0, 1.0, 9)
points = []
vectors = []

for a in grid:
    for b in grid:
        if plane == "xy":
            X = np.array([a, b, 0.0])
        elif plane == "xz":
            X = np.array([a, 0.0, b])
        else:
            X = np.array([a, b, 0.0])

        u = G @ X
        points.append(X)
        vectors.append(u)

points = np.array(points)
vectors = np.array(vectors)

mags = np.linalg.norm(vectors, axis=1)
max_mag = np.max(mags) if np.max(mags) > 0 else 1.0
vectors_plot = vectors / max_mag * 0.35

pdata = pv.PolyData(points)
pdata["vectors"] = vectors_plot

arrows = pdata.glyph(
    orient="vectors",
    scale=False,
    factor=1.0,
    geom=pv.Arrow(
        start=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        scale=1.0,
        shaft_radius=0.01,
        tip_radius=0.03,
        tip_length=0.10
    )
)
plotter.add_mesh(arrows, color="black")

# Ejes y rejilla
plotter.add_axes()
plotter.show_grid()


text = (
    "p(t) on S^2\n"
    f"{flow_label}\n"
    f"r = {r:.3f}, beta = {lam:.3f}\n"
    f"optical = {optical_label}, lambda_opt = {lambda_opt:.3e}\n"
    f"e_pol = ({e_pol[0]:.3f}, {e_pol[1]:.3f}, {e_pol[2]:.3f})"
)
plotter.add_text(text, position="upper_left", font_size=12, color="black")

legend_entries = [
    ["p(t)", "red"],
    ["t_0", "green"],
    ["t_f", "blue"],
]
plotter.add_legend(legend_entries, bcolor="white", face="circle")

plotter.show()