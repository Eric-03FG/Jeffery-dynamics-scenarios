import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pyvista as pv

import src_Jeffery as jf


# ============================================================
# 1) shear parameter
# ============================================================
gamma = 1.0
plane = "xy"              # "xy" o "xz" <------------ change

# ============================================================
# 1) extensional flow parameter
# ============================================================
epsilon_dot = 1.0
#plane = "yz"              # "xy", "xz" o "yz" <-------- change

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
t0, tf = 0.0, 3000.0
n_points = 4000
t_eval = np.linspace(t0, tf, n_points)

# ============================================================
# 5) PSEUDO Optic torque
# ============================================================
USE_OPTICAL = True        # Apagar o prender el torque xd (TRUE or FALSE)
lambda_opt = 5e-3
e_pol = np.array([0.0, 1.0, 0.0], dtype=float)

# ============================================================
# 6) Shear
# ============================================================
G = jf.grad_u_simple_shear(gamma, plane=plane)
#G = jf.grad_u_extensional(epsilon_dot, plane=plane)
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

drift_h = jf.jeffery_rhs_vector(0.0, d0, E, W, lam)
drift_o = jf.optical_alignment_drift(d0, e_pol=e_pol, lambda_opt=lambda_opt)

print("||drift_hidrodinamico|| =", np.linalg.norm(drift_h))
print("||drift_optico||       =", np.linalg.norm(drift_o))
print("ratio opt/hid =", np.linalg.norm(drift_o) / np.linalg.norm(drift_h))

# ============================================================
# 8) Graphics 2x2
# ============================================================

if plane == "xy":
    flow_label_plain = f"u = ({gamma} y, 0, 0)"
    flow_label_math = rf"$\mathbf{{u}} = ({gamma}y,\ 0,\ 0)$"
elif plane == "xz":
    flow_label_plain = f"u = ({gamma} z, 0, 0)"
    flow_label_math = rf"$\mathbf{{u}} = ({gamma}z,\ 0,\ 0)$"
else:
    flow_label_plain = "simple shear"
    flow_label_math = r"$\mathbf{u}$"

"""
if plane == "xy":
    flow_label_plain = f"u = ({epsilon_dot} x, -{epsilon_dot} y, 0)"
    flow_label_math = rf"$\mathbf{{u}} = ({epsilon_dot}x,\ -{epsilon_dot}y,\ 0)$"
elif plane == "xz":
    flow_label_plain = f"u = ({epsilon_dot} x, 0, -{epsilon_dot} z)"
    flow_label_math = rf"$\mathbf{{u}} = ({epsilon_dot}x,\ 0,\ -{epsilon_dot}z)$"
elif plane == "yz":
    flow_label_plain = f"u = (0, {epsilon_dot} y, -{epsilon_dot} z)"
    flow_label_math = rf"$\mathbf{{u}} = (0,\ {epsilon_dot}y,\ -{epsilon_dot}z)$"
else:
    flow_label_plain = "extensional flow"
    flow_label_math = r"$\mathbf{u}$"
    """

optical_label = "ON" if USE_OPTICAL else "OFF"

# Title
suptitle = (
    rf"$\dot{{\mathbf{{p}}}}\;|\; r = {r:.3f},\ \beta = {lam:.3f},\ \Lambda = {lambda_opt:.0e}$"
    "\n"
    + flow_label_math
    + "\n"
    + rf"$\mathrm{{optical}} = {optical_label}$"
)

fig, axs = plt.subplots(2, 2, figsize=(11, 7.4))

# -------------------------
# (1) Components of p(t)
# -------------------------
axs[0, 0].plot(sol.t, d1, label=r"$p_x$", linewidth=1.7)
axs[0, 0].plot(sol.t, d2, label=r"$p_y$", linewidth=1.7)
axs[0, 0].plot(sol.t, d3, label=r"$p_z$", linewidth=1.7)
axs[0, 0].set_title(r"Components of $\mathbf{p}(t)$", fontsize=12)
axs[0, 0].set_xlabel(r"$t$", fontsize=11)
axs[0, 0].set_ylabel(r"$\mathbf{p}(t)$", fontsize=11)
axs[0, 0].grid(True, alpha=0.8)
axs[0, 0].legend(fontsize=10)

# -------------------------
# (2) theta(t)
# -------------------------
axs[0, 1].plot(sol.t, theta_deg, label=r"$\theta(t)$", linewidth=1.8)
axs[0, 1].set_title(r"$\theta(t)$", fontsize=12)
axs[0, 1].set_xlabel(r"$t$", fontsize=11)
axs[0, 1].set_ylabel(r"$\theta\ [^\circ]$", fontsize=11)
axs[0, 1].grid(True, alpha=0.8)
axs[0, 1].legend(fontsize=10)

# -------------------------
# (3) phi(t)
# -------------------------
axs[1, 0].plot(sol.t, phi_deg, label=r"$\phi(t)$", linewidth=1.8)
axs[1, 0].set_title(r"$\phi(t)$", fontsize=12)
axs[1, 0].set_xlabel(r"$t$", fontsize=11)
axs[1, 0].set_ylabel(r"$\phi\ [^\circ]$", fontsize=11)
axs[1, 0].grid(True, alpha=0.8)
axs[1, 0].legend(fontsize=10)

# -------------------------
# (4) norma ||p(t)||
# -------------------------
norm_p = np.linalg.norm(P, axis=1)

axs[1, 1].plot(sol.t, np.maximum(norm_error, 1e-16), label=r"$\|\mathbf{p}\|$", linewidth=1.6)
axs[1, 1].set_yscale("log")
axs[1, 1].set_title(r"$\|\mathbf{p}(t)\|$", fontsize=12)
axs[1, 1].set_xlabel(r"$t$", fontsize=11)
axs[1, 1].set_ylabel(r"$\|\mathbf{p}\|$", fontsize=11)
axs[1, 1].grid(True, alpha=0.8)
axs[1, 1].legend(fontsize=10)

fig.suptitle(suptitle, fontsize=15, y=0.98)


plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

# ============================================================
# 9) 3D
# ============================================================
plotter = pv.Plotter(window_size=(1100, 800))
plotter.set_background("white")


if plane == "xy":
    flow_label_3d = f"u = ({gamma:g}y, 0, 0)"
elif plane == "xz":
    flow_label_3d = f"u = ({gamma:g}z, 0, 0)"
else:
    flow_label_3d = "simple shear"
"""
if plane == "xy":
    flow_label_3d = f"u = ({epsilon_dot:g}x, -{epsilon_dot:g}y, 0)"
elif plane == "xz":
    flow_label_3d = f"u = ({epsilon_dot:g}x, 0, -{epsilon_dot:g}z)"
elif plane == "yz":
    flow_label_3d = f"u = (0, {epsilon_dot:g}y, -{epsilon_dot:g}z)"
else:
    flow_label_3d = "extensional flow"
    """

optical_label = "ON" if USE_OPTICAL else "OFF"
lambda_display = lambda_opt if USE_OPTICAL else 0.0

title_3d = (
    "p(t) on S^2\n"
    f"{flow_label_3d}\n"
    f"r = {r:.3f}, beta = {lam:.3f}, Lambda = {lambda_display:g}\n"
    f"optical = {optical_label}"
)

# ------------------------------------------------------------
# Unit sphere
# ------------------------------------------------------------
sphere = pv.Sphere(
    radius=1.0,
    theta_resolution=80,
    phi_resolution=80
)

plotter.add_mesh(
    sphere,
    color="lightgray",
    opacity=0.22,
    smooth_shading=True,
    specular=0.15,
    show_edges=False
)

# ------------------------------------------------------------
# Trayectory p(t)
# ------------------------------------------------------------
polyline = pv.lines_from_points(P)

plotter.add_mesh(
    polyline,
    color="red",
    line_width=4,
    label="p(t)"
)

# ------------------------------------------------------------
# Initial and final vector
# ------------------------------------------------------------
p0 = jf.normalize_vector(P[0])
pf = jf.normalize_vector(P[-1])

theta0_check = np.rad2deg(np.arccos(np.clip(p0[2], -1.0, 1.0)))
phi0_check = np.rad2deg(np.arctan2(p0[1], p0[0]))

thetaf_check = np.rad2deg(np.arccos(np.clip(pf[2], -1.0, 1.0)))
phif_check = np.rad2deg(np.arctan2(pf[1], pf[0]))

print("p0 =", p0)
print("||p0|| =", np.linalg.norm(p0))
print("theta0 =", theta0_check, "deg")
print("phi0 =", phi0_check, "deg")
print()
print("pf =", pf)
print("||pf|| =", np.linalg.norm(pf))
print("thetaf =", thetaf_check, "deg")
print("phif =", phif_check, "deg")

# initial and final points
start_pt = pv.PolyData(p0.reshape(1, 3))
end_pt = pv.PolyData(pf.reshape(1, 3))

plotter.add_mesh(
    start_pt,
    color="green",
    point_size=8,
    render_points_as_spheres=True
)

plotter.add_mesh(
    end_pt,
    color="blue",
    point_size=8,
    render_points_as_spheres=True
)

# Initial arrow (green)
arrow_t0 = pv.Arrow(
    start=(0.0, 0.0, 0.0),
    direction=p0,
    tip_length=0.22,
    tip_radius=0.05,
    shaft_radius=0.018,
    scale=0.98
)

plotter.add_mesh(
    arrow_t0,
    color="green"
)

# Final arrow (blue)
arrow_tf = pv.Arrow(
    start=(0.0, 0.0, 0.0),
    direction=pf,
    tip_length=0.22,
    tip_radius=0.05,
    shaft_radius=0.018,
    scale=0.98
)

plotter.add_mesh(
    arrow_tf,
    color="blue"
)

# ------------------------------------------------------------
# Velocity field
# ------------------------------------------------------------
grid = np.linspace(-1.0, 1.0, 11)

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
            X = np.array([a, b, 0.0])

        u = G @ X

        points.append(X)
        vectors.append(u)

points = np.array(points)
vectors = np.array(vectors)

mesh = pv.PolyData(points)
mesh["velocity"] = vectors

glyphs = mesh.glyph(
    orient="velocity",
    scale="velocity",
    factor=0.3
)

plotter.add_mesh(
    glyphs,
    color="black",
    opacity=0.85
)

# ------------------------------------------------------------
# axys and boxes
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# left upper text
# ------------------------------------------------------------
plotter.add_text(
    title_3d,
    font_size=16,
    position="upper_left",
    color="black"
)

# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------
plotter.add_legend(
    labels=[
        ["p(t)", "red"],
        ["t_0", "green"],
        ["t_f", "blue"]
    ],
    bcolor="white",
    border=False,
    face="circle",
    size=(0.16, 0.14)
)


plotter.camera_position = "iso"
plotter.camera.zoom(1.15)

plotter.show()