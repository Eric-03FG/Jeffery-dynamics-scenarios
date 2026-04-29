import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
 

#Velocity gradient 3x3 matrix G= nabla u, G[i, j] =du_x/dx_j. (x,y,z).
def grad_u_simple_shear(gamma: float, plane: str) -> np.ndarray:
    G = np.zeros((3,3), dtype=float)
    if plane == "xy":
        G[0,1] = gamma
    elif plane == "xz":
        G[0,2] = gamma
    else:
        raise ValueError("plane must be 'xy' or 'xz'")
    return G

#Jeffery shape parameter, c is the aspect ratio of the sheroid
def lambda_ar(c: float) -> float:
    c2 = c * c
    lamb = (c2 - 1.0) / (c2 + 1.0)
    return lamb


def decompose_grad_u(G):
    E = 0.5 * (G + G.T)
    W = 0.5 * (G - G.T)
    return E, W

def normalize_rows(P):
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    return P / norms

#Shperical angles to unit vector. 
def sph_to_vec(theta: float, phi: float) -> np.ndarray:
    return np.array([
        np.sin(theta) * np.cos(phi), #d1 = rcos(phi), r = sin(theta)
        np.sin(theta) * np.sin(phi), #d2 = rsin(phi), r = sin(theta)
        np.cos(theta)  #d3, d3´=-theta´sin(theta)
    ], dtype=float)

#Unit vector -> (theta, phi). Uses atan2 for robust quadrant.
def vec_to_sph(p: np.ndarray) -> tuple[float, float]:
    p = np.asarray(p, dtype=float)
    p = p / np.linalg.norm(p)
    theta = np.arccos(np.clip(p[2], -1.0, 1.0))
    phi = np.arctan2(p[1], p[0])
    return theta, phi

   
#Jeffery equation in vector form: 
#dp = (Omega x d) + lam (E p - (p^T E p) p)

def jeffery_rhs_vector(t: float, d: np.ndarray, E: np.ndarray, W: np.ndarray, lam: float) -> np.ndarray:

    d = np.asarray(d, dtype=float)
    Ep = E @ d #Ejk d
    Wp = W @ d #omega * d
    dEp = d @ Ep 
    dp = Wp + lam * (Ep - dEp * d)
    return dp


def integrate_director_vector(
    d0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray, 
    E: np.ndarray,
    W: np.ndarray,
    lam: float,
    rtol: float = 1e-9,
    atol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray,np.ndarray]:

    sol = solve_ivp(
        fun=lambda t, p: jeffery_rhs_vector(t, p, E, W, lam),
        t_span=t_span,
        y0=np.asarray(d0, dtype=float),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol
    )

    P_raw = sol.y.T  # (N,3)
    norm_raw = np.linalg.norm(P_raw, axis=1)

    P = normalize_rows(P_raw)
    return sol.t, P, norm_raw

 
#returns [dtheta/dt, dphi/dt]
def jeffery_rhs_theta_phi(t: float, y: np.ndarray, E: np.ndarray, W: np.ndarray, B: float) -> np.ndarray:
   
    theta, phi = y

    # unitary vector
    d = sph_to_vec(theta, phi)
    d_dot = jeffery_rhs_vector(t, d, E, W, B)

    # d3' = -theta'sin(theta) -> thta'=-d3'/sin(theta)
    sin_th = np.sin(theta)
    eps = 1e-12
    if abs(sin_th) < eps:
        theta_dot = 0.0
        phi_dot = 0.0
        return np.array([theta_dot, phi_dot], dtype=float)

    theta_dot = -d_dot[2] / sin_th

    # d1 * d2' - d1' * d2 = phi' sin^2(theta)
    sin2 = sin_th * sin_th
    phi_dot = (d[0] * d_dot[1] - d_dot[0] * d[1]) / sin2

    return np.array([theta_dot, phi_dot], dtype=float)


def integrate_theta_phi(
    y0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray,
    E: np.ndarray,
    W: np.ndarray,
    B: float,
    rtol: float = 1e-9,
    atol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    sol = solve_ivp(
        fun=lambda t, y: jeffery_rhs_theta_phi(t, y, E, W, B),
        t_span=t_span,
        y0=np.asarray(y0, dtype=float),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol
    )
    Y = sol.y.T  # (N,2)
    return sol.t, Y

#Extensional flow
def grad_u_extensional(epsilon_dot: float, plane: str) -> np.ndarray:
    G = np.zeros((3,3), dtype=float)
    if plane == "xy":
        G[0,0]= epsilon_dot
        G[1,1]=-epsilon_dot
    elif plane == "xz":
        G[0,0]= epsilon_dot
        G[2,2]=-epsilon_dot
    elif plane == "yz":
        G[1,1] = epsilon_dot
        G[2,2] = -epsilon_dot
    else:
        raise ValueError("plane must be 'xy', 'xz' or 'yz'")
    return G

#Rigid rotation, all contribution is in the antisimetric part
def grad_u_rotation(omega: float, plane: str) -> np.ndarray:
    G = np.zeros((3,3),dtype=float)
    if plane == "xy": # u = (-omega*y, omega*x,0), rotates around z
        G[0,1]=-omega
        G[1,0]=omega
    elif plane == "xz": # u = (omega*z,0, -omega*x), rotates around y
        G[0,2] = omega
        G[2,0] = -omega
    elif plane == "yz": # u = (0,-omega*z, omega*y), rotates around x
        G[1,2] = -omega
        G[2,1] = omega
    else:
        raise ValueError("plane must be 'xy', 'xz' or 'yz'")
    return G

#mixed flow shear + stretching
def grad_u_mixed_shear_stretch(gamma: float, s: float) -> np.ndarray: 
    G=np.zeros((3,3), dtype=float)
    G[0,0] = s
    G[1,1]= -s
    G[0,2] = gamma
    return G

#Normalize a 3D vector, n =||v|| 
def normalize_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-15:
        raise ValueError("Too close to 0")
    return v/n

#continuous angles in radians
def unwrap_angle_rad(angle_rad:np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(angle_rad, dtype=float))

#continuous angles in degrees
def unwrap_angle_deg(angle_deg:np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(angle_deg, dtype=float))))

#converts directors P of (N,3) to angles
def directors_to_angles(P:np.ndarray):
    P=np.asarray(P, dtype=float)
    d1 = P[:,0]
    d2 = P[:,1]
    d3 = P[:,2]

    theta_rad = np.arccos(np.clip(d3,-1.0,1.0))
    phi_rad_wrapped = np.arctan2(d2,d1)
    phi_rad_unwrapped = np.unwrap(phi_rad_wrapped)

    return theta_rad, phi_rad_wrapped, phi_rad_unwrapped


def optical_angular_velocity_effective(
    p: np.ndarray,
    e_pol: np.ndarray = None,
    lambda_opt: float = 0.0
) -> np.ndarray:
    """

    Model:
        U(p) = -(1/2) * Delta_alpha * E0^2 * (p · e_pol)^2

    Uses:

        lambda_opt   [1/s]

    Effective angular velocity:

        Omega_opt = lambda_opt * (p · e_pol) * (p x e_pol)

    Where:
        - p      : unit vector
        - e_pol  : unidirectional linear polarization
        - lambda_opt > 0 promotes alignment with ± e_pol

    Interpretation:
    - If p is already aligned with e_pol, then p x e_pol = 0
      and there is no additional optical rotation.
    - If p is tilted, an effective rotation occurs that tends
      to align p with the polarization direction.

    Parameters:
    - p: direction vector.
    - e_pol: Polarization direction. If None: x = (1,0,0).
    - lambda_opt: effective optical alignment intensity [1/s].

    Return:
    - Omega_opt: Effective angular velocity.
    """
    p = normalize_vector(np.asarray(p, dtype=float))

    if e_pol is None:
        e_pol = np.array([1.0, 0.0, 0.0], dtype=float)
    e_pol = normalize_vector(np.asarray(e_pol, dtype=float))

    c = float(np.dot(p, e_pol))
    Omega_opt = lambda_opt * c * np.cross(p, e_pol)
    return Omega_opt

def optical_alignment_drift(
    p: np.ndarray,
    e_pol: np.ndarray = None,
    lambda_opt: float = 0.0
) -> np.ndarray:
    """
    Devuelve la contribución óptica directa a p_dot.

    uses:
        p_dot_opt = Omega_opt x p

    with:
        Omega_opt = lambda_opt * (p · e_pol) * (p x e_pol)

    gets:

        p_dot_opt = lambda_opt * (p·e_pol) * [ e_pol - (p·e_pol) p ]

    """
    p = normalize_vector(np.asarray(p, dtype=float))

    Omega_opt = optical_angular_velocity_effective(
        p=p,
        e_pol=e_pol,
        lambda_opt=lambda_opt
    )

    p_dot_opt = np.cross(Omega_opt, p)
    return p_dot_opt


def jeffery_rhs_vector_optical(
    t: float,
    d: np.ndarray,
    E: np.ndarray,
    W: np.ndarray,
    lam: float,
    e_pol: np.ndarray = None,
    lambda_opt: float = 0.0
) -> np.ndarray:
    """
    right-hand side of the angular momentum equation:

        d_dot = d_dot_Jeffery + d_dot_opt

    Where:
        d_dot_Jeffery = W d + lam (E d - (d^T E d) d)
        d_dot_opt     = optical_alignment_drift(...)

    """
    d = normalize_vector(np.asarray(d, dtype=float))

    d_dot_jeffery = jeffery_rhs_vector(t, d, E, W, lam)
    d_dot_opt = optical_alignment_drift(
        p=d,
        e_pol=e_pol,
        lambda_opt=lambda_opt
    )

    return d_dot_jeffery + d_dot_opt