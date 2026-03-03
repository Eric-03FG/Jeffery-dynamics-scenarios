import numpy as np
from scipy.integrate import solve_ivp

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
) -> tuple[np.ndarray, np.ndarray]:

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