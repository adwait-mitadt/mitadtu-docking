# ============================================================
#  DEEP LEARNING-AUGMENTED GNC — MINI PROJECT
#  Development of Orbital Rendezvous and GNC Algorithm
#  for Satellite Refueling
#
#  All 4 steps combined for Google Colab
#  Run each cell in order, or Runtime > Run All
# ============================================================


# ╔══════════════════════════════════════════════════════════╗
# ║          CELL 1 — Install dependencies                  ║
# ╚══════════════════════════════════════════════════════════╝
# (Run this cell first, then restart runtime if prompted)

# %pip install numpy scipy matplotlib networkx --quiet


# ╔══════════════════════════════════════════════════════════╗
# ║          CELL 2 — Imports                               ║
# ╚══════════════════════════════════════════════════════════╝

import numpy as np
import matplotlib
matplotlib.use('Agg')                       # non-interactive backend for Colab
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.linalg import expm
from scipy.integrate import odeint
from numpy.linalg import inv
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

print("All imports successful.")


# ╔══════════════════════════════════════════════════════════╗
# ║   STEP 1 — Hohmann Transfer Orbit Calculation           ║
# ╚══════════════════════════════════════════════════════════╝

# ── Constants ──────────────────────────────────────────────
MU        = 398600.4418   # Earth's gravitational parameter (km^3/s^2)
R_EARTH   = 6378.137      # Earth radius, WGS-84 (km)

def compute_hohmann(alt_initial_km: float, alt_final_km: float):
    """
    Compute Hohmann transfer parameters between two circular orbits.

    Parameters
    ----------
    alt_initial_km : altitude of initial orbit above Earth surface (km)
    alt_final_km   : altitude of final orbit above Earth surface (km)

    Returns
    -------
    dict with all transfer parameters
    """
    if alt_initial_km <= 0 or alt_final_km <= 0:
        raise ValueError("Altitudes must be positive.")
    if alt_initial_km >= alt_final_km:
        raise ValueError("Initial orbit must be lower than final orbit.")

    r1 = alt_initial_km + R_EARTH          # radius from Earth centre (km)
    r2 = alt_final_km   + R_EARTH

    a_t = (r1 + r2) / 2.0                  # semi-major axis of transfer ellipse

    v1       = np.sqrt(MU / r1)            # circular velocity at r1
    v2       = np.sqrt(MU / r2)            # circular velocity at r2
    v_t_peri = np.sqrt(2*MU/r1 - MU/a_t)  # transfer ellipse velocity at perigee
    v_t_apo  = np.sqrt(2*MU/r2 - MU/a_t)  # transfer ellipse velocity at apogee

    dv1    = v_t_peri - v1
    dv2    = v2 - v_t_apo
    dv_tot = abs(dv1) + abs(dv2)
    tof    = np.pi * np.sqrt(a_t**3 / MU)  # time of flight (half period) in seconds
    e_t    = (r2 - r1) / (r2 + r1)         # eccentricity of transfer ellipse

    return dict(
        r1=r1, r2=r2, a_trans=a_t, e_trans=e_t,
        v1=v1, v2=v2, v_trans_peri=v_t_peri, v_trans_apo=v_t_apo,
        dv1=dv1, dv2=dv2, dv_total=dv_tot, tof=tof
    )


def kepler_position(a, e, mu, times):
    """
    Solve Kepler's equation for an array of times using Newton-Raphson.
    Returns (x_array, y_array) in km.
    """
    n = np.sqrt(mu / a**3)
    xs, ys = [], []
    for t in times:
        M = n * t
        E = M
        for _ in range(50):
            dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
            E -= dE
            if abs(dE) < 1e-12:
                break
        nu = 2.0 * np.arctan2(
            np.sqrt(1 + e) * np.sin(E / 2),
            np.sqrt(1 - e) * np.cos(E / 2)
        )
        r = a * (1 - e**2) / (1 + e * np.cos(nu))
        xs.append(r * np.cos(nu))
        ys.append(r * np.sin(nu))
    return np.array(xs), np.array(ys)


def plot_hohmann(params, n_points=200, save_gif=True):
    """Plot static Hohmann transfer and save an animation GIF."""
    r1, r2 = params['r1'], params['r2']
    a_t, e_t, tof = params['a_trans'], params['e_trans'], params['tof']

    theta  = np.linspace(0, 2 * np.pi, 300)
    t_samp = np.linspace(0, tof, n_points)
    tx, ty = kepler_position(a_t, e_t, MU, t_samp)

    leo_x, leo_y = r1 * np.cos(theta), r1 * np.sin(theta)
    geo_x, geo_y = r2 * np.cos(theta), r2 * np.sin(theta)

    # Full path: 1 LEO revolution → transfer arc → 1 GEO revolution
    geo_start = np.arctan2(ty[-1], tx[-1])
    theta_geo = np.linspace(0, 2 * np.pi, 300) + geo_start
    geo_shifted = np.column_stack((r2 * np.cos(theta_geo), r2 * np.sin(theta_geo)))
    leo_pts  = np.column_stack((leo_x, leo_y))
    trans_pts = np.column_stack((tx, ty))
    all_pts  = np.vstack([leo_pts, trans_pts, geo_shifted])

    # ── Static plot ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    earth = plt.Circle((0, 0), R_EARTH, color='royalblue', zorder=1, label='Earth')
    ax.add_patch(earth)
    ax.plot(leo_x, leo_y, 'g--', lw=1.2, label=f'LEO ({params["r1"]-R_EARTH:.0f} km)')
    ax.plot(geo_x, geo_y, 'r--', lw=1.2, label=f'GEO ({params["r2"]-R_EARTH:.0f} km)')
    ax.plot(tx, ty, 'b-', lw=2, label='Transfer arc')
    ax.plot(tx[0],  ty[0],  'g^', ms=10, zorder=5, label=f'Burn 1 ΔV={params["dv1"]:.3f} km/s')
    ax.plot(tx[-1], ty[-1], 'rs', ms=10, zorder=5, label=f'Burn 2 ΔV={params["dv2"]:.3f} km/s')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (km)', fontsize=11)
    ax.set_ylabel('Y (km)', fontsize=11)
    ax.set_title('Step 1 — Hohmann Transfer Orbit', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig('step1_hohmann_static.png', dpi=150)
    plt.show()
    print("Saved: step1_hohmann_static.png")

    # ── Animation ──────────────────────────────────────────
    if save_gif:
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        ax2.set_xlim(-1.2 * r2, 1.2 * r2)
        ax2.set_ylim(-1.2 * r2, 1.2 * r2)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.add_patch(plt.Circle((0, 0), R_EARTH, color='royalblue', zorder=1))
        ax2.plot(leo_x, leo_y, 'g--', lw=1, alpha=0.5)
        ax2.plot(geo_x, geo_y, 'r--', lw=1, alpha=0.5)
        ax2.plot(tx, ty, 'b-', lw=1, alpha=0.3)
        sat_dot,  = ax2.plot([], [], 'yo', ms=9, zorder=5, label='Satellite')
        trail,    = ax2.plot([], [], 'y-', lw=1, alpha=0.5)
        ax2.legend(fontsize=8)
        ax2.set_title('Step 1 — Hohmann Transfer Animation', fontsize=12, fontweight='bold')

        def init():
            sat_dot.set_data([], [])
            trail.set_data([], [])
            return sat_dot, trail

        def animate(i):
            sat_dot.set_data([all_pts[i, 0]], [all_pts[i, 1]])
            s = max(0, i - 40)
            trail.set_data(all_pts[s:i+1, 0], all_pts[s:i+1, 1])
            return sat_dot, trail

        ani = FuncAnimation(fig2, animate, init_func=init,
                            frames=len(all_pts), interval=30, blit=True)
        ani.save('step1_hohmann_animation.gif', writer=PillowWriter(fps=30))
        plt.close(fig2)
        print("Saved: step1_hohmann_animation.gif")


def run_step1(alt_initial_km=400.0, alt_final_km=35786.0):
    """Run Step 1 and return parameters for downstream steps."""
    print("\n" + "="*55)
    print("STEP 1 — HOHMANN TRANSFER ORBIT")
    print("="*55)

    params = compute_hohmann(alt_initial_km, alt_final_km)

    print(f"  Initial orbit altitude  : {alt_initial_km:.1f} km")
    print(f"  Final orbit altitude    : {alt_final_km:.1f} km")
    print(f"  Semi-major axis (a_t)   : {params['a_trans']:.2f} km")
    print(f"  Eccentricity (e)        : {params['e_trans']:.6f}")
    print(f"  Initial circ. velocity  : {params['v1']:.4f} km/s")
    print(f"  Final circ. velocity    : {params['v2']:.4f} km/s")
    print(f"  Velocity at perigee     : {params['v_trans_peri']:.4f} km/s")
    print(f"  Velocity at apogee      : {params['v_trans_apo']:.4f} km/s")
    print(f"  ΔV₁ (first burn)        : {params['dv1']:.4f} km/s")
    print(f"  ΔV₂ (second burn)       : {params['dv2']:.4f} km/s")
    print(f"  Total ΔV                : {params['dv_total']:.4f} km/s")
    print(f"  Transfer time (TOF)     : {params['tof']/60:.2f} min")

    plot_hohmann(params, save_gif=True)
    return params


# ╔══════════════════════════════════════════════════════════╗
# ║   STEP 2 — Lambert's Method for Orbital Rendezvous      ║
# ╚══════════════════════════════════════════════════════════╝

def lambert_minimum_energy(r0_vec, r_vec, mu=MU):
    """
    Minimum-energy Lambert transfer time.
    Returns (t_min, t_parabolic) or raises ValueError for degenerate cases.
    """
    r0_mag = np.linalg.norm(r0_vec)
    r_mag  = np.linalg.norm(r_vec)

    cos_dnu = np.dot(r0_vec, r_vec) / (r0_mag * r_mag)
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)

    c = np.sqrt(r0_mag**2 + r_mag**2 - 2 * r0_mag * r_mag * cos_dnu)
    s = (r0_mag + r_mag + c) / 2.0

    if c < 1e-6:
        raise ValueError("Degenerate Lambert: zero chord length.")
    if s < c:
        raise ValueError("Degenerate geometry: s < c.")

    a_min   = s / 2.0
    beta    = 2.0 * np.arcsin(np.sqrt((s - c) / s))
    t_min   = np.sqrt(a_min**3 / mu) * (np.pi - (beta - np.sin(beta)))
    t_parab = (np.sqrt(2.0 / mu) / 3.0) * (s**1.5 - (s - c)**1.5)
    return t_min, t_parab


def stumpff_C(z):
    if z > 0:  return (1.0 - np.cos(np.sqrt(z))) / z
    if z < 0:  return (np.cosh(np.sqrt(-z)) - 1.0) / (-z)
    return 0.5

def stumpff_S(z):
    if z > 0:  return (np.sqrt(z) - np.sin(np.sqrt(z))) / z**1.5
    if z < 0:  return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (-z)**1.5
    return 1.0 / 6.0


def _dt_of_z(z, r0_vec, r_vec, mu, tm):
    """
    Evaluate the Lambert time-of-flight as a function of universal variable z.
    Returns np.nan when the geometry is invalid (y < 0 or C ≈ 0).
    """
    r0 = np.linalg.norm(r0_vec)
    r  = np.linalg.norm(r_vec)
    cos_dnu = np.clip(np.dot(r0_vec, r_vec) / (r0 * r), -1.0, 1.0)
    A  = tm * np.sqrt(r0 * r * (1.0 + cos_dnu))
    C  = stumpff_C(z)
    S  = stumpff_S(z)
    if abs(C) < 1e-12:
        return np.nan
    y = r0 + r + A * (z * S - 1.0) / np.sqrt(C)
    if y < 0.0:
        return np.nan
    sq = np.sqrt(y)
    x  = sq / np.sqrt(C)
    return (x**3 * S + A * sq) / np.sqrt(mu)


def lambert_universal_variable(r0_vec, r_vec, dt, tm=1, mu=MU,
                                n_scan=2000, tol=1e-10):
    """
    Lambert's problem via universal variables with robust auto-bracketing.

    The classic implementation assumes a fixed z bracket of [-4π², 4π²],
    which fails when the valid domain of z is strictly positive (common for
    near-circular same-radius transfers like GEO proximity operations).

    This implementation:
      1. Scans z over [-4π², 4π²] to locate the valid domain and bracket
         the root dt(z) = target_dt.
      2. Bisects within that bracket to machine precision.
      3. Raises a clear ValueError if no bracket is found rather than
         silently diverging.

    tm = +1 → prograde short-way,  tm = -1 → retrograde.
    Returns (v0_vec, v_vec) in km/s.
    """
    r0 = np.linalg.norm(r0_vec)
    r  = np.linalg.norm(r_vec)
    cos_dnu = np.clip(np.dot(r0_vec, r_vec) / (r0 * r), -1.0, 1.0)
    A = tm * np.sqrt(r0 * r * (1.0 + cos_dnu))

    if abs(A) < 1e-6:
        raise ValueError("Lambert: A ≈ 0 — transfer angle is 0° or 180°.")

    # ── Step 1: scan to find a valid bracket [z_lo, z_hi] ─────────
    # dt(z) is continuous and monotone within the valid domain.
    # We need z_lo where dt < target and z_hi where dt > target (or vice-versa).
    z_scan  = np.linspace(-4.0 * np.pi**2, 4.0 * np.pi**2, n_scan)
    dt_scan = np.array([_dt_of_z(z, r0_vec, r_vec, mu, tm) for z in z_scan])

    # Keep only valid (non-nan) points
    valid_mask = ~np.isnan(dt_scan)
    if not np.any(valid_mask):
        raise RuntimeError("Lambert: no valid z found — check input geometry.")

    z_valid  = z_scan[valid_mask]
    dt_valid = dt_scan[valid_mask]

    # Find sign change: where dt crosses the target value
    diff = dt_valid - dt
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        # Target TOF may be outside the achievable range
        dt_min, dt_max = dt_valid.min(), dt_valid.max()
        raise RuntimeError(
            f"Lambert: target TOF={dt:.1f}s is outside achievable range "
            f"[{dt_min:.1f}, {dt_max:.1f}]s for this geometry.\n"
            f"Consider using a longer TOF or different phase angle."
        )

    # Use the first sign-change bracket (prograde solution)
    idx     = sign_changes[0]
    z_lo    = z_valid[idx]
    z_hi    = z_valid[idx + 1]

    # ── Step 2: bisect within the bracket ─────────────────────────
    for _ in range(300):
        z_mid  = (z_lo + z_hi) / 2.0
        dt_mid = _dt_of_z(z_mid, r0_vec, r_vec, mu, tm)

        if np.isnan(dt_mid):
            # Invalid midpoint — shrink toward the valid side
            z_hi = z_mid
            continue

        if (dt_mid - dt) * (dt_valid[idx] - dt) < 0.0:
            z_hi = z_mid
        else:
            z_lo = z_mid

        if abs(z_hi - z_lo) < tol:
            break

    z_sol = (z_lo + z_hi) / 2.0

    # ── Step 3: recover f, g and compute velocities ───────────────
    C     = stumpff_C(z_sol)
    S     = stumpff_S(z_sol)
    y     = r0 + r + A * (z_sol * S - 1.0) / np.sqrt(C)
    f     = 1.0 - y / r0
    g     = A * np.sqrt(y / mu)
    g_dot = 1.0 - y / r

    v0_vec = (r_vec  - f     * r0_vec) / g
    v_vec  = (g_dot  * r_vec - r0_vec) / g
    return v0_vec, v_vec


def propagate_two_body(r0, v0, dt, mu=MU, steps=500):
    """RK4 Keplerian propagator. Returns (steps+1, 3) position array."""
    def accel(rv):
        rmag = np.linalg.norm(rv[:3])
        return np.concatenate([rv[3:], -mu / rmag**3 * rv[:3]])

    rv   = np.concatenate([r0, v0])
    traj = [rv[:3].copy()]
    h    = dt / steps
    for _ in range(steps):
        k1 = accel(rv)
        k2 = accel(rv + 0.5 * h * k1)
        k3 = accel(rv + 0.5 * h * k2)
        k4 = accel(rv +       h * k3)
        rv += (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        traj.append(rv[:3].copy())
    return np.array(traj)


def run_step2(hohmann_params=None):
    """Run Step 2 and return Lambert rendezvous parameters."""
    print("\n" + "="*55)
    print("STEP 2 — LAMBERT'S METHOD FOR ORBITAL RENDEZVOUS")
    print("="*55)

    # Use GEO positions seeded from Step 1 if available
    if hohmann_params is not None:
        r2 = hohmann_params['r2']
        r_chaser       = np.array([r2, 0.0, 0.0])
        phi            = np.radians(5.0)          # 5° phase offset
        r_target_start = np.array([r2 * np.cos(phi), r2 * np.sin(phi), 0.0])
        r_target_end   = np.array([r2 * np.cos(phi + np.radians(2)),
                                   r2 * np.sin(phi + np.radians(2)), 0.0])
    else:
        r_chaser       = np.array([35000.0,  0.0,    0.0])
        r_target_start = np.array([35500.0,  400.0,  0.0])
        r_target_end   = np.array([35800.0,  800.0,  0.0])

    # Minimum-energy TOF
    try:
        t_min, _ = lambert_minimum_energy(r_chaser, r_target_end)
    except ValueError as e:
        print(f"  Min-energy warning: {e}. Using 1800 s default TOF.")
        t_min = 1800.0

    print(f"  Chaser initial position : {r_chaser} km")
    print(f"  Target final position   : {r_target_end} km")
    print(f"  Min-energy TOF          : {t_min:.1f} s ({t_min/60:.2f} min)")

    v0, vf = lambert_universal_variable(r_chaser, r_target_end, t_min)
    print(f"  Chaser burn velocity    : {v0}  |v| = {np.linalg.norm(v0):.4f} km/s")
    print(f"  Arrival velocity        : {vf}  |v| = {np.linalg.norm(vf):.4f} km/s")

    # Keplerian trajectories for visualisation
    chaser_traj = propagate_two_body(r_chaser, v0, t_min)
    v_tgt_circ  = np.sqrt(MU / np.linalg.norm(r_target_start))
    v_tgt       = np.array([0.0, v_tgt_circ, 0.0])
    target_traj = propagate_two_body(r_target_start, v_tgt, t_min)

    # Constellation graph
    G_graph = nx.DiGraph()
    G_graph.add_node("Chaser", pos=r_chaser[:2])
    G_graph.add_node("Target", pos=r_target_end[:2])
    G_graph.add_edge("Chaser", "Target", label=f"{t_min/60:.1f} min TOF")

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    pos_dict   = nx.get_node_attributes(G_graph, 'pos')
    nx.draw(G_graph, pos=pos_dict, with_labels=True, ax=ax1,
            node_size=700, node_color=['navy', 'firebrick'],
            font_color='white', font_weight='bold', font_size=10)
    nx.draw_networkx_edge_labels(G_graph, pos=pos_dict,
                                  edge_labels=nx.get_edge_attributes(G_graph, 'label'), ax=ax1)
    ax1.set_title('Step 2 — Constellation Graph: Chaser → Target', fontsize=13, fontweight='bold')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('step2_constellation.png', dpi=150)
    plt.show()
    print("Saved: step2_constellation.png")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(chaser_traj[:, 0], chaser_traj[:, 1], 'b-', lw=2, label='Chaser (Lambert arc)')
    ax2.plot(target_traj[:, 0], target_traj[:, 1], 'r-', lw=2, label='Target (Keplerian)')
    ax2.plot(*r_chaser[:2],     'b^', ms=10, label='Chaser start')
    ax2.plot(*r_target_start[:2], 'rs', ms=8,  label='Target start')
    ax2.plot(*r_target_end[:2], 'r*', ms=12, label='Meeting point')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X (km)', fontsize=11)
    ax2.set_ylabel('Y (km)', fontsize=11)
    ax2.set_title('Step 2 — Lambert Rendezvous Trajectories', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('step2_trajectories.png', dpi=150)
    plt.show()
    print("Saved: step2_trajectories.png")

    return dict(
        r_chaser       = r_chaser,
        r_target_start = r_target_start,
        r_target_end   = r_target_end,
        v_chaser_burn  = v0,
        v_arrival      = vf,
        t_transfer     = t_min,
    )


# ╔══════════════════════════════════════════════════════════╗
# ║   STEP 3 — Extended Kalman Filter for Rendezvous        ║
# ╚══════════════════════════════════════════════════════════╝

# EKF constants
EKF_MU       = MU
EKF_R_ORB    = R_EARTH + 35786.0            # GEO orbit radius (km)
OMEGA_O      = np.sqrt(EKF_MU / EKF_R_ORB**3)  # orbital angular rate (rad/s)
N_STATES     = 12                            # pos(3) vel(3) quat(4) bias(2 pad)
DT_EKF       = 1.0                          # timestep (s)

# Process and measurement noise
Q_EKF = np.diag(np.concatenate([[1e-6]*3, [1e-8]*3, [1e-7]*4, [1e-9]*2]))[:N_STATES, :N_STATES]
R_LASER = np.diag([1e-6, 1e-10, 1e-10])    # rho(km)^2, alpha(rad)^2, beta(rad)^2

# Initial state: 5 km radial separation, 3° yaw offset
SEP_INIT = 5.0
X0_EKF   = np.zeros(N_STATES)
X0_EKF[0] = SEP_INIT
X0_EKF[4] = -3.0 * OMEGA_O * SEP_INIT
theta0     = np.radians(3.0)
X0_EKF[6:10] = [0., 0., np.sin(theta0/2), np.cos(theta0/2)]
P0_EKF = np.diag([1.0]*3 + [0.01]*3 + [0.1]*4 + [0.001]*2)[:N_STATES, :N_STATES]


def cw_stm(dt, n=OMEGA_O):
    """Exact closed-form Clohessy-Wiltshire state transition matrix (6×6)."""
    sn, cn = np.sin(n*dt), np.cos(n*dt)
    return np.array([
        [4-3*cn,        0,  0,   sn/n,           2*(1-cn)/n,      0   ],
        [6*(sn-n*dt),   1,  0,  -2*(1-cn)/n,     (4*sn-3*n*dt)/n, 0   ],
        [0,             0,  cn,  0,               0,               sn/n],
        [3*n*sn,        0,  0,   cn,              2*sn,            0   ],
        [-6*n*(1-cn),   0,  0,  -2*sn,            4*cn-3,          0   ],
        [0,             0, -n*sn, 0,              0,               cn  ],
    ])


def ekf_predict(X, P, omega_body, dt=DT_EKF):
    """EKF prediction: CW translational dynamics + quaternion kinematics."""
    Phi    = np.eye(N_STATES)
    Phi[:6, :6] = cw_stm(dt)

    # Quaternion propagation via Omega matrix
    q        = X[6:10]
    wx, wy, wz = omega_body
    Omega = 0.5 * np.array([
        [ 0,   wz, -wy,  wx],
        [-wz,  0,   wx,  wy],
        [ wy, -wx,  0,   wz],
        [-wx, -wy, -wz,  0 ]
    ])
    q_pred = q + dt * (Omega @ q)
    q_pred /= np.linalg.norm(q_pred)

    X_pred = Phi @ X
    X_pred[6:10] = q_pred

    P_pred = Phi @ P @ Phi.T + Q_EKF
    P_pred = 0.5 * (P_pred + P_pred.T)
    return X_pred, P_pred


def h_laser(X):
    """Laser measurement model: [rho (km), alpha (rad), beta (rad)]."""
    x, y, z = X[0], X[1], X[2]
    rho = np.sqrt(x**2 + y**2 + z**2)
    if rho < 1e-6:
        return np.zeros(3)
    alpha = np.arcsin(np.clip(-z / rho, -1, 1))
    beta  = np.arctan2(y, x)
    return np.array([rho, alpha, beta])


def H_jacobian(X):
    """Analytic Jacobian of laser measurement model."""
    x, y, z  = X[0], X[1], X[2]
    rho      = np.sqrt(x**2 + y**2 + z**2)
    rho_xy   = np.sqrt(x**2 + y**2)
    H        = np.zeros((3, N_STATES))
    if rho < 1e-6:
        return H
    H[0, 0] = x / rho;  H[0, 1] = y / rho;  H[0, 2] = z / rho
    if rho_xy > 1e-6:
        denom    = rho**2 * rho_xy
        H[1, 0] =  x * z / denom
        H[1, 1] =  y * z / denom
        H[1, 2] = -rho_xy / rho**2
    else:
        H[1, 2] = -1.0 / rho
    if rho_xy**2 > 1e-6:
        H[2, 0] = -y / rho_xy**2
        H[2, 1] =  x / rho_xy**2
    return H


def ekf_update(X_pred, P_pred, z_meas):
    """Joseph-form EKF update for numerical stability."""
    H     = H_jacobian(X_pred)
    z_hat = h_laser(X_pred)
    innov = z_meas - z_hat
    # Angle wrapping
    innov[1] = (innov[1] + np.pi) % (2*np.pi) - np.pi
    innov[2] = (innov[2] + np.pi) % (2*np.pi) - np.pi

    S    = H @ P_pred @ H.T + R_LASER
    K    = P_pred @ H.T @ np.linalg.solve(S.T, np.eye(3)).T

    X_upd = X_pred + K @ innov
    I_KH  = np.eye(N_STATES) - K @ H
    P_upd = I_KH @ P_pred @ I_KH.T + K @ R_LASER @ K.T
    P_upd = 0.5 * (P_upd + P_upd.T)

    qn = np.linalg.norm(X_upd[6:10])
    if qn > 1e-6:
        X_upd[6:10] /= qn
    return X_upd, P_upd, innov


def simulate_laser(true_rel_pos):
    """Simulate a noisy laser range/bearing measurement."""
    x, y, z = true_rel_pos
    rho_t   = np.sqrt(x**2 + y**2 + z**2)
    if rho_t < 1e-6:
        return np.zeros(3)
    alpha_t = np.arcsin(np.clip(-z / rho_t, -1, 1))
    beta_t  = np.arctan2(y, x)
    noise   = np.array([np.random.normal(0, 0.001),
                        np.random.normal(0, 1e-4),
                        np.random.normal(0, 1e-4)])
    return np.array([rho_t, alpha_t, beta_t]) + noise


def quat_to_euler(q):
    """Convert quaternion [x,y,z,w] to Euler angles [roll, pitch, yaw]."""
    x, y, z, w = q
    roll  = np.arctan2(2*(w*x + y*z),   1 - 2*(x**2 + y**2))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y),   1 - 2*(y**2 + z**2))
    return np.array([roll, pitch, yaw])


def run_step3(n_steps=600):
    """Run Step 3 EKF simulation and return logged arrays."""
    print("\n" + "="*55)
    print("STEP 3 — EXTENDED KALMAN FILTER (EKF)")
    print("="*55)

    np.random.seed(42)

    X_hat    = X0_EKF.copy()
    P_hat    = P0_EKF.copy()
    k_decay  = 0.005              # true position decay rate (s⁻¹)
    omega_body = np.array([0.0, 0.0, OMEGA_O])

    times       = np.arange(n_steps) * DT_EKF
    est_pos     = np.zeros((n_steps, 3))
    true_pos    = np.zeros((n_steps, 3))
    innov_hist  = np.zeros((n_steps, 3))
    dist_hist   = np.zeros(n_steps)
    trace_P     = np.zeros(n_steps)
    euler_hist  = np.zeros((n_steps, 3))

    print(f"  Initial separation  : {SEP_INIT:.1f} km")
    print(f"  EKF steps           : {n_steps}  (dt={DT_EKF} s)")

    for k in range(n_steps):
        t        = k * DT_EKF
        r_true   = X0_EKF[:3] * np.exp(-k_decay * t)   # ground truth (exponential approach)
        z        = simulate_laser(r_true)

        X_hat, P_hat          = ekf_predict(X_hat, P_hat, omega_body, DT_EKF)
        X_hat, P_hat, innov   = ekf_update(X_hat, P_hat, z)

        est_pos[k]   = X_hat[:3]
        true_pos[k]  = r_true
        innov_hist[k]= innov
        dist_hist[k] = float(np.linalg.norm(X_hat[:3]))
        trace_P[k]   = float(np.trace(P_hat[:6, :6]))
        euler_hist[k]= quat_to_euler(X_hat[6:10])

    print(f"  Final EKF range     : {dist_hist[-1]:.4f} km")
    print(f"  Final cov trace     : {trace_P[-1]:.2e}")

    # ── Plots ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Step 3 — EKF Relative Navigation', fontsize=14, fontweight='bold')
    colors = ['r', 'g', 'b']
    labels = ['X (radial)', 'Y (along-track)', 'Z (cross-track)']

    ax = axes[0, 0]
    for i in range(3):
        ax.plot(times, true_pos[:, i], '--', color=colors[i], lw=1, alpha=0.6,
                label=f'True {labels[i]}')
        ax.plot(times, est_pos[:, i],  '-',  color=colors[i], lw=1.5,
                label=f'EKF  {labels[i]}')
    ax.set(xlabel='Time (s)', ylabel='Position (km)', title='Relative position: true vs EKF')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times, dist_hist, 'm-', lw=1.5)
    ax.set(xlabel='Time (s)', ylabel='Range (km)', title='Chaser-target range')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for i, lbl in enumerate(['ρ (km)', 'α (rad)', 'β (rad)']):
        ax.plot(times, innov_hist[:, i], lw=1, label=lbl)
    ax.set(xlabel='Time (s)', ylabel='Innovation', title='Measurement innovations')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(times, trace_P, 'k-', lw=1.5)
    ax.set(xlabel='Time (s)', ylabel='tr(P) log scale', title='Covariance trace (convergence)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('step3_ekf.png', dpi=150)
    plt.show()
    print("Saved: step3_ekf.png")

    # RSW position plot
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    fig2.suptitle('Step 3 — RSW Frame Relative Position', fontsize=13, fontweight='bold')
    rsw_labels = ['Radial (R)', 'Along-track (S)', 'Cross-track (W)']
    rsw_colors = ['r', 'g', 'b']
    for i in range(3):
        axes2[i].plot(times, est_pos[:, i], color=rsw_colors[i], lw=1.5)
        axes2[i].set(xlabel='Time (s)', ylabel='km', title=rsw_labels[i])
        axes2[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('step3_rsw_position.png', dpi=150)
    plt.show()
    print("Saved: step3_rsw_position.png")

    # Euler angles plot
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    for i, (lbl, col) in enumerate(zip(['Roll (φ)', 'Pitch (θ)', 'Yaw (ψ)'], ['r', 'g', 'b'])):
        ax3.plot(times, np.degrees(euler_hist[:, i]), color=col, lw=1.5, label=lbl)
    ax3.set(xlabel='Time (s)', ylabel='Angle (degrees)', title='Step 3 — Chaser Euler Angles')
    ax3.legend(); ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('step3_euler_angles.png', dpi=150)
    plt.show()
    print("Saved: step3_euler_angles.png")

    return dict(
        times=times, est_pos=est_pos, true_pos=true_pos,
        dist_hist=dist_hist, trace_P=trace_P,
        final_range_km=dist_hist[-1],
        final_quat=X_hat[6:10].copy(),
    )


# ╔══════════════════════════════════════════════════════════╗
# ║   STEP 4 — Attitude Control with PID                    ║
# ╚══════════════════════════════════════════════════════════╝

# ── PID parameters ─────────────────────────────────────────
J_INERTIA = np.diag([0.01, 0.015, 0.02])    # inertia tensor (kg·m²)
J_INV     = np.linalg.inv(J_INERTIA)

KP_ATT   = 0.08   # proportional gain  (tuned for inertia J_max=0.02 kg·m²)
KD_ATT   = 0.8    # derivative gain — applied directly to angular velocity
TAU_MAX  = 0.5    # torque saturation (Nm)
OMEGA_MAX = 5.0   # angular velocity clamp (rad/s) — prevents numerical overflow
DT_PID   = 0.01   # timestep (s)
T_PID    = 800.0  # simulation duration (s) — sufficient for 90° slew
CONV_TOL = 1e-3   # convergence tolerance (rad)

# Initial and target quaternions
Q_INIT_PID   = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0])  # 90° about x
Q_TARGET_PID = np.array([np.sqrt(3)/2, 0.0, 0.5, 0.0])           # 60° about y


# ── Quaternion utilities ───────────────────────────────────

def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def qmul(q1, q2):
    w1,x1,y1,z1 = q1;  w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_error_pid(q_curr, q_target):
    """Error quaternion enforcing shortest-path rotation."""
    dq = qmul(q_target, qconj(q_curr))
    return -dq if dq[0] < 0 else dq

def qdot_pid(q, omega):
    return 0.5 * qmul(q, np.array([0.0, *omega]))

def rk4_quat_pid(q, omega, dt):
    k1 = qdot_pid(q,              omega)
    k2 = qdot_pid(q + 0.5*dt*k1, omega)
    k3 = qdot_pid(q + 0.5*dt*k2, omega)
    k4 = qdot_pid(q +     dt*k3, omega)
    q_new = q + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    n = np.linalg.norm(q_new)
    return q_new / n if n > 1e-8 else np.array([1., 0., 0., 0.])

def quat_to_euler_pid(q):
    w, x, y, z = q
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return np.array([roll, pitch, yaw])


def run_step4():
    """
    Run Step 4 attitude PD control simulation and return results.

    Control law: τ = Kp·e_v − Kd·ω   (PD with direct omega feedback)

    Using angular velocity directly as the derivative term avoids the
    numerical instability of finite-differencing omega over a small DT,
    which amplifies noise by 1/DT and causes overflow for large-angle slews.
    This is the standard formulation used in spacecraft attitude control.
    """
    print("\n" + "="*55)
    print("STEP 4 — ATTITUDE CONTROL WITH PID")
    print("="*55)

    q        = Q_INIT_PID.copy()
    omega    = np.zeros(3)              # start from rest
    t_log, euler_log, omega_log, torque_log, err_log = [], [], [], [], []

    t = 0.0
    converged_at = None

    while t <= T_PID:
        # ── Quaternion error (shortest path) ──────────────
        dq    = quat_error_pid(q, Q_TARGET_PID)
        e_vec = 2.0 * dq[1:4]          # vector part of error quaternion

        # ── PD control law: P on attitude error, D on omega ─
        torque = np.clip(KP_ATT * e_vec - KD_ATT * omega, -TAU_MAX, TAU_MAX)

        # ── Rigid-body dynamics ────────────────────────────
        omega_dot = J_INV @ (torque - np.cross(omega, J_INERTIA @ omega))
        omega     = omega + omega_dot * DT_PID
        omega     = np.clip(omega, -OMEGA_MAX, OMEGA_MAX)  # overflow guard
        if not np.all(np.isfinite(omega)):
            omega = np.zeros(3)

        # ── Quaternion kinematics (RK4) ────────────────────
        q = rk4_quat_pid(q, omega, DT_PID)

        t_log.append(t)
        euler_log.append(quat_to_euler_pid(q))
        omega_log.append(omega.copy())
        torque_log.append(torque.copy())
        err_log.append(np.linalg.norm(e_vec))

        t += DT_PID

        if np.linalg.norm(e_vec) < CONV_TOL and np.linalg.norm(omega) < 1e-4:
            converged_at = t
            print(f"  Converged at t = {t:.2f} s  |e| = {np.linalg.norm(e_vec):.2e}")
            break

    if converged_at is None:
        print(f"  Did not converge within {T_PID} s. Final |e| = {err_log[-1]:.4f}")

    t_arr      = np.array(t_log)
    euler_arr  = np.array(euler_log)
    omega_arr  = np.array(omega_log)
    torque_arr = np.array(torque_log)
    err_arr    = np.array(err_log)

    e_init  = quat_to_euler_pid(Q_INIT_PID)
    e_final = quat_to_euler_pid(q)
    e_tgt   = quat_to_euler_pid(Q_TARGET_PID)
    print(f"  Initial Euler (deg) : Roll={np.degrees(e_init[0]):.2f}  "
          f"Pitch={np.degrees(e_init[1]):.2f}  Yaw={np.degrees(e_init[2]):.2f}")
    print(f"  Final   Euler (deg) : Roll={np.degrees(e_final[0]):.2f}  "
          f"Pitch={np.degrees(e_final[1]):.2f}  Yaw={np.degrees(e_final[2]):.2f}")
    print(f"  Target  Euler (deg) : Roll={np.degrees(e_tgt[0]):.2f}  "
          f"Pitch={np.degrees(e_tgt[1]):.2f}  Yaw={np.degrees(e_tgt[2]):.2f}")
    print(f"  Final quaternion    : {q}")
    print(f"  Final error norm    : {err_arr[-1]:.4e} rad")

    # ── Plots ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Step 4 — Attitude PD Control', fontsize=14, fontweight='bold')

    axes[0,0].plot(t_arr, np.degrees(euler_arr))
    axes[0,0].set(title='Euler angles', xlabel='t (s)', ylabel='deg')
    axes[0,0].legend(['Roll', 'Pitch', 'Yaw']); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(t_arr, omega_arr)
    axes[0,1].set(title='Angular velocity', xlabel='t (s)', ylabel='rad/s')
    axes[0,1].legend(['ωx', 'ωy', 'ωz']); axes[0,1].grid(True, alpha=0.3)

    axes[0,2].plot(t_arr, torque_arr)
    axes[0,2].set(title='Control torque', xlabel='t (s)', ylabel='Nm')
    axes[0,2].legend(['Tx', 'Ty', 'Tz']); axes[0,2].grid(True, alpha=0.3)

    axes[1,0].semilogy(t_arr, np.maximum(err_arr, 1e-10))
    axes[1,0].set(title='Orientation error norm (log)', xlabel='t (s)', ylabel='|e| rad')
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(t_arr, np.linalg.norm(torque_arr, axis=1))
    axes[1,1].set(title='Torque norm', xlabel='t (s)', ylabel='Nm')
    axes[1,1].grid(True, alpha=0.3)

    axes[1,2].plot(err_arr, np.linalg.norm(torque_arr, axis=1), lw=0.8, alpha=0.7)
    axes[1,2].set(title='Phase portrait', xlabel='|e| (rad)', ylabel='|τ| (Nm)')
    axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('step4_attitude_pid.png', dpi=150)
    plt.show()
    print("Saved: step4_attitude_pid.png")

    return dict(
        t=t_arr, euler=euler_arr, omega=omega_arr,
        torque=torque_arr, err=err_arr,
        q_final=q, converged_at=converged_at,
    )


# ╔══════════════════════════════════════════════════════════╗
# ║   MAIN — Run all steps in sequence                      ║
# ╚══════════════════════════════════════════════════════════╝

def run_all(alt_initial_km=400.0, alt_final_km=35786.0, ekf_steps=600):
    """
    Execute all four GNC steps in sequence.

    Parameters
    ----------
    alt_initial_km : initial parking orbit altitude (km above surface)
    alt_final_km   : target orbit altitude (km above surface)
    ekf_steps      : number of EKF simulation timesteps

    Returns
    -------
    dict with results from each step
    """
    print("\n" + "★"*55)
    print("  ORBITAL RENDEZVOUS AND GNC ALGORITHM")
    print("  Development of GNC for Satellite Refueling")
    print("★"*55)
    print(f"  Initial orbit : {alt_initial_km} km altitude")
    print(f"  Final orbit   : {alt_final_km} km altitude")

    # Step 1 — Hohmann Transfer
    h_params = run_step1(alt_initial_km, alt_final_km)

    # Step 2 — Lambert Rendezvous (seeded from Step 1)
    l_params = run_step2(hohmann_params=h_params)

    # Step 3 — EKF Navigation
    ekf_results = run_step3(n_steps=ekf_steps)

    # Step 4 — Attitude PID Control
    pid_results = run_step4()

    # ── Final summary ──────────────────────────────────────
    print("\n" + "="*55)
    print("MISSION SUMMARY")
    print("="*55)
    print(f"  Total ΔV (Hohmann)     : {h_params['dv_total']:.4f} km/s")
    print(f"  Transfer time          : {h_params['tof']/60:.1f} min")
    print(f"  Lambert burn |v|       : {np.linalg.norm(l_params['v_chaser_burn']):.4f} km/s")
    print(f"  Lambert TOF            : {l_params['t_transfer']/60:.1f} min")
    print(f"  Final EKF range        : {ekf_results['final_range_km']:.4f} km")
    print(f"  EKF cov trace (final)  : {ekf_results['trace_P'][-1]:.2e}")
    if pid_results['converged_at'] is not None:
        print(f"  PID converged at       : {pid_results['converged_at']:.1f} s")
    else:
        print(f"  PID final |e|          : {pid_results['err'][-1]:.4e} rad")
    print("\n  Output files saved:")
    print("    step1_hohmann_static.png")
    print("    step1_hohmann_animation.gif")
    print("    step2_constellation.png")
    print("    step2_trajectories.png")
    print("    step3_ekf.png")
    print("    step3_rsw_position.png")
    print("    step3_euler_angles.png")
    print("    step4_attitude_pid.png")
    print("\n" + "★"*55)
    print("  ALL STEPS COMPLETE")
    print("★"*55)

    return dict(hohmann=h_params, lambert=l_params,
                ekf=ekf_results, pid=pid_results)


# ── Entry point ────────────────────────────────────────────
# Change these values to run different scenarios:
#   LEO → GEO  : alt_initial_km=400,  alt_final_km=35786
#   LEO → ISS  : alt_initial_km=300,  alt_final_km=420
#   LEO → MEO  : alt_initial_km=400,  alt_final_km=20200

results = run_all(
    alt_initial_km = 400.0,     # LEO parking orbit (km)
    alt_final_km   = 35786.0,   # GEO target orbit  (km)
    ekf_steps      = 600,       # EKF simulation steps
)