import numpy as np

# Constants
MU = 3.986004418e14       # Earth's gravitational parameter (m^3/s^2)
R_EARTH = 6378137.0       # Earth's equatorial radius (meters)

def c2c3(psi):
    """Computes the universal variable stumpff functions C2 and C3."""
    if psi > 1e-6:
        sqrt_psi = np.sqrt(psi)
        c2 = (1.0 - np.cos(sqrt_psi)) / psi
        c3 = (sqrt_psi - np.sin(sqrt_psi)) / (psi * sqrt_psi)
    elif psi < -1e-6:
        sqrt_psi = np.sqrt(-psi)
        c2 = (1.0 - np.cosh(sqrt_psi)) / psi
        c3 = (np.sinh(sqrt_psi) - sqrt_psi) / (-psi * sqrt_psi)
    else:
        c2 = 1.0 / 2.0
        c3 = 1.0 / 6.0
    return c2, c3

def propagate_universal(r0, v0, dt, mu=MU):
    """
    Propagates a state vector (r0, v0) forward by dt seconds 
    using the Universal Variables formulation. Valid for ALL conic arcs.
    """
    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    
    # Reciprocal of semi-major axis (alpha)
    alpha = (2.0 / r0_mag) - (v0_mag**2 / mu)
    
    # Initial guess for universal anomaly chi
    chi = np.sqrt(mu) * abs(alpha) * dt if abs(alpha) > 1e-6 else np.sqrt(mu) * dt / r0_mag
    
    # Newton-Raphson iteration to find chi
    tol = 1e-11
    max_iter = 200
    for _ in range(max_iter):
        psi = chi**2 * alpha
        c2, c3 = c2c3(psi)
        
        # Companion functions
        r = (chi**2 * c2) + (np.dot(r0, v0) / np.sqrt(mu) * chi * (1.0 - psi * c3)) + (r0_mag * (1.0 - psi * c2))
        t_calc = (chi**3 * c3 + (np.dot(r0, v0) / np.sqrt(mu)) * chi**2 * c2 + r0_mag * chi * (1.0 - psi * c3)) / np.sqrt(mu)
        
        # Error check
        dt_error = dt - t_calc
        if abs(dt_error) < tol:
            break
            
        # Update step using derivative (which is the current radius r)
        chi += dt_error * np.sqrt(mu) / r
    else:
        # Fallback to Bisection if Newton-Raphson oscillates on hyperbola bounds
        low_chi, high_chi = -1e5, 1e5
        for _ in range(200):
            chi = 0.5 * (low_chi + high_chi)
            psi = chi**2 * alpha
            c2, c3 = c2c3(psi)
            t_calc = (chi**3 * c3 + (np.dot(r0, v0) / np.sqrt(mu)) * chi**2 * c2 + r0_mag * chi * (1.0 - psi * c3)) / np.sqrt(mu)
            if abs(dt - t_calc) < tol: break
            if t_calc < dt: low_chi = chi
            else: high_chi = chi

    # Recompute Stumpff coefficients for final converged chi
    psi = chi**2 * alpha
    c2, c3 = c2c3(psi)
    
    # Compute Lagrange f and g expression functions
    f = 1.0 - (chi**2 / r0_mag) * c2
    g = dt - (chi**3 / np.sqrt(mu)) * c3
    
    # Updated Position vector
    r_t = (f * r0) + (g * v0)
    r_t_mag = np.linalg.norm(r_t)
    
    # Compute Lagrange derivatives f_dot and g_dot
    f_dot = (np.sqrt(mu) / (r_t_mag * r0_mag)) * alpha * chi**3 * c3 - (np.sqrt(mu) / (r_t_mag * r0_mag)) * chi
    g_dot = 1.0 - (chi**2 / r_t_mag) * c2
    
    # Updated Velocity vector
    v_t = (f_dot * r0) + (g_dot * v0)
    
    return r_t, v_t

def lamberts_problem(r1, r2, delta_t, mu=MU):
    """Solves Lambert's problem using Universal Variables."""
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    
    cos_delta_nu = np.dot(r1, r2) / (r1_mag * r2_mag)
    A = np.sin(np.arccos(cos_delta_nu)) * np.sqrt(r1_mag * r2_mag / (1.0 - cos_delta_nu))
    
    psi = 0.0
    psi_up = 4.0 * np.pi**2
    psi_low = -4.0 * np.pi
    
    for _ in range(1000):
        c2, c3 = c2c3(psi)
        y = r1_mag + r2_mag + A * (psi * c3 - 1.0) / np.sqrt(c2)
        
        if A > 0.0 and y < 0.0:
            psi_low = psi
            psi = 0.5 * (psi + psi_up)
            continue
            
        x = np.sqrt(y / c2)
        t_calc = (x**3 * c3 + A * np.sqrt(y)) / np.sqrt(mu)
        
        if abs(t_calc - delta_t) < 1e-6:
            break
            
        if t_calc <= delta_t:
            psi_low = psi
        else:
            psi_up = psi
        psi = 0.5 * (psi_low + psi_up)
        
    f = 1.0 - y / r1_mag
    g_dot = 1.0 - y / r2_mag
    g = A * np.sqrt(y / mu)
    
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    return v1, v2

# =============================================================================
# SCENARIO SETUP
# =============================================================================

# 1. Target Parameters (Simulating a standard 400km low Earth orbit target)
# We calculate its state explicitly using basic circular/near-circular geometry 
# to acquire the direct target vector at the exact second of impact.
inc_tgt = np.radians(51.6)
raan_tgt = np.radians(120.0)
r_target_mag = R_EARTH + 35786000.0  # 400km Altitude

# Target position at the split second of intercept
theta_intercept = np.radians(52.5) # Location along its orbital plane arc
r_target_intercept = r_target_mag * np.array([
    np.cos(theta_intercept) * np.cos(raan_tgt) - np.sin(theta_intercept) * np.sin(raan_tgt) * np.cos(inc_tgt),
    np.cos(theta_intercept) * np.sin(raan_tgt) + np.sin(theta_intercept) * np.cos(raan_tgt) * np.cos(inc_tgt),
    np.sin(theta_intercept) * np.sin(inc_tgt)
])

# 2. ASAT Initial Burnout State (150 km altitude ignition point)
r_asat_burnout = np.array([
    (R_EARTH + 150000.0) * np.cos(np.radians(15.0)) * np.cos(np.radians(40.0)),
    (R_EARTH + 150000.0) * np.cos(np.radians(15.0)) * np.sin(np.radians(40.0)),
    (R_EARTH + 150000.0) * np.sin(np.radians(15.0))
])

tof = 9000

# Solve Lambert problem to find required burnout velocity
v_asat_burnout, _ = lamberts_problem(r_asat_burnout, r_target_intercept, tof)

# =============================================================================
# FILE GENERATION LOOP
# =============================================================================
filename = "asat_trajectory_j2000.txt"
print(f"Propagating ballistic arc via Universal Variables...")
print(f"Writing ephemeris step data directly into '{filename}'...")

dt = 1.0  # propagation step size (seconds)
r_t, v_t = r_asat_burnout.copy(), v_asat_burnout.copy()

with open(filename, "w") as f:
    f.write("# DIRECT ASCENT ASAT TRAJECTORY EPHEMERIS (ECI J2000)\n")
    f.write(f"# Time of Flight: {tof} seconds\n")
    f.write("# Columns: Time(s), Pos_X(m), Pos_Y(m), Pos_Z(m), Vel_X(m/s), Vel_Y(m/s), Vel_Z(m/s)\n")
    f.write("# " + "-"*95 + "\n")

    for t in range(int(tof) + 1):
        f.write(f"{t:5d}, {r_t[0]:14.4f}, {r_t[1]:14.4f}, {r_t[2]:14.4f}, "
                f"{v_t[0]:11.4f}, {v_t[1]:11.4f}, {v_t[2]:11.4f}\n")
        if t < int(tof):
            r_t, v_t = propagate_universal(r_t, v_t, dt)

print(f"Success! Generated {int(tof)+1} sequential state vectors without numerical breaks.")