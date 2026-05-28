import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =====================================================================
# 1. PARAMETERS & MATERIAL PROPERTIES
# =====================================================================
# Sphere Geometry & Coating
R_CORE = 0.1          # Radius of the core (meters)
THICKNESS = 0.002     # Coating thickness (meters)
R_OUTER = R_CORE + THICKNESS

DENSITY = 2200        # Coating density (kg/m^3)
CP = 700              # Specific heat capacity (J/kg*K)
EMISSIVITY = 0.85     # Surface emissivity (epsilon)
ABSORPTIVITY = 0.85   # Surface solar absorptivity (alpha) - assumed equal if unspecified

# Constants
SIGMA = 5.670374e-8   # Stefan-Boltzmann constant (W/m^2*K^4)
S_SOLAR = 1361.0      # Solar constant at Earth (W/m^2)
R_EARTH = 6378137.0   # Earth radius (meters)

# Calculate Coating Mass and Areas
V_COATING = (4/3) * np.pi * (R_OUTER**3 - R_CORE**3)
MASS = DENSITY * V_COATING
A_CROSS = np.pi * R_OUTER**2
A_SURFACE = 4 * np.pi * R_OUTER**2

print(f"Coating Mass: {MASS:.4f} kg")
print(f"Thermal Capacitance (m*Cp): {MASS * CP:.2f} J/K")

# =====================================================================
# 2. ORBIT & ECLIPSE GEOMETRY HELPERS
# =====================================================================
def get_eclipse_factor(r_sat, r_sun):
    """
    Simple cylindrical shadow model.
    Returns 1.0 for full sun, 0.0 for total eclipse.
    """
    # Normalize vectors
    u_sun = r_sun / np.linalg.norm(r_sun)
    
    # Projection of satellite position onto the Earth-Sun line
    dot_product = np.dot(r_sat, u_sun)
    
    # If the satellite is on the sunward side of Earth, it's definitely sunlit
    if dot_product > 0:
        return 1.0
    
    # Perpendicular distance from satellite to the Earth-Sun line
    r_perp = np.sqrt(np.linalg.norm(r_sat)**2 - dot_product**2)
    
    # If it's behind Earth and within Earth's radius, it's in shadow
    if r_perp < R_EARTH:
        return 0.0
    
    return 1.0

#Mock Keplerian orbit data interpolator for demonstration. 
#Replace these with your actual astronomical model outputs.
def get_vectors_at_time(t):
    # Let's simulate a low Earth orbit with an orbital period of ~5400 seconds (90 mins)
    omega = 2 * np.pi / 5400
    alt = 400000 + R_EARTH
    
    # Sat orbiting in X-Y plane
    r_sat = np.array([alt * np.cos(omega * t), alt * np.sin(omega * t), 0.0])
    # Sun stationary along the X-axis for simplicity
    r_sun = np.array([1.496e11, 0.0, 0.0]) 
    
    return r_sat, r_sun

# =====================================================================
# 3. THERMAL ODE
# =====================================================================
def thermal_ode(t, T):
    # Get current ECR positions
    r_sat, r_sun = get_vectors_at_time(t)
    
    # Evaluate eclipse
    f_eclipse = get_eclipse_factor(r_sat, r_sun)
    
    # Heat Rates
    Q_in = S_SOLAR * A_CROSS * ABSORPTIVITY * f_eclipse
    Q_out = EMISSIVITY * SIGMA * A_SURFACE * (T[0]**4)
    
    # dT/dt = Q_net / (m * Cp)
    dT_dt = (Q_in - Q_out) / (MASS * CP)
    return [dT_dt]

# =====================================================================
# 4. SIMULATION EXECUTION (Long term over multiple orbits)
# =====================================================================
# Simulate for 5 orbits to ensure steady state equilibrium
orbital_period = 5400 
t_span = (0, orbital_period * 5)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
T_init = [273.15] # Start at 0 degrees Celsius

solution = solve_ivp(thermal_ode, t_span, T_init, t_eval=t_eval, method='RK45')

# =====================================================================
# 5. POST-PROCESSING & PLOTTING
# =====================================================================
time_hours = solution.t / 3600
temp_celsius = solution.y[0] - 273.15

# Extract long term bounds from the final orbit
final_orbit_indices = solution.t > (t_span[1] - orbital_period)
max_temp = np.max(temp_celsius[final_orbit_indices])
min_temp = np.min(temp_celsius[final_orbit_indices])

print(f"\n--- Long Term Temperature Bounds ---")
print(f"Max Sunlit Surface Temp: {max_temp:.2f} °C")
print(f"Min Shadowed Surface Temp: {min_temp:.2f} °C")

plt.figure(figsize=(10, 5))
plt.plot(time_hours, temp_celsius, label="Surface Temperature")
plt.axhline(max_temp, color='r', linestyle='--', label=f'Max: {max_temp:.1f}°C')
plt.axhline(min_temp, color='b', linestyle='--', label=f'Min: {min_temp:.1f}°C')
plt.xlabel("Time (Hours)")
plt.ylabel("Temperature (°C)")
plt.title("Long-Term Fuzzy Sphere Coating Transient Temperature Profile")
plt.grid(True)
plt.legend()
plt.show()