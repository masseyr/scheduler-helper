import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# =====================================================================
# 1. GENERATE MESH (Fibonacci Sphere)
# =====================================================================
def generate_fibonacci_sphere(num_nodes):
    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(num_nodes):
        y = 1 - (i / float(num_nodes - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
        
    return np.array(points) # Array of shape (N, 3) - these are unit vectors n_i

N_NODES = 200 
NODE_VECTORS = generate_fibonacci_sphere(N_NODES)

# =====================================================================
# 2. MULTI-NODE THERMAL ODE
# =====================================================================
def multi_node_thermal_ode(t, T, sat_func, sun_func, params, sunlit=True):
    # T is an array of shapes (N_NODES,)
    
    r_sat = sat_func(t)
    r_sun = sun_func(t)
    
    # 1. Global Eclipse Check
    f_eclipse = int(sunlit) # 1 or 0
    
    # 2. Sun vector unit direction
    u_sun = r_sun / np.linalg.norm(r_sun)
    
    # 3. Calculate dot products for all nodes at once (Vectorized!)
    # NODE_VECTORS is (N, 3), u_sun is (3,) -> cos_theta is (N,)
    cos_theta = np.dot(NODE_VECTORS, u_sun)
    
    dT_dt = np.zeros(N_NODES)
    
    # Extract constants
    A_node = params['A_SURFACE'] / N_NODES
    m_node = params['MASS'] / N_NODES
    Cp = params['CP']
    sigma = params['SIGMA']
    epsilon = params['EMISSIVITY']
    alpha = params['ABSORPTIVITY']
    S_solar = params['S_SOLAR']
    
    # 4. Loop through nodes (or vectorize completely)
    for i in range(N_NODES):
        # Solar input based on local orientation
        if f_eclipse > 0 and cos_theta[i] > 0:
            Q_in = S_solar * A_node * alpha * cos_theta[i]
        else:
            Q_in = 0.0
            
        # Local space radiation
        Q_out = epsilon * sigma * A_node * (T[i]**4)
        
        # (Optional) Conduction heat transfer could be added here
        Q_cond = 0.0 
        
        dT_dt[i] = (Q_in - Q_out + Q_cond) / (m_node * Cp)
        
    return dT_dt


def calculate_apparent_temperature(T_nodes, node_vectors, u_obs, params):
    """
    T_nodes: array of shape (N_NODES,) representing the temperatures of all nodes at a specific time.
    node_vectors: array of shape (N_NODES, 3) pointing outwards from the sphere center.
    u_obs: unit vector (3,) pointing from the sphere to the observer.
    """
    # 1. Normalize the observer vector just in case
    u_obs = u_obs / np.linalg.norm(u_obs)
    
    # 2. Calculate alignment (cos_phi) of all nodes relative to the observer
    cos_phi = np.dot(node_vectors, u_obs)
    
    # 3. Create a mask for nodes that are on the visible hemisphere
    visible_mask = cos_phi > 0
    
    # Extract properties
    A_node = params['A_SURFACE'] / len(T_nodes)
    sigma = params['SIGMA']
    epsilon = params['EMISSIVITY']
    R_outer = params['R_OUTER']
    
    # 4. Calculate total power seen by the observer (Sum over visible nodes)
    # Vectorized calculation:
    P_nodes = epsilon * sigma * A_node * cos_phi[visible_mask] * (T_nodes[visible_mask]**4)
    P_total = np.sum(P_nodes)
    
    # 5. Total projected area of the sphere (a flat disk)
    A_disk = np.pi * (R_outer**2)
    
    # 6. Back-calculate the apparent temperature (Kelvin)
    T_apparent = (P_total / (epsilon * sigma * A_disk))**(1/4)
    
    return T_apparent


# =====================================================================
# 3. EXECUTION
# =====================================================================
# T_init must now be an array of length N_NODES
T_init = np.full(N_NODES, 273.15) 

# Run solve_ivp exactly as before!
# solution.y will yield a shape of (N_NODES, time_steps)