"""
Core physics computations for the crane model.
These functions are optimized with Numba for performance.
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def compute_mass_matrix(q, m_t, m_h, m_l, l_2):
    """
    Compute the mass matrix of the crane system.

    Args:
        q: State vector [x, r, theta1, theta2]
        m_t: Trolley mass
        m_h: Hook mass
        m_l: Load mass
        l_2: Hook to load distance

    Returns:
        4x4 mass matrix
    """
    x, r, theta1, theta2 = q
    sin_t1, cos_t1 = np.sin(theta1), np.cos(theta1)
    sin_t2, cos_t2 = np.sin(theta2), np.cos(theta2)

    M = np.zeros((4, 4), dtype=q.dtype)

    M[0, 0] = m_t + m_h + m_l
    M[0, 1] = (m_h + m_l) * sin_t1
    M[0, 2] = (m_h + m_l) * r * cos_t1
    M[0, 3] = m_l * l_2 * cos_t2

    M[1, 0] = M[0, 1]
    M[1, 1] = m_h + m_l
    M[1, 3] = m_l * l_2 * np.sin(theta2 - theta1)

    M[2, 0] = M[0, 2]
    M[2, 2] = (m_h + m_l) * r ** 2
    M[2, 3] = m_l * r * l_2 * np.cos(theta2 - theta1)

    M[3, 0] = M[0, 3]
    M[3, 1] = M[1, 3]
    M[3, 2] = M[2, 3]
    M[3, 3] = m_l * l_2 ** 2

    return M


@jit(nopython=True)
def compute_coriolis_matrix(q, q_dot, m_h, m_l, l_2):
    """
    Compute the Coriolis matrix of the crane system.

    Args:
        q: State vector [x, r, theta1, theta2]
        q_dot: Velocity vector [x_dot, r_dot, theta1_dot, theta2_dot]
        m_h: Hook mass
        m_l: Load mass
        l_2: Hook to load distance

    Returns:
        4x4 Coriolis matrix
    """
    x, r, theta1, theta2 = q
    x_dot, r_dot, theta1_dot, theta2_dot = q_dot
    sin_t1, cos_t1 = np.sin(theta1), np.cos(theta1)
    sin_t2, cos_t2 = np.sin(theta2), np.cos(theta2)

    C = np.zeros((4, 4), dtype=q.dtype)

    C[0, 1] = (m_h + m_l) * cos_t1 * theta1_dot
    C[0, 2] = (m_h + m_l) * (r_dot * cos_t1 - r * sin_t1 * theta1_dot)
    C[0, 3] = -m_h * l_2 * sin_t2 * theta2_dot

    C[1, 2] = -(m_h + m_l) * r * theta1_dot
    C[1, 3] = -np.cos(theta1 - theta2) * m_h * l_2 * theta2_dot

    C[2, 1] = (m_h + m_l) * r * theta1_dot
    C[2, 2] = (m_h + m_l) * r * r_dot
    C[2, 3] = np.sin(theta1 - theta2) * m_h * l_2 * r * theta2_dot

    C[3, 1] = np.cos(theta1 - theta2) * m_h * l_2 * theta1_dot
    C[3, 2] = m_h * l_2 * np.cos(theta1 - theta2) * r_dot - np.sin(theta1 - theta2) * r * theta1_dot

    return C


@jit(nopython=True)
def compute_gravity_vector(q, g, m_h, m_l, l_2):
    """
    Compute the gravity vector of the crane system.

    Args:
        q: State vector [x, r, theta1, theta2]
        g: Gravity acceleration
        m_h: Hook mass
        m_l: Load mass
        l_2: Hook to load distance

    Returns:
        4-element gravity vector
    """
    x, r, theta1, theta2 = q
    sin_t1, cos_t1 = np.sin(theta1), np.cos(theta1)
    sin_t2, cos_t2 = np.sin(theta2), np.cos(theta2)

    G = np.zeros(4, dtype=q.dtype)

    G[1] = -(m_h + m_l) * g * cos_t1
    G[2] = (m_h + m_l) * g * r * sin_t1
    G[3] = m_l * g * l_2 * sin_t2

    return G


@jit(nopython=True)
def compute_friction_forces(q_dot, d_x, d_r, d_h, d_l, fr0x, epsx, krx):
    """
    Compute the friction forces of the crane system.

    Args:
        q_dot: Velocity vector [x_dot, r_dot, theta1_dot, theta2_dot]
        d_x: Trolley damping coefficient
        d_r: Rope damping coefficient
        d_h: Hook damping coefficient
        d_l: Load damping coefficient
        fr0x: Static friction parameter
        epsx: Regularization parameter
        krx: Dynamic friction parameter

    Returns:
        4-element friction force vector
    """
    x_dot, r_dot, theta1_dot, theta2_dot = q_dot

    F = np.zeros(4, dtype=q_dot.dtype)

    F[0] = -(fr0x * np.tanh(x_dot / epsx) + krx * np.abs(x_dot) * x_dot + d_x * x_dot)
    F[1] = -d_r * r_dot
    F[2] = -d_h * theta1_dot
    F[3] = -d_l * theta2_dot

    return F


@jit(nopython=True)
def regularize_matrix(M):
    """
    Regularize a matrix to improve numerical stability.

    Args:
        M: Matrix to regularize

    Returns:
        Regularized matrix
    """
    condition_number = np.linalg.cond(M)
    if condition_number > 1e10:
        reg_factor = min(20, max(0, np.log10(condition_number) - 10))
        M += np.eye(M.shape[0], dtype=M.dtype) * (10 ** reg_factor)
    return M


@jit(nopython=True)
def apply_constraints(q, q_dot, trolley_min, trolley_max, cable_min, cable_max, angle_min, angle_max):
    """
    Apply physical constraints to the state.

    Args:
        q: Position vector [x, r, theta1, theta2]
        q_dot: Velocity vector [x_dot, r_dot, theta1_dot, theta2_dot]
        trolley_min: Minimum trolley position
        trolley_max: Maximum trolley position
        cable_min: Minimum cable length
        cable_max: Maximum cable length
        angle_min: Minimum angle value
        angle_max: Maximum angle value

    Returns:
        Tuple of constrained position and velocity vectors
    """
    # Trolley position constraint
    if q[0] < trolley_min:
        q[0] = trolley_min
        q_dot[0] = 0
    elif q[0] > trolley_max:
        q[0] = trolley_max
        q_dot[0] = 0

    # Cable length constraint
    if q[1] < cable_min:
        q[1] = cable_min
        q_dot[1] = 0
    elif q[1] > cable_max:
        q[1] = cable_max
        q_dot[1] = 0

    # Angle constraints
    for i in [2, 3]:
        if q[i] < angle_min:
            q[i] = angle_min
            q_dot[i] = 0
        elif q[i] > angle_max:
            q[i] = angle_max
            q_dot[i] = 0

    return q, q_dot


@jit(nopython=True)
def compute_system_energy(q, q_dot, g, m_t, m_h, m_l, l_2):
    """
    Compute the total energy of the crane system.

    Args:
        q: Position vector [x, r, theta1, theta2]
        q_dot: Velocity vector [x_dot, r_dot, theta1_dot, theta2_dot]
        g: Gravity acceleration
        m_t: Trolley mass
        m_h: Hook mass
        m_l: Load mass
        l_2: Hook to load distance

    Returns:
        Total energy of the system
    """
    x, r, theta1, theta2 = q
    x_dot, r_dot, theta1_dot, theta2_dot = q_dot

    v_t = x_dot
    v_h = np.sqrt((x_dot + r * np.cos(theta1) * theta1_dot) ** 2 + (r_dot - r * np.sin(theta1) * theta1_dot) ** 2)
    v_l = np.sqrt((x_dot + r * np.cos(theta1) * theta1_dot + l_2 * np.cos(theta2) * theta2_dot) ** 2 +
                  (r_dot - r * np.sin(theta1) * theta1_dot - l_2 * np.sin(theta2) * theta2_dot) ** 2)

    T = 0.5 * (m_t * v_t ** 2 + m_h * v_h ** 2 + m_l * v_l ** 2)
    U = (m_h + m_l) * g * r * (1 - np.cos(theta1)) + m_l * g * l_2 * (1 - np.cos(theta2))

    return T + U


@jit(nopython=True)
def compute_state_derivative(state, control_inputs, g, m_t, m_h, m_l, l_2,
                             d_x, d_r, d_h, d_l, fr0x, epsx, krx,):
    """
    Compute the derivative of the state vector.

    Args:
        t: Current time
        state: Current state vector
        control_inputs: Control inputs vector
        g: Gravity acceleration
        m_t: Trolley mass
        m_h: Hook mass
        m_l: Load mass
        l_2: Hook to load distance
        d_x: Trolley damping coefficient
        d_r: Rope damping coefficient
        d_h: Hook damping coefficient
        d_l: Load damping coefficient
        fr0x: Static friction parameter
        epsx: Regularization parameter
        krx: Dynamic friction parameter
        apply_physical_constraints_flag: Whether to apply physical constraints
        trolley_min: Minimum trolley position
        trolley_max: Maximum trolley position
        cable_min: Minimum cable length
        cable_max: Maximum cable length
        angle_min: Minimum angle value
        angle_max: Maximum angle value
        time_step: Simulation time step

    Returns:
        Derivative of state vector
    """
    q, q_dot = state[:4], state[4:8]

    M = compute_mass_matrix(q, m_t, m_h, m_l, l_2)
    M += np.eye(M.shape[0]) * 1e-6  # Add regularization

    C = compute_coriolis_matrix(q, q_dot, m_h, m_l, l_2)
    G = compute_gravity_vector(q, g, m_h, m_l, l_2)
    F = compute_friction_forces(q_dot, d_x, d_r, d_h, d_l, fr0x, epsx, krx)

    M = regularize_matrix(M)

    q_ddot = np.linalg.solve(M, control_inputs + F - C @ q_dot - G)

    # Limit acceleration if needed
    q_ddot[1] = min(q_ddot[1], g)

    # Return the derivatives [q_dot, q_ddot] without integration
    result = np.zeros(8, dtype=state.dtype)
    result[:4] = q_dot  # velocities
    result[4:] = q_ddot  # accelerations

    return result