import numpy as np


def crane_state_derivative(state, control_inputs, g, m_t, m_h, m_l, l_2,
                           d_x, d_r, d_h, d_l, fr0x, epsx, krx):
    """Simplified compute_state_derivative function for the crane model"""
    # Extract state components
    q, q_dot = state[:4], state[4:8]
    x, r, theta1, theta2 = q
    x_dot, r_dot, theta1_dot, theta2_dot = q_dot

    # Simplified mass matrix calculation (for demonstration)
    sin_t1, cos_t1 = np.sin(theta1), np.cos(theta1)
    sin_t2, cos_t2 = np.sin(theta2), np.cos(theta2)

    M = np.zeros((4, 4))
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

    # Add small regularization to ensure matrix is invertible
    M += np.eye(4) * 1e-6

    # Simplified gravity vector
    G = np.zeros(4)
    G[1] = -(m_h + m_l) * g * cos_t1
    G[2] = (m_h + m_l) * g * r * sin_t1
    G[3] = m_l * g * l_2 * sin_t2

    # Simplified friction forces
    F = np.zeros(4)
    F[0] = -(fr0x * np.tanh(x_dot / epsx) + krx * np.abs(x_dot) * x_dot + d_x * x_dot)
    F[1] = -d_r * r_dot
    F[2] = -d_h * theta1_dot
    F[3] = -d_l * theta2_dot

    # Simplified Coriolis matrix (for demonstration)
    C = np.zeros((4, 4))
    C[0, 1] = (m_h + m_l) * cos_t1 * theta1_dot
    C[0, 2] = (m_h + m_l) * (r_dot * cos_t1 - r * sin_t1 * theta1_dot)
    C[0, 3] = -m_l * l_2 * sin_t2 * theta2_dot

    # Solve for accelerations: M * q_ddot = control_inputs + F - C * q_dot - G
    q_ddot = np.linalg.solve(M, control_inputs + F - C @ q_dot - G)

    # Return state derivatives [q_dot, q_ddot]
    result = np.zeros(8)
    result[:4] = q_dot  # velocities
    result[4:] = q_ddot  # accelerations

    return result


def rk4_step(state, control_inputs, dt, crane_params):
    """
    Perform one step of RK4 integration

    Args:
        state: Current state [x, r, theta1, theta2, x_dot, r_dot, theta1_dot, theta2_dot]
        control_inputs: Control forces [Fx, Fr, tau1, tau2]
        dt: Time step
        crane_params: Dictionary containing all crane parameters

    Returns:
        New state after time step
    """
    # Unpack parameters
    g = crane_params['g']
    m_t = crane_params['m_t']
    m_h = crane_params['m_h']
    m_l = crane_params['m_l']
    l_2 = crane_params['l_2']
    d_x = crane_params['d_x']
    d_r = crane_params['d_r']
    d_h = crane_params['d_h']
    d_l = crane_params['d_l']
    fr0x = crane_params['fr0x']
    epsx = crane_params['epsx']
    krx = crane_params['krx']

    # Function to compute state derivative with fixed parameters
    def f(state):
        return crane_state_derivative(
            state, control_inputs, g, m_t, m_h, m_l, l_2,
            d_x, d_r, d_h, d_l, fr0x, epsx, krx
        )

    # RK4 algorithm
    k1 = f(state)
    k2 = f(state + dt / 2 * k1)
    k3 = f(state + dt / 2 * k2)
    k4 = f(state + dt * k3)

    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# Example with real numbers
def run_example():
    """Run a numerical example of RK4 integration with the crane model"""

    # Crane parameters
    crane_params = {
        'g': 9.81,  # Gravity (m/s^2)
        'm_t': 500.0,  # Trolley mass (kg)
        'm_h': 50.0,  # Hook mass (kg)
        'm_l': 1000.0,  # Load mass (kg)
        'l_2': 2.0,  # Hook to load distance (m)
        'd_x': 500.0,  # Trolley damping (Ns/m)
        'd_r': 100.0,  # Rope damping (Ns/m)
        'd_h': 50.0,  # Hook damping (Nms/rad)
        'd_l': 50.0,  # Load damping (Nms/rad)
        'fr0x': 400.0,  # Static friction (N)
        'epsx': 0.01,  # Regularization parameter
        'krx': 200.0  # Dynamic friction parameter (Ns^2/m^2)
    }

    # Initial state: [x, r, theta1, theta2, x_dot, r_dot, theta1_dot, theta2_dot]
    # Starting with trolley at x=0, rope length 5m, and small hook and load angles
    initial_state = np.array([
        0.0,  # x (m)
        5.0,  # r (m)
        0.1,  # theta1 (rad) - slight hook swing
        0.2,  # theta2 (rad) - slight load swing
        0.0,  # x_dot (m/s)
        0.0,  # r_dot (m/s)
        0.0,  # theta1_dot (rad/s)
        0.0  # theta2_dot (rad/s)
    ])

    # Control inputs: [Fx, Fr, tau1, tau2]
    # Applying a force to move the trolley to the right
    control_inputs = np.array([
        1000.0,  # Fx (N) - force to move trolley right
        0.0,  # Fr (N)
        0.0,  # tau1 (Nm)
        0.0  # tau2 (Nm)
    ])

    # Time step
    dt = 0.01  # 10ms

    # Number of steps
    num_steps = 100  # Simulate 1 second

    # Storage for results
    states = np.zeros((num_steps + 1, 8))
    states[0] = initial_state

    # Run the simulation
    for i in range(num_steps):
        states[i + 1] = rk4_step(states[i], control_inputs, dt, crane_params)

        # Print progress every 10 steps
        if i % 10 == 0:
            print(f"Step {i}, Time = {i * dt:.2f}s")
            print(f"  Trolley position: {states[i + 1][0]:.4f} m")
            print(f"  Rope length: {states[i + 1][1]:.4f} m")
            print(f"  Hook angle: {states[i + 1][2]:.4f} rad")
            print(f"  Load angle: {states[i + 1][3]:.4f} rad")
            print(f"  Trolley velocity: {states[i + 1][4]:.4f} m/s")

    return states


# Execute the example and display detailed steps of first RK4 iteration
def detailed_rk4_example():
    """Show detailed calculation of first RK4 step"""

    # Crane parameters (same as above)
    crane_params = {
        'g': 9.81,
        'm_t': 500.0,
        'm_h': 50.0,
        'm_l': 1000.0,
        'l_2': 2.0,
        'd_x': 500.0,
        'd_r': 100.0,
        'd_h': 50.0,
        'd_l': 50.0,
        'fr0x': 400.0,
        'epsx': 0.01,
        'krx': 200.0
    }

    # Initial state and control inputs
    state = np.array([0.0, 5.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
    control_inputs = np.array([1000.0, 0.0, 0.0, 0.0])
    dt = 0.01

    # Unpack parameters
    g = crane_params['g']
    m_t = crane_params['m_t']
    m_h = crane_params['m_h']
    m_l = crane_params['m_l']
    l_2 = crane_params['l_2']
    d_x = crane_params['d_x']
    d_r = crane_params['d_r']
    d_h = crane_params['d_h']
    d_l = crane_params['d_l']
    fr0x = crane_params['fr0x']
    epsx = crane_params['epsx']
    krx = crane_params['krx']

    # Step 1: Calculate k1 = f(state)
    print("RK4 Step 1: k1 = f(state)")
    k1 = crane_state_derivative(
        state, control_inputs, g, m_t, m_h, m_l, l_2,
        d_x, d_r, d_h, d_l, fr0x, epsx, krx
    )
    print("k1 =", k1)

    # Step 2: Calculate k2 = f(state + dt/2 * k1)
    print("\nRK4 Step 2: k2 = f(state + dt/2 * k1)")
    state2 = state + dt / 2 * k1
    print("Intermediate state =", state2)
    k2 = crane_state_derivative(
        state2, control_inputs, g, m_t, m_h, m_l, l_2,
        d_x, d_r, d_h, d_l, fr0x, epsx, krx
    )
    print("k2 =", k2)

    # Step 3: Calculate k3 = f(state + dt/2 * k2)
    print("\nRK4 Step 3: k3 = f(state + dt/2 * k2)")
    state3 = state + dt / 2 * k2
    print("Intermediate state =", state3)
    k3 = crane_state_derivative(
        state3, control_inputs, g, m_t, m_h, m_l, l_2,
        d_x, d_r, d_h, d_l, fr0x, epsx, krx
    )
    print("k3 =", k3)

    # Step 4: Calculate k4 = f(state + dt * k3)
    print("\nRK4 Step 4: k4 = f(state + dt * k3)")
    state4 = state + dt * k3
    print("Intermediate state =", state4)
    k4 = crane_state_derivative(
        state4, control_inputs, g, m_t, m_h, m_l, l_2,
        d_x, d_r, d_h, d_l, fr0x, epsx, krx
    )
    print("k4 =", k4)

    # Final step: Combine results
    print("\nRK4 Final Step: state_new = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)")
    new_state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    print("New state =", new_state)

    return new_state


# Call the detailed example to see the RK4 calculations step by step
if __name__ == "__main__":
    print("Detailed RK4 calculation for first time step:")
    detailed_rk4_example()

    print("\nRunning full simulation:")
    states = run_example()

    # Basic analysis of results
    final_state = states[-1]
    print("\nFinal state after 1 second:")
    print(f"Trolley position: {final_state[0]:.4f} m")
    print(f"Rope length: {final_state[1]:.4f} m")
    print(f"Hook angle: {final_state[2]:.4f} rad")
    print(f"Load angle: {final_state[3]:.4f} rad")
    print(f"Trolley velocity: {final_state[4]:.4f} m/s")