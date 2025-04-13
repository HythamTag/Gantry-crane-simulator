import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_target_position_trajectory(t, params):
    t_x = params['trajectory']['t_x']
    k_v = params['trajectory']['k_v']
    k_a = params['trajectory']['k_a']
    epsilon = params['trajectory']['epsilon']

    term1 = t_x / 2
    term2_numerator = np.cosh((2 * k_a * t) / k_v - epsilon)
    term2_denominator = np.cosh((2 * k_a * t) / k_v - epsilon - (2 * k_a * t_x) / k_v ** 2)
    term2_log = np.log(term2_numerator / term2_denominator)
    term2 = (k_v ** 2 / (4 * k_a)) * term2_log

    return term1 + term2


def find_stabilization_point(t_values, trajectory_values, tolerance=1e-5):
    """
    Find the final stabilization point in an S-shaped trajectory.

    Parameters:
    t_values (numpy.ndarray): Time values
    trajectory_values (numpy.ndarray): Corresponding trajectory positions
    tolerance (float): Threshold for considering the trajectory stable

    Returns:
    tuple: Stabilization time and position
    """
    # Focus on the last 30% of the data to avoid early stabilization
    start_index = int(len(t_values) * 0.7)

    for i in range(start_index + 1, len(t_values)):
        # Check if changes are consistently small
        changes_small = all(
            abs(trajectory_values[j] - trajectory_values[j - 1]) < tolerance
            for j in range(i - 5, i)
        )

        if changes_small:
            return t_values[i], trajectory_values[i]

    # If no stable point found
    return None, None


def main():
    # Define parameters
    params = {
        'trajectory': {
            't_x': 2,  # Example value
            'k_v': 0.9,
            'k_a': 0.5,
            'epsilon': 3
        }
    }

    # Generate time values
    t_values = np.linspace(0, 10, 1000)

    # Compute the target position trajectory
    trajectory_values = compute_target_position_trajectory(t_values, params)

    # Find stabilization point
    stabilization_time, stabilization_position = find_stabilization_point(
        t_values,
        trajectory_values
    )

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(t_values, trajectory_values, label='Trajectory')

    # Highlight stabilization point if found
    if stabilization_time is not None:
        plt.scatter(stabilization_time, stabilization_position,
                    color='red', s=100, label='Stabilization Point')
        print(f"Stabilization Time: {stabilization_time:.4f}")
        print(f"Stabilization Position: {stabilization_position:.4f}")

    plt.title('Target Position Trajectory')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()