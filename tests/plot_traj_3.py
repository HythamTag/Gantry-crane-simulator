import numpy as np
import matplotlib.pyplot as plt


def compute_target_position_trajectory(t, params):
    """
    Compute the target position trajectory based on given parameters.

    Parameters:
    t (float or numpy.ndarray): Time values
    params (dict): Dictionary containing trajectory parameters

    Returns:
    float or numpy.ndarray: Target position at given time(s)
    """
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


def find_stabilization_point_function(t_values, trajectory_values, target_position, tolerance=0.1):
    """
    Find when the trajectory reaches within tolerance of target position using a function.

    Returns:
    tuple: (time when target is reached, corresponding position)
    """
    for i in range(len(t_values)):
        if abs(trajectory_values[i] - target_position) <= tolerance:
            return t_values[i], trajectory_values[i]

    return None, None


def find_stabilization_point_lambda(t_values, trajectory_values, target_position, tolerance=0.1):
    """
    Find when the trajectory reaches within tolerance of target position using a lambda.

    Returns:
    tuple: (time when target is reached, corresponding position)
    """
    # Lambda to check if a value is within tolerance
    tolerance_reached = lambda q, target: np.abs(q - target) <= tolerance

    # Find the first index where tolerance is reached
    tolerance_indices = np.where(tolerance_reached(trajectory_values, target_position))[0]

    # Return the first point that meets the tolerance
    if len(tolerance_indices) > 0:
        first_index = tolerance_indices[0]
        return t_values[first_index], trajectory_values[first_index]

    return None, None


def plot_trajectory(t_values, trajectory_values, target_position, tolerance,
                    function_result, lambda_result, title):
    """
    Create a visualization of the trajectory and tolerance region.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(t_values, trajectory_values, label='Trajectory')

    # Plot target line and tolerance region
    plt.axhline(y=target_position, color='green', linestyle='--', label='Target Position')
    plt.fill_between(
        t_values,
        target_position - tolerance,
        target_position + tolerance,
        color='green',
        alpha=0.2,
        label='Tolerance Region'
    )

    # Highlight points from both methods
    colors = ['red', 'purple']
    labels = ['Function Method', 'Lambda Method']
    results = [function_result, lambda_result]

    for (result, color, label) in zip(results, colors, labels):
        if result[0] is not None:
            plt.scatter(result[0], result[1],
                        color=color, s=100, label=label)

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Position (meters)')
    plt.legend()
    plt.grid(True)
    plt.show()


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

    # Target position and tolerance
    target_position = 2.0  # meters
    tolerance = 0.01  # meters

    # Generate time values
    t_values = np.linspace(0, 10, 10000000)

    # Compute the target position trajectory
    trajectory_values = compute_target_position_trajectory(t_values, params)

    # Find when target is reached using both methods
    function_result = find_stabilization_point_function(
        t_values, trajectory_values, target_position, tolerance
    )

    lambda_result = find_stabilization_point_lambda(
        t_values, trajectory_values, target_position, tolerance
    )

    # Print results
    print("Function Method Result:")
    if function_result[0] is not None:
        print(f"Time to Reach Target: {function_result[0]:.4f}")
        print(f"Position at Target: {function_result[1]:.4f}")
    else:
        print("Target not reached")

    print("\nLambda Method Result:")
    if lambda_result[0] is not None:
        print(f"Time to Reach Target: {lambda_result[0]:.4f}")
        print(f"Position at Target: {lambda_result[1]:.4f}")
    else:
        print("Target not reached")

    # Compare results
    print("\nComparison:")
    if function_result[0] == lambda_result[0]:
        print("Both methods found EXACTLY the same point!")
    else:
        print("Methods found different points.")
        print(f"Function Method Time: {function_result[0]}")
        print(f"Lambda Method Time: {lambda_result[0]}")

    # Visualize the trajectory
    plot_trajectory(
        t_values, trajectory_values,
        target_position, tolerance,
        function_result, lambda_result,
        'Trajectory Tolerance Comparison'
    )


if __name__ == "__main__":
    main()