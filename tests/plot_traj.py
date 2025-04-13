import numpy as np
import plotly.graph_objects as go

# Define the function to compute the target position trajectory
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
t_values = np.linspace(0, 10, 500)
# Compute the target position trajectory
trajectory_values = compute_target_position_trajectory(t_values, params)

# Create the interactive plot
fig = go.Figure()

fig.add_trace(go.Scatter(x=t_values, y=trajectory_values, mode='lines', name='Trajectory'))

# Add hover labels
fig.update_traces(
    hoverinfo='text',
    hovertext=[
        f't: {t:.2f}, position: {pos:.2f}'
        for t, pos in zip(t_values, trajectory_values)
    ]
)

# Set plot titles and labels
fig.update_layout(
    title='Target Position Trajectory',
    xaxis_title='Time',
    yaxis_title='Position',
    hovermode='closest'
)

# Show the plot
fig.show()