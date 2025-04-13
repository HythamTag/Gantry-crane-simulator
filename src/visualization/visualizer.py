import math
import os
import pygame
import numpy as np
import matplotlib.pyplot as plt
import openpyxl


def generate_folder_name(params):
    physical_params = params['crane_system']['physical_params']
    simulation_parameters = params['simulation']
    initial_conditions = params['crane_system']['initial_conditions']
    targets = params['crane_system']['target_positions']

    folder_name = (
        f"{simulation_parameters['controller_type']}_"
        f"mt{physical_params['masses']['trolley']}_"
        f"mh{physical_params['masses']['hook']}_"
        f"ml{physical_params['masses']['load']}_"
        f"ix{initial_conditions['trolley_position']:.1f}_"
        f"il{initial_conditions['rope_length']:.1f}_"
        f"tx{targets['trolley']:.1f}_"
        f"tl{targets['rope']:.1f}_"
        f"hl{physical_params['hook_to_load_distance']:.1f}"
    )

    return folder_name


class CraneVisualizer:
    def __init__(self, params):
        self.params = params
        self.physical_parameters = params['crane_system']['physical_params']
        self.constraints_parameters = params['crane_system']['constraints']
        self.initial_conditions_parameters = params['crane_system']['initial_conditions']
        self.targets_parameters = params['crane_system']['target_positions']
        self.simulation_parameters = params['simulation']

        folder_name = generate_folder_name(params)
        print(folder_name)
        self.output_dir = os.path.join('..\\simulation_plots', folder_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.factor_scale = 500
        if self.params['visualizer']['render']:
            pygame.init()
            self.screen = pygame.display.set_mode((1600, 800))
            pygame.display.set_caption("Crane Simulation")
            self.clock = pygame.time.Clock()

    def animate(self, t_values, q_values, control_inputs, int_values):
        if not self.params['visualizer']['render']:
            return

        ## TODO bug here when i remove try except
        try:
            for t, q, u, int_val in zip(t_values, q_values, control_inputs, int_values):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                self.screen.fill((255, 255, 255))
                self.render_components(q, t, u, int_val)
                pygame.display.flip()
                self.clock.tick(1 / self.simulation_parameters['time_step'])

            pygame.quit()
        except Exception as e:
            pass

    def draw_text(self, text, position, font_size=20, color=(0, 0, 0)):
        font = pygame.font.Font(None, font_size)
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, position)

    def render_components(self, q, t, u, int_val):
        trolley_color = (0, 0, 255)  # Blue
        hook_color = (255, 0, 0)  # Red
        load_color = (0, 255, 0)  # Green

        x, l1, theta1, theta2 = q[:4]
        x_screen = int(x * self.factor_scale) + 100
        y_screen = 50

        hook_x = int(x_screen + l1 * math.sin(theta1) * self.factor_scale)
        hook_y = int(y_screen + l1 * math.cos(theta1) * self.factor_scale)
        load_x = int(hook_x + self.physical_parameters['hook_to_load_distance'] * math.sin(theta2) * self.factor_scale)
        load_y = int(hook_y + self.physical_parameters['hook_to_load_distance'] * math.cos(theta2) * self.factor_scale)

        pygame.draw.rect(self.screen, trolley_color, (x_screen - 20, y_screen - 10, 40, 20))
        pygame.draw.line(self.screen, (0, 0, 0), (x_screen, y_screen), (hook_x, hook_y), 2)
        pygame.draw.circle(self.screen, hook_color, (hook_x, hook_y), 10)
        pygame.draw.line(self.screen, (0, 0, 0), (hook_x, hook_y), (load_x, load_y), 2)
        pygame.draw.circle(self.screen, load_color, (load_x, load_y), 10)

        self.draw_text(f"Time: {t:.2f}s", (10, 10))
        self.draw_text(f"Trolley Position: {x:.2f}m", (10, 40))
        self.draw_text(f"Rope Length: {l1:.2f}m", (10, 70))
        self.draw_text(f"Control Input U: {u[0]:.2f}, {u[1]:.2f}", (10, 100))
        try:
            self.draw_text(f"Integral Error X: {int_val[0]:.2f}, L1: {int_val[1]:.2f}", (10, 130))
        except:
            pass

    def plot_results(self, t_values, q_values, q_dot_values, q_ddot_values, q_dddot_values, int_values, control_inputs, energy, trajectory_pos_vector):
        data = {
            'State Variables': (q_values, ['Trolley Position', 'Rope Length', 'Hook Angle', 'Load Angle']),
            'Velocities': (q_dot_values, ['Trolley Velocity', 'Rope Length Rate', 'Hook Angular Velocity', 'Load Angular Velocity']),
            'Accelerations': (q_ddot_values, ['Trolley Acceleration', 'Rope Length Acceleration', 'Hook Angular Acceleration', 'Load Angular Acceleration']),
            'Jerk': (q_dddot_values, ['Trolley Jerk', 'Rope Length Jerk', 'Hook Angular Jerk', 'Load Angular Jerk']),
            'Integral Errors': (int_values, ['Integral Error X', 'Integral Error L1']),
            'Control Inputs': (control_inputs, ['Trolley Force', 'Rope Length Control', 'Dummy1', 'Dummy2']),
            'Energy': (energy.reshape(-1, 1), ['Total Energy']),
            'Trajectory': (np.column_stack((trajectory_pos_vector, q_values[:, 0])), ['Desired Trajectory', 'Actual Trajectory'])
        }

        wb = openpyxl.Workbook()
        wb.remove(wb.active)  # Remove default sheet

        for sheet_name, (values, labels) in data.items():
            ws = wb.create_sheet(title=sheet_name)

            try:
                # Add headers
                headers = ['Time'] + labels
                ws.append(headers)

                # Add data
                for i, time in enumerate(t_values):
                    row = [time] + list(values[i])
                    ws.append(row)

                # Generate plot
                plt.figure(figsize=(12, 8))
                if values.ndim == 1:
                    plt.plot(t_values, values, label=labels[0])
                else:
                    for val, label in zip(values.T, labels):
                        plt.plot(t_values, val, label=label)
                plt.xlabel('Time (s)')
                plt.ylabel(sheet_name)
                plt.title(sheet_name)
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, f'{sheet_name.lower().replace(" ", "_")}.png'))
                plt.close()
            except Exception as e:
                print(f"Error processing {sheet_name}: {e}")

        wb.save(os.path.join(self.output_dir, 'crane_simulation_data.xlsx'))
        print(f"Data saved to Excel and plots generated successfully in folder: {self.output_dir}")

