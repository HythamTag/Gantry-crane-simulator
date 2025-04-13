# import cProfile
# import pstats
# from pstats import SortKey
import cProfile
import pstats

import yaml

from src.simulation.simulator import CraneSimulation


def main():
    config = load_config('../config/simulation_params.yaml')
    simulation = CraneSimulation(config)
    simulation.simulate(print_data=True)
    simulation.visualize_results()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)


if __name__ == "__main__":
    main()

    cProfile.run('main()', 'output.prof')
    # Print sorted stats
    with open('output_stats.txt', 'w') as stream:
        stats = pstats.Stats('output.prof', stream=stream).sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats()
        stats.print_callers()
