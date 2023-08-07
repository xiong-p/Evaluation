import os
from multiprocessing import Pool

from sceanrio_operations import create_and_run_scenario, create_scenario
count = 20
network_dir = '/Users/xiongjinxin/A-xjx/SRIBD/PowerModelsSecurityConstrained.jl/test/data/Network_1/'
os.chdir(network_dir)


def main(start_id, count, network_dir):
    # Define the range of values for the parallel for loop
    ids = range(start_id+1, start_id+1+count)
    for i in ids:
        create_scenario(i, network_dir)

    # Create a multiprocessing pool with the number of desired cores (use the number of available cores if not specified)
    num_cores = 4  # Replace with the number of cores you want to use
    with Pool(processes=num_cores) as pool:
        pool.starmap(create_and_run_scenario, [(i, network_dir) for i in ids])


if __name__ == "__main__":
    scenarios = list(os.listdir(network_dir))
    scenario_id = [int(s.split('_')[1]) for s in scenarios if s.startswith('scenario_')]
    if len(scenario_id) < 1:
        start_id = 0
    else:
        start_id = max(scenario_id)
    main(start_id, count, network_dir)




