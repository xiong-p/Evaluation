import os
import random_load_profile

count = 1
network_dir = '/Users/xiongjinxin/A-xjx/SRIBD/PowerModelsSecurityConstrained.jl/test/data/Network_1/'
os.chdir(network_dir)


def create_scenario(scenario_id, network_dir):
    # copy the scenario_01 folder
    scenario_dir = os.path.join(network_dir, 'scenario_' + str(scenario_id))
    os.mkdir(scenario_dir)
    os.chdir(scenario_dir)
    os.system('cp -r ../scenario/* .')


scenarios = list(os.listdir(network_dir))
scenario_id = [int(s.split('_')[1]) for s in scenarios if s.startswith('scenario_')]
if len(scenario_id) < 1:
    start_id = 0
else:
    start_id = max(scenario_id)

for i in range(start_id+1, start_id+1+count):
    create_scenario(i, network_dir)
    scenario = 'scenario_' + str(i)
    scenario_dir = os.path.join(network_dir, scenario)
    # modify the case.raw
    case_raw = os.path.join(scenario_dir, 'case.raw')
    case_raw_orig = os.path.join(scenario_dir, 'case_orig.raw')
    random_load_profile.main(case_raw_orig, case_raw)

    # call the julia code to generate the first stage solutions
    os.system('julia /Users/xiongjinxin/A-xjx/SRIBD/PowerModelsSecurityConstrained.jl/src/scripts/c1'
              '/goc_c1_huristic_cli.jl --scenario ' + scenario + " --file " + "inputfiles.ini")

    # call the julia code to generate other feasible solutions for the first-stage, and calculate the second-stage
    os.system('julia /Users/xiongjinxin/A-xjx/SRIBD/PowerModelsSecurityConstrained.jl/src/scripts/c1'
              '/generate_stage1_feasible.jl --scenario ' + scenario)





