import os
import random_load_profile


def create_scenario(scenario_id, network_dir):
    # copy the scenario_01 folder
    scenario_dir = os.path.join(network_dir, 'scenario_' + str(scenario_id))
    os.mkdir(scenario_dir)
    os.chdir(scenario_dir)
    os.system('cp -r ../scenario/* .')


def create_and_run_scenario(i, network_dir):
    # create_scenario(i, network_dir)
    scenario = 'scenario_' + str(i)
    scenario_dir = os.path.join(network_dir, scenario)

    # Modify the case.raw
    case_raw = os.path.join(scenario_dir, 'case.raw')
    case_raw_orig = os.path.join(scenario_dir, 'case_orig.raw')
    random_load_profile.main(case_raw_orig, case_raw)

    # Call the julia code to generate the first stage solutions
    os.system('julia /Users/xiongjinxin/A-xjx/SRIBD/PowerModelsSecurityConstrained.jl/src/scripts/c1'
              '/goc_c1_huristic_cli.jl --scenario ' + scenario + " --file " + "inputfiles.ini")

    # Call the julia code to generate other feasible solutions for the first-stage, and calculate the second-stage
    os.system('julia /Users/xiongjinxin/A-xjx/SRIBD/PowerModelsSecurityConstrained.jl/src/scripts/c1'
              '/generate_stage1_feasible.jl --scenario ' + scenario)