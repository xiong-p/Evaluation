import os

import matplotlib.pyplot as plt

import evaluation
import pandas as pd

import numpy as np

outout_file = "output.csv"

case_dir='/Users/xiongjinxin/A-xjx/SRIBD/PowerModelsSecurityConstrained2.jl/test/data/Network_02O-173/'

## list all folders in a directory
file_names = os.listdir(case_dir)
summary = case_dir + 'summary.csv'
detail = case_dir + 'detail.csv'

# ## create a dataframe to store the output
# df = pd.DataFrame(columns=['scenario', 'sol1', 'penalty', 'cost', 'obj', 'infeas'])
# ## add a row to the dataframe
# # df.loc[0] = ['s1', 's1', 1, 2, 3, 4]
#
# c = 0
# for scenario in file_names:
#     if not scenario.startswith("s"):
#         continue
#     raw = case_dir + scenario + '/case.raw'
#     rop = case_dir + scenario + '/case.rop'
#     con = case_dir + scenario + '/case.con'
#     inl = case_dir + scenario + '/case.inl'
#
#     sol_names = os.listdir(case_dir + scenario)
#
#     for sol in sol_names:
#         if not sol.startswith("sol1"):
#             continue
#         sol1 = case_dir + scenario + '/' + sol
#         result = evaluation.run(raw, rop, con, inl, sol1_name=sol1, sol2_name=None, summary_name=summary, detail_name=detail)
#         df.loc[c] = [scenario, sol, result[1], result[2], result[3], result[0]]
#         c += 1
#     df.to_csv(outout_file)


#################
# c = 0
# mag = []
# for scenario in ["scenario_15"]:
#     mag_tmp = []
#     if not scenario.startswith("s"):
#         continue
#     raw = case_dir + scenario + '/case.raw'
#     rop = case_dir + scenario + '/case.rop'
#     con = case_dir + scenario + '/case.con'
#     inl = case_dir + scenario + '/case.inl'
#
#     # sol_names = os.listdir(case_dir + scenario)
#     sol_names = os.listdir(os.path.join(case_dir, scenario, "sol1"))
#     for sol in sol_names:
#         if not sol.startswith("sol1") or not sol.endswith(".txt"):
#             continue
#         sol1 = os.path.join(case_dir, scenario, "sol1", sol)
#         result = evaluation.read_sol1(raw, rop, con, inl, sol1_name=sol1, sol2_name=None, summary_name=summary, detail_name=detail)
#         mag_tmp.append(result.gen_pow_real[73])
#         c += 1
#     mag.append(mag_tmp)
#
# df = pd.DataFrame(mag)
# df.to_csv("mag.csv")



################################
c = 0
mag = {"scenario": [], "mod": [], "var": [], "value": []}
for scenario in ["scenario_15"]:
    # mag_tmp = []
    if not scenario.startswith("s"):
        continue
    raw = case_dir + scenario + '/case.raw'
    rop = case_dir + scenario + '/case.rop'
    con = case_dir + scenario + '/case.con'
    inl = case_dir + scenario + '/case.inl'

    # sol_names = os.listdir(case_dir + scenario)
    sol_names = os.listdir(os.path.join(case_dir, scenario, "sol1"))
    for sol in sol_names:
        if not sol.startswith("sol1") or not sol.endswith(".txt"):
            continue
        sol1 = os.path.join(case_dir, scenario, "sol1", sol)
        mod = int(sol.split(".")[0].split("_")[1])
        var = int(sol.split(".")[0].split("_")[2])
        result = evaluation.read_sol1(raw, rop, con, inl, sol1_name=sol1, sol2_name=None, summary_name=summary, detail_name=detail)
        # mag_tmp.append(result.gen_pow_real[45])
        mag["scenario"].append(scenario)
        mag["mod"].append(mod)
        mag["var"].append(var)
        mag["value"].append(result.gen_pow_real[73])
        c += 1
    # mag.append(mag_tmp)

df = pd.DataFrame(mag)
df = df.sort_values(["mod", "var"], ascending=True)
df.to_csv("mag.csv")

plt.figure()
plt.plot(np.arange(len(df)), df["value"])
plt.xlabel("sample")
plt.ylabel("value")
plt.xticks(np.arange(0, len(df), 5))
plt.show()