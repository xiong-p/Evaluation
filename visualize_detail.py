import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os


network_dir = '/Users/xiongjinxin/A-xjx/SRIBD/PowerModelsSecurityConstrained2.jl/test/data/Network_02O-173/'
scenario = 'scenario_15'
detail_folder = 'detail'
detail_dir = os.path.join(network_dir, scenario, detail_folder)
print(detail_dir)

# Load data: read all csv files in detail_dir
detail = pd.DataFrame()
for file in os.listdir(detail_dir):
    if file.endswith('.csv'):
        tmp = pd.read_csv(os.path.join(detail_dir, file))
        tmp["file_mod"] = int(file.split(".")[0].split("_")[1])
        tmp["file_var"] = int(file.split(".")[0].split("_")[2])
        detail = pd.concat([detail, tmp])
        # detail = detail.append(pd.read_csv(os.path.join(detail_dir, file)))

# order dataframe by column "file"
detail = detail.sort_values(["file_mod", "file_var"], ascending=True)

print("total number of files in detail_dir: ", len(os.listdir(detail_dir)))
print("total number of rows in detail: ", len(detail))
print("unique number of contingencies: ", len(detail['ctg'].unique()))

# select all rows with column "ctg" equals NaN
sol1_detail = detail[detail["ctg"].isna()].reset_index()
sol2_detail = detail[~detail["ctg"].isna()].reset_index()
print(len(sol1_detail))
print(len(sol2_detail))


# select all ctg equals to ctg from sol2_detail
# tmp = sol2_detail.loc[sol2_detail["ctg"].isin([ctg])][["pen", "file_mod", "file_var"]]
# tmp = tmp.sort_values(["file_mod", "file_var"], ascending=True)
#
# plt.figure()
# plt.plot(np.arange(len(tmp)), tmp["pen"])
# plt.xlabel("samples")
# plt.ylabel("penalty")
# plt.plot()
# plt.show()

ctg_list = sol2_detail["ctg"].unique()
print(len(ctg_list))
fig1 = plt.figure(figsize=(20, 10))
for ctg in ctg_list:
    tmp = sol2_detail.loc[sol2_detail["ctg"].isin([ctg])][["pen", "file_mod", "file_var"]]
    tmp = tmp.sort_values(["file_mod", "file_var"], ascending=True)
    plt.plot(np.arange(len(tmp)), tmp["pen"], label=ctg)
plt.xlabel("samples")
plt.ylabel("penalty")
# plt.legend()
# set the x ticks every 5 samples
plt.xticks(np.arange(0, len(tmp), 5))
plt.title("{}: Contingency penalty".format(scenario))
plt.plot()
# plt.show()




sol1_detail = sol1_detail.sort_values(["file_mod", "file_var"], ascending=True)
fig2 = plt.figure(figsize=(20, 10))
plt.plot(np.arange(len(sol1_detail)), sol1_detail["pen"])
plt.xlabel("samples")
plt.ylabel("penalty")
plt.xticks(np.arange(0, len(sol1_detail), 5))
plt.title("{}: Base case sol penalty".format(scenario))
plt.plot()
# plt.show()


sol1_detail = sol1_detail.sort_values(["file_mod", "file_var"], ascending=True)
fig3 = plt.figure(figsize=(20, 10))
plt.plot(np.arange(len(sol1_detail)), sol1_detail["cost"])
plt.xlabel("samples")
plt.ylabel("cost")
plt.xticks(np.arange(0, len(sol1_detail), 5))
plt.title("{}: Base case sol cost".format(scenario))
plt.plot()
# plt.show()


sol1_tmp = sol1_detail.loc[sol1_detail["file_var"].isin([2, 3, 4])]
fig, ax1 = plt.subplots(figsize=(20, 10))
ax1.plot(np.arange(len(sol1_tmp)), sol1_tmp["cost"], label="cost")
ax1.set_xlabel("samples")
ax1.set_ylabel("cost")
ax1.set_xticks(np.arange(0, len(sol1_tmp), 3))
ax1.legend(loc="upper left")
ax2 = ax1.twinx()
ax2.plot(np.arange(len(sol1_tmp)), sol1_tmp["pen"], label="pen", color="red")
ax2.set_ylabel("penalty")
ax2.legend(loc="upper right")
plt.title("{}: Base case sol penalty /w extremes".format(scenario))
# plt.show()
# save the plot as pdf


pp = PdfPages(os.path.join(detail_dir, '{}.pdf'.format(scenario)))
pp.savefig(fig1)
pp.savefig(fig2)
pp.savefig(fig3)
pp.savefig(fig)
pp.close()