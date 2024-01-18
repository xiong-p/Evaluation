import pandas as pd
import numpy as np
import os
import evaluation
import data
import math

network_dir = '/home/jxxiong/A-xjx/Network_1/'

BASE = 100


# the power need to be rescaled by the BASE the angle need to be rescaled by the pi/180. It is of radian when doing
# calculation, but it is of degree when reading from the raw file


def get_load_data(scenario):
    raw_name = os.path.join(network_dir, scenario, 'case.raw')
    p = data.Data()
    p.raw.read(raw_name)
    loads = sorted(p.raw.loads.values(), key=lambda x: x.i)
    dic = {"pl" + str(l.i): l.pl / BASE for l in loads}
    q_dic = {"ql" + str(l.i): l.ql / BASE for l in loads}
    dic.update(q_dic)
    return dic


def get_stage2_pen(scenario, code):
    if code:
        code = "_" + code
    else:
        code = ""
    detail_dir = os.path.join(network_dir, scenario, "detail", "detail" + code + ".csv")
    df = pd.read_csv(detail_dir)
    print(len(df))
    obj = df['obj'].values
    return {"stage2_pen": obj[-1] - obj[0]}


def get_stage1_sol(scenario, code, sol1_dir="sol1/sol1_"):
    sol1_dir = os.path.join(network_dir, scenario, sol1_dir + code + ".txt")
    raw_name = os.path.join(network_dir, scenario, 'case.raw')
    p = data.Data()
    p.raw.read(raw_name)
    num_bus = len(p.raw.buses)
    num_gen = len(p.raw.generators)
    s1 = evaluation.Solution1()
    s1.read(sol1_dir, num_bus, num_gen)
    # sort the bus_df by the value of column "i"
    s1.bus_df = s1.bus_df.sort_values(by="i", ascending=True).reset_index(drop=True)
    s1.gen_df = s1.gen_df.sort_values(by="i", ascending=True).reset_index(drop=True)
    dic = {"vm" + str(int(s1.bus_df.iloc[i]["i"])): s1.bus_df.iloc[i]["vm"] for i in range(len(s1.bus_df))}
    va_dic = {"va" + str(int(s1.bus_df.iloc[i]["i"])): s1.bus_df.iloc[i]["va"] * (math.pi / 180.0) for i in
              range(len(s1.bus_df))}
    pg_dic = {"pg" + str(int(s1.gen_df.iloc[i]["i"])): s1.gen_df.iloc[i]["pg"] / BASE for i in range(len(s1.gen_df))}
    qg_dic = {"qg" + str(int(s1.gen_df.iloc[i]["i"])): s1.gen_df.iloc[i]["qg"] / BASE for i in range(len(s1.gen_df))}
    dic.update(va_dic)
    dic.update(pg_dic)
    dic.update(qg_dic)
    return dic


def update_df(df, scenario, code):
    load = get_load_data(scenario)
    stage1 = get_stage1_sol(scenario, code)
    stage2 = get_stage2_pen(scenario, code)
    load.update(stage1)
    load.update(stage2)
    if df is None or len(df) == 0:
        df = pd.DataFrame(load, index=[0])
    else:
        # df = pd.concat(df, load, ignore_index=True)
        df = pd.concat([df, pd.DataFrame(load, index=[0])], ignore_index=True)

    return df

def get_stage2_penv2(scenario, code):
    if code:
        code = "_" + code
    else:
        code = ""
    detail_dir = os.path.join(network_dir, scenario, "detail", "detail" + code + ".csv")
    df = pd.read_csv(detail_dir)
    print(len(df))
    obj = df['obj'].values
    return {"stage1_obj": obj[0], "stage2_pen": obj[-1] - obj[0]}

def update_df_pen(df, scenario, code=""):
    data = {"scenario": scenario}
    stage2 = get_stage2_penv2(scenario, code)
    data.update(stage2)
    if df is None or len(df) == 0:
        df = pd.DataFrame(data, index=[0])
    else:
        df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)

    return df

###################### get only penalty ########################
# df = None
# for s in range(501, 502):
#     df = update_df_pen(df, "scenario_" + str(s))
# df.to_csv("/home/jxxiong/A-xjx/Evaluation/data/stage2_tmp.csv", index=False)


###################### genearte data ############################
df = None
for s in range(1, 9):
# for s in range(10, 10):
    for i in range(11, 14):
        for j in range(1, 6):
            code = str(i) + "_" + str(j)
            df = update_df(df, "scenario_" + str(s), code)
    i = 14
    for j in range(1, 3):
        code = str(i) + "_" + str(j)
        df = update_df(df, "scenario_" + str(s), code)

# df.to_csv("/home/jxxiong/A-xjx/Evaluation/data/stage2_tmp.csv", index=False)

# write the inactive columns to a file
cols = df.columns.tolist()
inactive_cols = []
for c in cols:
    if np.sum(df[c].values) == 0:
        inactive_cols.append(c)
with open("/home/jxxiong/A-xjx/Evaluation/data/inactive_cols.txt", "w") as f:
    for c in inactive_cols:
        f.write(c + "\n")
df.drop(columns=inactive_cols, inplace=True)
df.to_csv("/home/jxxiong/A-xjx/Evaluation/data/stage2_tmp2.csv", index=False)
# df_train = df.iloc[:184*50]
# df_train.to_csv("/Users/xiongjinxin/A-xjx/SRIBD/Evaluation/data/stage2_pen_train.csv", index=False)
# df_val = df.iloc[184*50: 194*50]
# df_val.to_csv("/Users/xiongjinxin/A-xjx/SRIBD/Evaluation/data/stage2_pen_val.csv", index=False)
# df_test = df.iloc[194*50:]
# df_test.to_csv("/Users/xiongjinxin/A-xjx/SRIBD/Evaluation/data/stage2_pen_test.csv", index=False)
#
# bus_cols = []
# for c in cols:
#     if "va" in c or "vm" in c:
#         bus_cols.append(c)
# df_train = df_train.drop(columns=bus_cols)
# df_val = df_val.drop(columns=bus_cols)
# df_test = df_test.drop(columns=bus_cols)
# df_train.to_csv("/Users/xiongjinxin/A-xjx/SRIBD/Evaluation/data/stage2_pen_train_small.csv", index=False)
# df_val.to_csv("/Users/xiongjinxin/A-xjx/SRIBD/Evaluation/data/stage2_pen_val_small.csv", index=False)
# df_test.to_csv("/Users/xiongjinxin/A-xjx/SRIBD/Evaluation/data/stage2_pen_test_small.csv", index=False)
################################################################
# df = pd.read_csv("../data/stage2_pen_active.csv")
# df = df[df['stage2_pen'] > 0]
# train_pct = 0.9
# val_pct = 0.05
# test_pct = 0.05
# num_samples = len(df)
# df_train = df.iloc[:int(train_pct * num_samples)]
# df_train.to_csv("../data/stage2_pen_train.csv", index=False)
# df_val = df.iloc[int(train_pct * num_samples): int((train_pct + val_pct) * num_samples)]
# df_val.to_csv("../data/stage2_pen_val.csv", index=False)
# df_test = df.iloc[int((train_pct + val_pct) * num_samples):]
# df_test.to_csv("../data/stage2_pen_test.csv", index=False)

###################### evaluate generated solution #############
# read in the inactive columns
# with open("/Users/xiongjinxin/A-xjx/SRIBD/Evaluation/data/inactive_cols.txt", "r") as f:
#     inactive_cols = f.readlines()
# inactive_cols = [c.strip() for c in inactive_cols]

# with open("/Users/xiongjinxin/A-xjx/SRIBD/Evaluation/data/cols_small.txt", "r") as f:
#     small_cols = f.readlines()
# small_cols = [c.strip() for c in small_cols]

# scenario = "scenario_12"
# code = "test_approx"
# stage1 = get_stage1_sol(scenario, code, sol1_dir="sol1_")
# load = get_load_data(scenario)
# load.update(stage1)
# # load the NN
# from approx_model import ApproxNet
# import torch
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# learning_rate = 0.0001
# batch_size = 128
# # NUM_HIDDEN_LIST = [512, 128, 16]
# NUM_HIDDEN_LIST = [8, 64, 8]
#
# log_dir = "/Users/xiongjinxin/A-xjx/SRIBD/Evaluation/model/weights/"
# print(f'log_dir: {log_dir}')
# print("using device: ", DEVICE)
# # input_dim = 1502
# input_dim = 502
# approx_net = ApproxNet(learning_rate, log_dir, input_dim, NUM_HIDDEN_LIST)
# approx_net.load(0)
# df = pd.DataFrame(load, index=[0])
# df.drop(columns=inactive_cols, inplace=True)
# # df = df[small_cols]
# x = df.values
# x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
# approx_net._network.eval()
# y = approx_net._network(x)*1e5
# print(y)
