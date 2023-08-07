
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data import Con
import evaluation


class GenDataset(Dataset):
    def __init__(self, file_path='gen_train.csv'):
        # convert into PyTorch tensors and remember them
        df = pd.read_csv(file_path)
        self.X = df[['gen_p', 'gen_q', 'gen_bus_v']].values
        self.y = df[['pen']].values
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target


def read_contingencies(network_dir, scenario):
    """Read contingencies from file, return a dataframe"""
    ctgs = Con()
    # con = case_dir + scenario + '/case.con'
    con_name = os.path.join(network_dir, scenario, 'case.con')
    ctgs.read(con_name)
    return ctgs


def generate_features(network_dir, scenario):
    ctgs = read_contingencies(network_dir, scenario)
    df = ctgs.to_df()
    df['scenario'] = scenario
    df_all = pd.DataFrame()

    for file in list(os.listdir(os.path.join(network_dir, scenario, "sol1"))):
    # for file in ['sol1_1_1.txt']:
        if not file.startswith("sol1") or not file.endswith(".txt"):
            continue
        tmp_df = df.copy()
        result = get_sol1(network_dir, scenario, file)
        mod = int(file.split('.')[0].split('_')[1])
        var = int(file.split('.')[0].split('_')[2])

        detail_file = os.path.join(network_dir, scenario, "detail", "detail_{}_{}.csv".format(mod, var))
        sol2 = pd.read_csv(detail_file)
        sol2 = sol2[~sol2["ctg"].isna()].set_index('ctg')
        tmp_df['pen'] = sol2.loc[tmp_df['label'].values]['pen'].values

        for idx, gen in enumerate(result.gen_i):
            print(idx, gen)
            tmp_df['p_gen_{}'.format(gen)] = result.gen_pow_real[idx]
        print(len(tmp_df.columns))
        for idx, gen in enumerate(result.gen_i):
            tmp_df['q_gen_{}'.format(gen)] = result.gen_pow_imag[idx]

        for idx, bus in enumerate(result.bus_i):
            tmp_df['v_bus_{}'.format(bus)] = result.bus_volt_mag[idx]
        print(len(result.gen_i))
        print(len(result.gen_pow_real))
        print(len(result.gen_pow_imag))
        print(len(result.bus_i))
        print(len(result.bus_volt_mag))
        print(len(tmp_df.columns))
        df_all = pd.concat([df_all, tmp_df], axis=0, ignore_index=True)

    return df_all


def generate_features_gen(network_dir, scenario):
    ctgs = read_contingencies(network_dir, scenario)
    df = ctgs.to_df()
    df = df[df['gen'] == 1]
    # add scenario column
    df['scenario'] = scenario
    failed_gens = df['i'].values
    gen_df = pd.DataFrame()
    branch_df = pd.DataFrame()

    for file in list(os.listdir(os.path.join(network_dir, scenario, "sol1"))):
        if not file.startswith("sol1") or not file.endswith(".txt"):
            continue
        tmp_df = df.copy()
        result = get_sol1(network_dir, scenario, file)
        mod = int(file.split('.')[0].split('_')[1])
        var = int(file.split('.')[0].split('_')[2])
        failed_gen_idx = np.searchsorted(result.gen_i, failed_gens)
        tmp_df['case_mod'] = mod
        tmp_df['case_var'] = var
        tmp_df['gen_p'] = result.gen_pow_real[failed_gen_idx]
        tmp_df['gen_q'] = result.gen_pow_imag[failed_gen_idx]
        tmp_df['gen_bus_v'] = result.bus_volt_mag[failed_gen_idx]


        ## read the corresponding detail file
        detail_file = os.path.join(network_dir, scenario, "detail", "detail_{}_{}.csv".format(mod, var))
        sol2 = pd.read_csv(detail_file)
        sol2 = sol2[~sol2["ctg"].isna()].set_index('ctg')
        tmp_df['pen'] = sol2.loc[tmp_df['label'].values]['pen'].values

        gen_df = pd.concat([gen_df, tmp_df], axis=0, ignore_index=True)

    return gen_df


def get_sol1(network_dir, scenario, case="sol1_1_1.txt"):
    raw = network_dir + scenario + '/case.raw'
    rop = network_dir + scenario + '/case.rop'
    con = network_dir + scenario + '/case.con'
    inl = network_dir + scenario + '/case.inl'

    sol1 = os.path.join(network_dir, scenario, "sol1", case)
    result = evaluation.read_sol1(raw, rop, con, inl, sol1_name=sol1)
    return result


if __name__ == "__main__":
    network_dir = '/Users/xiongjinxin/A-xjx/SRIBD/PowerModelsSecurityConstrained.jl/test/data/Network_02O-173/'
    scenario = 'scenario_15'
    ctgs = read_contingencies(network_dir, scenario)
    df = ctgs.to_df()
    print(df.head())
    # gen_df = generate_features_gen(network_dir, scenario)
    gen_df = generate_features(network_dir, scenario)
    gen_df.to_csv("train_all.csv", index=False)
    print(gen_df.head())
    # train_dataset = GenDataset('gen_train.csv')
    # val_dataset = GenDataset('gen_val.csv')
    # gen_loader_train = DataLoader(train_dataset, shuffle=True, batch_size=16)
    # gen_loader_val = DataLoader(val_dataset, shuffle=True, batch_size=16)
    # for X_batch, y_batch in gen_loader_train:
    #     print(X_batch.shape, y_batch.shape)
    #     break