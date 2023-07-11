import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('output.csv')
print(df['penalty'].mean())
print(df['cost'].mean())

penalty = df['penalty'].values
bins = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.1, 1, 10, 100]
peanlty_his = np.histogram(penalty, bins=bins)

plt.figure()
plt.bar(np.arange(len(peanlty_his[0])), peanlty_his[0])
plt.xticks(np.arange(len(peanlty_his[0])), bins[1:])
plt.xlabel('penalty')
plt.ylabel('count')
plt.show()


df[["scenario", "penalty"]].groupby("scenario").agg({"penalty": ["mean", "std", "max", "min"]})

