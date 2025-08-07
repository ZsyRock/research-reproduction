# %%
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# %%

import pandas as pd
data = pd.read_csv(data_file)
print(data)

# %%
input = data.iloc[:, 0:2]
print(input)

# %%
input = input.fillna(input.mean())
print(input)
# %%
output = data.iloc[:, 2]
print(output)
# %%
input = pd.get_dummies(input, dummy_na=True)
print(input)
# %%
import torch

x = torch.tensor(input.to_numpy(dtype=float))
y = torch.tensor(output.to_numpy(dtype=float))
x, y
# %%
x.shape, y.shape
# %%
u = torch.tensor([3.0, -4.0])
torch.norm(u)
# %%
%matplotlib inline
import torch
from torch.distributions import multinomial
from d2l import torch as d2l
# %%
fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()
# %%
multinomial.Multinomial(10, fair_probs).sample()
# %%
# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000  # 相对频率作为估计值
# %%
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
# %%
