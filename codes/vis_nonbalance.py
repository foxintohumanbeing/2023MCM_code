from matplotlib import pyplot as plt
import scienceplots
plt.style.use('ieee')
import matplotlib as mpl
import pandas as pd 
from matplotlib import ticker
import numpy as np
# mpl.rcParams['axes.linewidth'] = 2
data = pd.read_excel('datafeatures_14_0220a.xlsx')
data = data.to_numpy()

fig, ax = plt.subplots()

ax.plot(data[289, 6:13], '-', color='#B00000', linewidth=2)
ax.plot(data[252, 6:13], '-', color='#00B000', linewidth=2)
ax.plot(data[117, 6:13], '-', color='#0000B0', linewidth=2)
cmap = plt.get_cmap('Spectral')
for i in range(len(data)//6):
    ax.plot(data[i, 6:13], '-', color=cmap(np.random.rand()) ,alpha=0.25)

ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: str(int(x)+1) if x < 6 else '7+')
)
ax.grid(alpha=0.5)
ax.set_xlabel('Number of trails')
ax.set_ylabel('Probability (%)')
ax.legend(['Sample 1: "mummy"', 'Sample 2: "parer"', 'Sample 3: "train"'], loc='upper left')
# plt.show()
fig.tight_layout(pad=0.5)
fig.savefig('nonbalance.png')