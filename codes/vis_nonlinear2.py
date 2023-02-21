from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
import scienceplots

plt.style.use('ieee')

data = pd.read_excel('datafeatures_14_0220a.xlsx')

trial = data['6 tries']
print(trial)

feature1 = data['double']
feature2 = data['commonality']

fig, ax = plt.subplots()
norm = plt.Normalize(trial.min(), trial.max())
cmap = mpl.colormaps['plasma'] #https://matplotlib.org/stable/tutorials/colors/colormaps.html
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='Trial 6 Value')
ax.grid(alpha=0.5)

for i in range(len(feature1)):
    # print(feature1[i], feature2[i], trial[i], cmap(trial[i]/trial.max()))
    # ax.plot(feature1[i], feature2[i], 'o', color=cmap(trial[i]/trial.max()))
    ax.scatter(feature1[i], feature2[i], color=cmap(trial[i]/trial.max()), s=20,
                alpha=0.5, edgecolors='none')
ax.set_xlabel('feature "double"')
ax.set_ylabel('feature "commonality"')
fig.tight_layout(pad=0.5)
# ax.legend()

# plt.show()
plt.savefig('nonlinear2.png')