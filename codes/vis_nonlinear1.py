from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
import scienceplots

plt.style.use('ieee')

data = pd.read_excel('datafeatures_14_0220a.xlsx')

trial = data['2 tries']
print(trial)


feature1 = data['freq var']
feature2 = data['spaced double']

fig, ax = plt.subplots()
norm = plt.Normalize(trial.min(), trial.max())
cmap = mpl.colormaps['viridis'] #https://matplotlib.org/stable/tutorials/colors/colormaps.html
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='Trial 2 Value')
ax.grid(alpha=0.5)

for i in range(len(feature1)):
    ax.scatter(feature1[i], feature2[i], color=cmap(trial[i]/trial.max()), s=20,
                alpha=0.5, edgecolors='none')
ax.set_xlabel('feature "freq_var"')
ax.set_ylabel('feature "spaced_double"')
fig.tight_layout(pad=0.5)
# ax.legend()

# plt.show()
plt.savefig('nonlinear1.png')