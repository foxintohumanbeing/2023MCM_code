from matplotlib import pyplot as plt
import scienceplots
plt.style.use('ieee')
import matplotlib as mpl
import pandas as pd 
from matplotlib import ticker
import numpy as np

data = pd.read_excel('datafeatures_14_0220a.xlsx')
# data = data.to_numpy()

fig, ax = plt.subplots()

# paint here
ax.plot(data['Number in hard mode']/\
    data['Number of reported results'], '-', color='#0000B0', linewidth=1)

# ax.xaxis.set_major_formatter(
#     ticker.FuncFormatter(lambda x, pos: str(int(x)+1) if x < 6 else '7+')
# )

ax.set_xlabel('Days')
ax.set_ylabel('Hard mode ratio')
ax.legend(['Hard mode ratio'], loc='upper left')

ax.grid(alpha=0.5)
fig.tight_layout(pad=0.5)
fig.savefig('filename.png')