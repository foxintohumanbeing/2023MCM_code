from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd 
from matplotlib import ticker
import numpy as np
import scienceplots
plt.style.use('ieee')

# mpl.rcParams['axes.linewidth'] = 2
data = pd.read_excel('datafeatures_14_0220a.xlsx')
# data = data.to_numpy()
print(data['Number of reported results'])


fig , ax1 = plt.subplots()
# fig = plt.figure()

# ax1 = fig.add_axes([0,0,1,1])

ax1.plot(data['Number of reported results'], 'g', linewidth=1)#, '-', color='#0000B0', linewidth=2)

ax2 = ax1.twinx()

ax2.plot(data['Number in hard mode'], 'r', linewidth=1)#, '-', color='#0000B0', linewidth=2)

# ax.xaxis.set_major_formatter(
#     ticker.FuncFormatter(lambda x, pos: str(int(x)+1) if x < 6 else '7+')
# )
ax1.grid(alpha=0.5)
ax1.set_xlabel('Days')
ax1.set_ylabel('Number of reported - total')
ax2.set_ylabel('Number of reported - in hard mode')
ax2.set_ybound(0, 2e4)
fig.legend(['Total reported', 'Hard reported'], loc='upper center')
# plt.show()
fig.tight_layout(pad=0.5)
fig.savefig('numpeople.png')