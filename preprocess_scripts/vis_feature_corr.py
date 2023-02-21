from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel('datafeatures_14new.xlsx')
data = df.to_numpy()[:, 1:]
# id, word, num_total, num_hard, 1, 2, 3, 4, 5, 6, X
n = data.shape[0]
print(np.corrcoef(data[:, 14:].astype(np.float64)))
plt.imshow(
    np.corrcoef(data[:, 14:].astype(np.float64), rowvar=False)
)
fig, ax = plt.subplots()
im = ax.imshow(
    np.corrcoef(data[:, 14:].astype(np.float64), rowvar=False)
)

fig.colorbar(im, ax=ax, label='CORR')

plt.show()
plt.show()