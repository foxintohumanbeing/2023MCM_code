import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
df = pd.read_csv('freq_5letters.csv')

plt.hist(df['freq'], bins=100)
plt.show()