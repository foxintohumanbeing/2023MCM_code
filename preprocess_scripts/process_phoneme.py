
import pandas as pd
import numpy as np

with open('phoneme_raw.txt', 'r') as f:
    lines = f.readlines()

data = pd.read_excel('../UPDATE_Problem_C_Data_Wordle.xlsx')
data = data.to_numpy()[::-1, 1:]

num_phoneme = np.zeros(len(data))
for i in range(len(lines)):
    if data[i, 1] != lines[i].split()[0]:
        print('error')
    num_phoneme[i] = len(lines[i].split()) - 1
print(num_phoneme)

with open('phoneme_processed.txt', 'w') as f:
    f.writelines(' '.join(map(str, num_phoneme.astype(np.int64))))