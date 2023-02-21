import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
df = pd.read_csv('unigram_freq.csv')
print(df)

data = df.to_numpy()
print(data)

count = 0 
f_words = []
f_freqs = []
for i in range(len(data)):
    if len(str(data[i, 0])) == 5:
        count +=1 
        f_words.append(data[i, 0])
        f_freqs.append(data[i, 1])
    

print(count)

with open('allowed_words.txt', 'r') as f:
    p_allwords = f.read().split('\n')

count = 0

p_freq = []
p_word = []
for i in range(len(p_allwords)):
    if p_allwords[i] in f_words:
        p_freq.append( f_freqs[f_words.index(p_allwords[i])] )
        p_word.append(p_allwords[i])
        count += 1
print('count', count)
# plt.hist(exist, bins=110)
# plt.show()

# nanmean = np.nanmean(p_freq)

# p_freq = np.nan_to_num(p_freq, nan=nanmean)

# f_freqs =  np.log2( np.array(f_freqs) )
# f_freqs = (f_freqs - f_freqs.mean()) / f_freqs.std()

p_freq =  np.log2( np.array(p_freq, dtype=np.float64) )
print(p_freq.mean(), p_freq.std())
p_freq = (p_freq - p_freq.mean()) / p_freq.std()

print(p_freq)
df = pd.DataFrame({
    'word': p_word, 
    'freq': p_freq
})
# df.to_csv('freq_5letters.csv')