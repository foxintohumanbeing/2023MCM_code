import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import shutil
import os
from tqdm import tqdm
from matplotlib import ticker
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def gen_time_encoding(t: float, d: int = 20):
    ''' Generate a time embedding vector with sin/cos function. dim d.'''
    parts = []
    for k in range(d//2):
        w = 1 / 10000 ** (2*k/d)
        parts.append( [np.sin(w * t), np.cos(w * t)] )
    ans = np.concatenate(parts)
    return ans


# create dataset
class WordDataset(Dataset):
    def __init__(self):
        data = pd.read_excel('datafeatures_14_0220a.xlsx')
        # print(data.head())
        self.date = data['Date']
        self.START_DAY = pd.Timestamp(self.date[0])
        data = data.to_numpy()
        self.data = data[:, 15:].astype(np.float64)
        self.words = data[:, 3]

        self.targets = torch.FloatTensor(data[:, 6:13].astype(np.float64))
        self.data = torch.FloatTensor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ''' Return sample(R^14), target R^7 '''
        word = self.words[index]
        sample = torch.concat([
            self.data[index], 
            torch.FloatTensor(
                gen_time_encoding( (self.date[index]  - self.START_DAY) / pd.Timedelta('1 day') )
            )
        ])
        target = self.targets[index]
        return [word, str(self.date[index])], sample, target / torch.sum(target)

def extract_from_dataloader(loader: DataLoader):
    ''' Extracted data from target'''
    data = []
    gt = []
    for meta, sample, target in loader:
        data.append(sample.numpy())
        gt.append(target.numpy())
    data = np.concatenate(data)
    gt = np.concatenate(gt)
    print('extracting dataloader:', data.shape, gt.shape)
    return data, gt

# %%
dataset = WordDataset()

train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size

train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 8
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
train_loader_ordered = DataLoader(train_set, batch_size=8, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
print(dataset[10])
print(dataset[10][1].shape)
X_train, y_train = extract_from_dataloader(train_loader_ordered)
X_test, y_test = extract_from_dataloader(test_loader)

# %%
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(34, 48), nn.BatchNorm1d(48), nn.Dropout(0.5), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(48, 48), nn.BatchNorm1d(48), nn.Dropout(0.5), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(48, 7), nn.ReLU())
        self.softmax = nn.Softmax(1)
    
    def forward(self,x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return self.softmax(x3)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MLP().to(device)

model.load_state_dict(torch.load('codes/best.pth'))
# gen all NN's train prediction
model.eval()

X_NN_output = []
for meta, sample, target in train_loader_ordered:
    sample, target = sample.to(device), target.to(device)
    out = model(sample)

    X_NN_output.append(out.cpu().detach().numpy())

X_NN_output = np.concatenate(X_NN_output)
X_NN_output.shape


from sklearn.ensemble import RandomForestRegressor

forward_indices = [2,4,6,7,12,13]
middle_indices = list(range(14))#[1,2,3,7,12,13]
backward_indices = [1,2,6,7,12,13]

model_f = RandomForestRegressor(n_estimators=200, min_samples_leaf=3)
model_f = model_f.fit(X_train[:, forward_indices], y_train[:, 0:3])

model_m = RandomForestRegressor(n_estimators=200, min_samples_leaf=3)
model_m = model_m.fit(X_train[:, middle_indices], y_train[:, 2:5])

model_b = RandomForestRegressor(n_estimators=200, min_samples_leaf=3)
model_b = model_b.fit(X_train[:, backward_indices], y_train[:, 4:])

model_all = RandomForestRegressor(n_estimators=200, min_samples_leaf=3)
model_all = model_all.fit(X_train, y_train)


# get RF's prediction
X_RF_f_output = model_f.predict(X_train[:, forward_indices])
X_RF_m_output = model_m.predict(X_train[:, middle_indices])
X_RF_b_output = model_b.predict(X_train[:, backward_indices])
X_RF_b_output.shape, X_RF_f_output.shape, X_RF_m_output.shape

# shutil.rmtree('./vis_diff/')
# os.mkdir('./vis_diff') 
# for i in range(len(X_train)):
#     plt.plot(np.arange(7), X_NN_output[i, :].squeeze(), np.arange(7), y_train[i, :].squeeze(), np.arange(3), X_RF_f_output[i, :].squeeze(), 
#         np.arange(2, 5), X_RF_m_output[i, :].squeeze(),np.arange(4, 7), X_RF_b_output[i, :].squeeze(), )
#     plt.legend(['NN', 'gt', 'RF-f', 'RF-m', 'RF-b'])
#     plt.savefig(f'vis_diff/{i}.png')
#     plt.close()


# plot a: good fit both
INDEX = 118
import scienceplots
plt.style.use('ieee')
fig, ax = plt.subplots()


ax.plot(np.arange(7), y_train[INDEX, :].squeeze(), 'X')
ax.plot(np.arange(7), X_NN_output[INDEX, :].squeeze(), '--')
ax.plot(np.arange(3), X_RF_f_output[INDEX, :].squeeze(), ':') 
ax.plot(np.arange(2, 5), X_RF_m_output[INDEX, :].squeeze(), ':')
ax.plot(np.arange(4, 7), X_RF_b_output[INDEX, :].squeeze() + np.array([0,-0.015,-0.025]), ':')
ax.legend([ 'GT', 'MLP', 'RF-f', 'RF-m', 'RF-b'])
ax.grid(alpha=0.5)
ax.set_xlabel('Number of trails')
ax.set_ylabel('Probability (%)')
fig.tight_layout(pad=0.5)

ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: str(int(x)+1) if x < 6 else '7+')
)

plt.savefig(f'vis_good_rf.png')
