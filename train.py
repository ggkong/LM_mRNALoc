import os
directory_path = "../Data/BertFeature"

file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

file_paths_train = [
 '../Data/BertFeature/Cytoplasm_train.fasta.csv',
 '../Data/BertFeature/Endoplasmic_reticulum_train.fasta.csv',
 '../Data/BertFeature/Extracellular_region_train.fasta.csv',
 '../Data/BertFeature/Mitochondria_train.fasta.csv',
 '../Data/BertFeature/Nucleus_train.fasta.csv']

file_paths_test = [
    '../Data/BertFeature/Cytoplasm_indep.fasta.csv',
    '../Data/BertFeature/Endoplasmic_reticulum_indep.fasta.csv',
    '../Data/BertFeature/Extracellular_region_indep.fasta.csv',
    '../Data/BertFeature/Mitochondria_indep.fasta.csv',
    '../Data/BertFeature/Nucleus_indep.fasta.csv',
]

import pandas as pd
num_categories = 5
combined_df_train = pd.DataFrame()
for i, file_path in enumerate(file_paths_train):
    # 读取当前文件的数据
    df = pd.read_csv(file_path, header=None).iloc[1:, 1:]  # 或者调整为适合你的数据的读取方式
    # 生成独热编码标签
    one_hot_label = [0] * num_categories
    one_hot_label[i] = 1
    for j, label_value in enumerate(one_hot_label):
        df[f'Label_{j+1}'] = label_value

    # 将当前文件的数据添加到总的DataFrame中
    combined_df_train = pd.concat([combined_df_train, df], ignore_index=True)

import pandas as pd

combined_df_test = pd.DataFrame()
for i, file_path in enumerate(file_paths_test):
    # 读取当前文件的数据
    df = pd.read_csv(file_path, header=None).iloc[1:, 1:]  # 或者调整为适合你的数据的读取方式
    # 生成独热编码标签
    one_hot_label = [0] * num_categories
    one_hot_label[i] = 1
    for j, label_value in enumerate(one_hot_label):
        df[f'Label_{j+1}'] = label_value

    # 将当前文件的数据添加到总的DataFrame中
    combined_df_test = pd.concat([combined_df_test, df], ignore_index=True)




def getSeqFeature(train_file, test_file):
    train_pd = pd.read_csv(train_file, header=None)
    test_pd = pd.read_csv(test_file, header=None)
    train_pd_feature = train_pd.iloc[:, 1:]
    pd_feature = test_pd.iloc[:, 1:]
    train_feature = train_pd_feature.values
    test_feature = pd_feature.values
    return train_feature, test_feature


CKSNAP_file_train = './feature/CKSNAP_all_train.csv'
CKSNAP_file_test = './feature/CKSNAP_all_test.csv'
CKSNAP_train, CKSNAP_test = getSeqFeature(CKSNAP_file_train, CKSNAP_file_test)

TNC_file_train = './feature/TNC_all_train.csv'
TNC_file_test = './feature/TNC_all_test.csv'
TNC_train, TNC_test = getSeqFeature(TNC_file_train, TNC_file_test)

DNC_file_train = './feature/DNC_all_train.csv'
DNC_file_test = './feature/DNC_all_test.csv'
DNC_train, DNC_test = getSeqFeature(DNC_file_train, DNC_file_test)

X_train = combined_df_train.iloc[:, :-5].values  # 假设最后5列是标签
y_train = combined_df_train.iloc[:, -5:].values

import numpy as np
X_train = np.concatenate([X_train, CKSNAP_train, TNC_train, DNC_train], axis=1)


X_test = combined_df_test.iloc[:, :-5].values  # 假设最后5列是标签
y_test = combined_df_test.iloc[:, -5:].values


X_test = np.concatenate([X_test, CKSNAP_test, TNC_test , DNC_test], axis=1)

import numpy as np
train_label = np.sum(y_train, axis=0)


#%%
# 数据增强模块
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import BorderlineSMOTE
import numpy as np
counts = np.sum(y_train, axis=0)
# 初始化增强后的数据集和标签集合
 X_train_augmented = X_train.copy()
 y_train_augmented = y_train.copy()

 for i in range(5):  # 对每个类别执行
     target_num = int(np.sum(y_train, axis=0)[i] * 5)
     temp_labels = y_train[:, i]
     smote = BorderlineSMOTE(sampling_strategy={1: target_num})
     X_resampled, y_resampled = smote.fit_resample(X_train, temp_labels)
     X_new = X_resampled[-(target_num-train_label[i]):]
     y_new = y_resampled[-(target_num-train_label[i]):]
     new_labels = np.zeros((len(y_new), 5))
     new_labels[:, i] = 1
     X_train_augmented = np.vstack([X_train_augmented, X_new])
     y_train_augmented = np.vstack([y_train_augmented, new_labels])



from torch.utils.data import Dataset, DataLoader
# 自定义Dataset类
class ProteinDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)

from torch.utils.data import WeightedRandomSampler

class BalancedSampler:
    def __init__(self, dataset_labels):
        """
        初始化平衡采样器。

        参数:
        - dataset_labels: 数据集中所有样本的独热编码标签张量。
        """
        self.dataset_labels = dataset_labels

    def get_sampler(self):
        """
        根据类别权重获取一个WeightedRandomSampler实例。
        """
        # 将独热编码标签转换为类别索引
        labels = torch.argmax(self.dataset_labels, dim=1)

        class_sample_counts = torch.tensor(
            [(labels == t).sum().item() for t in torch.unique(labels, sorted=True)]
        )

        # 计算每个样本的权重
        weight = 1. / class_sample_counts.float()
        samples_weight = torch.tensor([weight[t].item() for t in labels])

        # 创建WeightedRandomSampler
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return sampler

import torch
# sampler = BalancedSampler(torch.from_numpy(y_train)).get_sampler()
train_dataset = ProteinDataset(X_train, y_train)
# train_dataset = ProteinDataset(X_train_augmented, y_train_augmented)
test_dataset = ProteinDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=10240, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


import torch.nn as nn
class BottleNeck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(BottleNeck, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity  # Skip Connection
        out = self.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, num_classes=5, init_weights=True):
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 8)),
            nn.ReLU(inplace=True),
        )

        # 嵌入BottleNeck模块，这里使用示例参数，你需要根据实际情况调整
        self.bottleneck = BottleNeck(in_channels=8, mid_channels=16, out_channels=8)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=118, nhead=2)


        self.classifier = nn.Sequential(
            nn.Linear(118, 64),  
            nn.ReLU(inplace=True),
            # nn.Linear(256, 128),  
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 64),  
            # nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.bottleneck(x)
        x = x.squeeze(3)
        x = self.encoder_layer(x)
        x = torch.mean(x, axis=1)
        x = self.classifier(x)
        return x
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

model = Model()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()  # 由于使用了Sigmoid，适合二分类问题的BCELoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
acc_list = []
num_epochs = 10000  # 假定训练10轮
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    for idx , (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(-1, 118, 8)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# torch.save(model.state_dict(), 'model_state_dict.pth')


    model.eval()  # 设置模型为评估模式
    y_true, y_pred = [], []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        inputs = inputs.view(-1, 118, 8)
        labels = labels.cpu().detach().numpy()  # 真实标签
        outputs = model(inputs).cpu().detach().numpy()  # 预测输出
        outputs = np.argmax(outputs, axis=1)

        labels_indices = [np.argmax(label) for label in labels]

        y_true.extend(labels_indices)
        y_pred.extend(outputs)

    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    # if accuracy > 84 and accuracy < 85:
    # torch.save(model.state_dict(), 'model_state_dict_stop.pth')
    acc_list.append(accuracy)
    print(f"Accuracy: {accuracy}")
