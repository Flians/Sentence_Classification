import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

class TxtDataset(Dataset):
    def __init__(self, VectData, labels):
        # 传入初始数据，特征向量和标签
        self.VectData = VectData
        self.labels = labels

    def __getitem__(self, index):
        # DataLoader 会根据 index 获取数据
        # toarray() 是因为 VectData 是一个稀疏矩阵，如果直接使用 VectData.toarray() 占用内存太大，勿尝试
        return self.VectData[index].toarray(), self.labels[index]

    def __len__(self):
        return len(self.labels)

class TextNN(nn.Module):
    def __init__(self, config):
        super(TextNN, self).__init__()
        self.config = config
        self.classifier = nn.Sequential(
            nn.Linear(config.n_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, config.num_classes)
        )
        self.optimizer =  torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=int(config.num_epochs/2), gamma=config.gamma)

        self.losser = nn.CrossEntropyLoss()

    def forward(self, x):
        output = self.classifier(x.double())
        return output.squeeze(1)

    def dataloader(self, vec_data,labels,batch_size,shuffle=True,num_workers=1):
        # 线下内存足够大可以考虑增大 num_workers，并行读取数据
        # 加载训练数据集
        dataset = TxtDataset(vec_data, labels)
        return DataLoader(dataset, batch_size=batch_size,shuffle=shuffle,num_workers=num_workers), len(dataset)
        