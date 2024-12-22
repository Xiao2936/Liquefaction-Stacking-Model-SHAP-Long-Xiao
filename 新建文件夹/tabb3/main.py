import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from sklearn import preprocessing
import sys
import os

# 获取当前脚本所在的路径
script_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))

# 构建数据文件的完整路径
data_file_path = os.path.join(script_dir, "5.25新数据.xlsx")

# 读取数据文件
data = pd.read_excel(data_file_path)

Y = data.iloc[:, -1]
X = data.iloc[:, :-1]



min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
X = torch.from_numpy(X).type(torch.float32)


Y = np.array(Y, dtype=float)
Y = torch.from_numpy(Y).reshape(-1, 1)

X = np.array(X, dtype=float)
X = torch.from_numpy(X)

X = X.type(torch.FloatTensor)
Y = Y.type(torch.FloatTensor)

BATCHSIZE = 16

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.33)
train_ds = TensorDataset(train_x, train_y)
train_dl = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True)

test_ds = TensorDataset(test_x, test_y)
test_dl = DataLoader(test_ds, batch_size=BATCHSIZE)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc_1 = nn.Linear(11, 460)
        self.fc_2 = nn.Linear(460, 144)
        self.bn1 = nn.BatchNorm1d(1)

        self.conv1 = nn.Conv1d(1, 16, 3, padding="same")
        self.bn1_conv1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, 3, padding="same")
        self.bn1_conv2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 1, 3, padding="same")
        self.bn1_conv3 = nn.BatchNorm1d(1)

        self.drop1 = nn.Dropout(0.2)
        self.mp = nn.MaxPool1d(1)
        self.fc3 = nn.Linear(144, 144)
        self.fc4 = nn.Linear(144, 1)

    def forward(self, input):
        x = F.relu(self.fc_1(input))
        x = self.drop1(x)
        x = F.relu(self.fc_2(x))

        x = x.view(-1, 1, 144)
        x = self.bn1(x)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.bn1_conv1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.bn1_conv2(x)
        x = F.relu(self.mp(self.conv3(x)))
        x = self.bn1_conv3(x)

        x = x.view(-1, 144)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x

model = MyModel()
if torch.cuda.is_available():
    model.to('cuda')

loss_fn = nn.BCELoss()
optim = torch.optim.Adamax(model.parameters(), lr=0.001)


def fit(epoch, model, trainloader, testloader, max_acc):
    correct = 0
    total = 0
    running_loss = 0
    model.train()  # 相当于告诉模型是训练模式。此时dropout会发挥作用
    total_samples_train = len(trainloader.dataset)  # 总样本数
    for x, y in trainloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            with torch.no_grad():
                y_pred = (y_pred > 0.5).type(torch.int32)
                correct += (y_pred == y).sum().item()
                total += y.size(0)
                running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader)
    epoch_acc = correct / total_samples_train

    test_correct = 0
    test_total = 0
    test_running_loss = 0

    model.eval()  # 告诉模型此为预测模式，dropout不发挥作用
    total_samples_test = len(testloader.dataset)
    for x, y in testloader:
        if torch.cuda.is_available():
            x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = (y_pred > 0.5).type(torch.int32)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(testloader)
    epoch_test_acc = test_correct / total_samples_test

    # 保存最优模型
    if epoch_test_acc > max_acc:
        max_acc = epoch_test_acc
        print("save model")
        # 保存模型语句
        torch.save(model.state_dict(),"model.pth")

    print('epoch:', epoch,
          'loss: ', round(epoch_loss, 3),
          'accuracy: ', round(epoch_acc, 3),
          'test_loss: ', round(epoch_test_loss, 3),
          'test_acc: ', round(epoch_test_acc, 3)
          )

    return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc, max_acc

epochs = 125
max_acc = 0.91

train_loss = []
train_acc = []
test_loss = []
test_acc = []

if __name__ == "__main__":
    for epoch in range(epochs):
        epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc, min_loss = fit(epoch, model, train_dl, test_dl, max_acc)

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)