import torch
import pandas as pd
import numpy as np
from sklearn import preprocessing
from torch import nn
import torch.nn.functional as F


# 定义模型类
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


# 定义预测方法,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11

def predict(data_file_path,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11):
    # 创建模型并加载模型参数
    model = MyModel()
    model.load_state_dict(torch.load("./tabb3/model.pth",map_location=torch.device('cpu')))
    # 如果有GPU设备可用，将模型移动到GPU上
    if torch.cuda.is_available():
        model.to('cuda')

    # 读取新的输入数据
    data = pd.read_excel(data_file_path)
    X = data.iloc[:, :-1]
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X)
    X = min_max_scaler.transform(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11]).reshape(1, -1))
    X = torch.from_numpy(X).type(torch.float32)

    if torch.cuda.is_available():
        X = X.to('cuda')

    model.eval()  # 设置模型为预测模式
    with torch.no_grad():
        predictions = model(X)

    # 将预测结果转换为numpy数组并输出
    predictions = predictions.numpy()  # 将预测结果从GPU移动到CPU（如适用）

    # 返回预测结果的概率
    prediction_percentages = [round(float(pred[0]) * 100, 2) for pred in predictions]
    print(prediction_percentages)
    return prediction_percentages

def dz3(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11):
    data_file_path = "./tabb3/5.25新数据.xlsx"  # 输入数据文件路径

    # 调用预测方法
    prediction_probabilities = predict(data_file_path, x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11)
    print(prediction_probabilities[0])
    return str(prediction_probabilities[0])+"%"
if __name__ == '__main__':

    data_file_path = "5.25新数据.xlsx"  # 输入数据文件路径

    # 调用预测方法
    prediction_probabilities = predict(data_file_path,5.099999905,	5.000000477,	100,	0.439999968,	90.79999542	,44.20000076	,0.090000004	,0.49000001	,180	,7.400000095,	0.399999976)
    print(prediction_probabilities[0])
