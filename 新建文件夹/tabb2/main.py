from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import KFold, train_test_split as TTS
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from imblearn.over_sampling import BorderlineSMOTE


class MLPModel(nn.Module):
    def __init__(self, input_size, layer1_size, layer2_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, layer1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(layer1_size, layer2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(layer2_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class MLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size=3, layer1_size=64, layer2_size=128, epochs=500):
        self.model = MLPModel(input_size, layer1_size, layer2_size)
        self.epochs = epochs

    def fit(self, X, y):
        # 转换为PyTorch张量
        X = np.array(X)
        y = np.array(y)
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)

        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # 训练模型
        for epoch in range(self.epochs):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # 返回训练好的模型
        return self

    def predict_proba(self, X):
        # 转换为PyTorch张量
        X = np.array(X)
        X_tensor = torch.FloatTensor(X)

        # 预测概率
        with torch.no_grad():
            outputs = self.model(X_tensor)
            # predicted = (outputs > 0.5).type(torch.int32).numpy()
            outputs = np.array(outputs)
        return outputs

    def get_params(self, deep=True):
        return {'input_size': self.model.fc1.in_features,
                'layer1_size': self.model.fc1.out_features,
                'layer2_size': self.model.fc2.out_features,
                'epochs': self.epochs}
def dz2(x1,x2,x3,x4,x5,x6,x7,x8):
    data = pd.read_excel(r"./tabb2/XG.xlsx")

    Y = data.iloc[:, -2]
    X = data.iloc[:, :-2]
    # B = data.iloc[:, -1]
    column_name = X.columns

    Xtrain, Xtest, Ytrain, Ytest = TTS(X, Y, test_size=0.33, random_state=42, stratify=Y)

    smote_tomek = BorderlineSMOTE(random_state=42)
    X_resampled, Y_resampled = smote_tomek.fit_resample(Xtrain, Ytrain)

    Xtrain = X_resampled
    Ytrain = Y_resampled

    # 计算正负样本的比例
    positive_samples = sum(Ytrain == 1)
    negative_samples = sum(Ytrain == 0)

    # 避免除零错误
    if positive_samples == 0:
        scale_pos_weight = 1
    else:
        scale_pos_weight = negative_samples / positive_samples

    estimators = [("XGB",
                   XGBClassifier(learning_rate=0.108, n_estimators=136, max_depth=5, min_child_weight=1, subsample=0.78,
                                 colsample_bytree=0.9,
                                 gamma=0.9, scale_pos_weight=scale_pos_weight, reg_alpha=0.036, reg_lambda=0.466,
                                 random_state=42)),

                  ("LGBM", LGBMClassifier(learning_rate=0.22, n_estimators=56, max_depth=5, min_child_samples=20,
                                          colsample_bytree=0.9,
                                          num_leaves=30, min_child_weight=0.2, verbose=-1, random_state=42))
        , ("RDM", Pipeline([('scaler', StandardScaler()),
                            ('rfc', RandomForestClassifier(n_estimators=160, max_depth=8, random_state=42))]))]

    # 设置随机种子
    torch.manual_seed(40)

    # 创建包装后的MLP估计器
    mlp_wrapper = MLPWrapper()
    final_estimator = MLPWrapper()

    clf = StackingClassifier(estimators=estimators
                             ,final_estimator=final_estimator
                             ,stack_method='predict_proba',n_jobs=8)

    from sklearn.metrics import confusion_matrix, classification_report
    torch.manual_seed(63)
    # 1. 拟合融合模型到训练数据
    clf.fit(Xtrain, Ytrain)


    single_data = np.array([x1,x2,x3,x4,x5,x6,x7,x8]).reshape(1, -1)

    predicted_proba = clf.predict_proba(single_data)

    # 打印预测概率
    return str(int(round(predicted_proba[0][0],2)*100)) +"%"
# if __name__ == '__main__':
#     print(dz2(5.099999905,	5.000000477,	100	,0.439999968,	0.090000004,	0.49000001,	180	,7.400000095))


