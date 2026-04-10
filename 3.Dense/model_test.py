import os
# 1. 设置TensorFlow的C++日志级别，屏蔽oneDNN和CPU指令集提示
# '2' 将过滤掉INFO级别的提示，只显示WARNING和ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 2. 设置CUDA可见设备为-1，屏蔽GPU不可用的提示
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler          # 数据归一化
from sklearn.model_selection import train_test_split    # 数据集划分
import keras
from tensorflow.keras.layers import Dense               # 引入全连接层
from tensorflow.keras.utils import to_categorical       # 独热向量编码
from sklearn.metrics import classification_report       # 分类模型评估工具
from sklearn import datasets                            # 引入数据集
from keras.models import load_model                     # 导入模型

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
cancer = datasets.load_breast_cancer()
# 提取特征和标签
X, y = cancer.data, cancer.target
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# 将数据归一化
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# 将标签转换为独热编码
y_test_one = to_categorical(y_test, 2)

# 导入训练好的模型
model = load_model('my_model.keras')

# 利用训练好的模型进行测试（推理）
predict = model.predict(x_test)
y_pred = np.argmax(predict, axis = 1)

# 将识别的结果转化为汉字
result = np.where(y_pred == 0, '良性', '恶性')
# print(result)

# 打印模型的精确度和召回
report = classification_report(y_test, y_pred, labels=[0, 1], target_names=['良性', '恶性'])
print(report)


