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
y_train_one = to_categorical(y_train, 2)
y_test_one = to_categorical(y_test,2)

# 利用keras框架搭建深度学习网络模型
model = keras.Sequential()
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

# 定义训练参数：损失函数、优化器、评价指标
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# 开始训练
history = model.fit(x_train, y_train_one, epochs=150, batch_size = 16, verbose = 2, validation_data = (x_test, y_test_one))
# 保存模型
model.save('my_model.keras')

# 绘制训练集和验证集的loss值对比
plt.plot(history.history['loss'], label = 'train_loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.title('全连接神经网络loss值图')
plt.legend()
plt.show()




