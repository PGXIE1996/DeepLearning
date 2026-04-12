import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.layers import Input, Dense, LSTM
import keras

# 解决中文显示问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 加载数据集
dataset = pd.read_csv("LBMA-GOLD.csv", index_col=0)
# 处理空值
dataset = dataset.interpolate(method="linear")
dataset = dataset.bfill()
dataset = dataset.ffill()

# 时间序列数据不能打乱顺序
test_length = 200
train_length = dataset.values.shape[0] - test_length

# 获取训练集和测试集
train_set = dataset.iloc[:train_length, [0]]
test_set = dataset.iloc[train_length - 5 :, [0]]

# 对数据集进行归一化
sc = MinMaxScaler(feature_range=(0, 1))
train_set_scaled = sc.fit_transform(train_set)  # 归一化需要二维数据输入
test_set_scaled = sc.transform(test_set)

# 设置训练特征和训练集标签
x_train = []
y_train = []

# 利用for循环进行训练集特征和标签的制作
for i in range(train_length - 5):
    x_train.append(train_set_scaled[i : i + 5, 0])
    y_train.append(train_set_scaled[i + 5, 0])

# 将训练集转换为ndarray格式
x_train = np.array(x_train)
y_train = np.array(y_train)

# 循环神经网络训练特征格式应该是[样本数, 循环时间步, 特征个数]
x_train = x_train.reshape(len(x_train), 5, -1)

# 设置测试集特征和标签
x_test = []
y_test = []

# 利用for循环进行测试集特征和标签的制作
for i in range(test_length):
    x_test.append(test_set_scaled[i : i + 5, 0])
    y_test.append(test_set_scaled[i + 5, 0])

# 将训练集转换为ndarray格式
x_test = np.array(x_test)
y_test = np.array(y_test)

# 循环神经网络训练特征格式应该是[样本数, 循环时间步, 特征个数]
x_test = x_test.reshape(len(x_test), 5, -1)

# 利用keras添加神经网络
model = keras.Sequential()
model.add(Input(shape=(5, 1)))
model.add(LSTM(80, return_sequences=True, activation="relu"))
model.add(LSTM(100, return_sequences=False, activation="relu"))
model.add(Dense(units=10, activation="relu"))
model.add(Dense(1))

# 对搭建的神经网络进行编译
model.compile(loss="mse", optimizer=keras.optimizers.Adam(0.001), metrics=["mae"])

# 开始训练
history = model.fit(
    x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test)
)

# 保存训练好的模型
model.save("LSTM_GOLD.keras")

# 绘制图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss 曲线
ax1.plot(history.history["loss"], label="Train Loss")
ax1.plot(history.history["val_loss"], label="Validation Loss")
ax1.set_title("Loss over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.7)

# Accuracy 曲线
ax2.plot(history.history["mae"], label="Train mae")
ax2.plot(history.history["val_mae"], label="Validation mae")
ax2.set_title("mae over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("mae")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
fig.savefig("training_history.png", dpi=300)
