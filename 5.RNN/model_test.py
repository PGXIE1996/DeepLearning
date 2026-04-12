import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import load_model

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

# 导入模型
model = load_model("LSTM_GOLD.keras")

# 利用模型进行测试
predicted = model.predict(x_test)
# print(predicted.shape)

# 进行预测值的反归一化
prediction = sc.inverse_transform(predicted)
print(prediction.shape)

# 与真实值进行对比
real = test_set.values[5:]
print(real.shape)

# 确保 real 和 prediction 是一维数组，便于绘图
real_flat = real.flatten()
pred_flat = prediction.flatten()

# 计算 MAE 和 RMSE
mae = mean_absolute_error(real, prediction)
rmse = np.sqrt(mean_squared_error(real, prediction))

print(f"测试集 MAE : {mae:.4f}")
print(f"测试集 RMSE: {rmse:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 散点图
axes[0].scatter(real, prediction, alpha=0.6, edgecolors="k", linewidth=0.5)
min_val = min(real.min(), prediction.min())
max_val = max(real.max(), prediction.max())
axes[0].plot([min_val, max_val], [min_val, max_val], "r--", label="y = x")
axes[0].set_xlabel("真实值")
axes[0].set_ylabel("预测值")
axes[0].set_title("预测值与真实值散点图")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.7)

# 时序图
axes[1].plot(real, label="真实值", linewidth=2)
axes[1].plot(prediction, label="预测值", linewidth=2, alpha=0.8)
axes[1].set_xlabel("测试集时间步")
axes[1].set_ylabel("黄金价格")
axes[1].set_title(f"时间序列对比 (MAE={mae:.4f}, RMSE={rmse:.4f})")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# 可选：保存图片
fig.savefig("prediction_vs_real.png", dpi=300)
