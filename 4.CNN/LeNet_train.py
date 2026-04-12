import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 只显示错误和警告，忽略 Info
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 解决中文显示问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取训练集数据
data_train_path = "./NEU-DET/train/images"
data_train = pathlib.Path(data_train_path)

# 读取验证集数据
data_val_path = "./NEU-DET/validation/images"
data_val = pathlib.Path(data_val_path)

# 给数据类别放置到列表数据中
class_names = np.array(
    [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]
)

# 设置图片大小和批次数
batch_size = 16
img_height = 32
img_width = 32

# 对数据进行归一化处理
image_generator = ImageDataGenerator(rescale=1.0 / 255)
# 训练集生成器
train_data_gen = image_generator.flow_from_directory(
    directory=data_train,
    batch_size=batch_size,
    target_size=(img_height, img_width),
    classes=list(class_names),
    shuffle=True,
)
# 验证集生成器
val_data_gen = image_generator.flow_from_directory(
    directory=data_val,
    batch_size=batch_size,
    target_size=(img_height, img_width),
    classes=list(class_names),
    shuffle=False,
)

# 利用keras搭建卷积神经网络
model = keras.Sequential()
model.add(keras.Input(shape=(32, 32, 3)))
model.add(Conv2D(6, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(120, kernel_size=(5, 5), activation="relu"))
model.add(Flatten())
model.add(Dense(84, activation="relu"))
model.add(Dense(6, activation="softmax"))

# 设置训练参数：
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 开始训练网络
history = model.fit(train_data_gen, validation_data=val_data_gen, epochs=50)

# 保存训练好的模型
model.save("LeNet.keras")

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
ax2.plot(history.history["accuracy"], label="Train Accuracy")
ax2.plot(history.history["val_accuracy"], label="Validation Accuracy")
ax2.set_title("Accuracy over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
fig.savefig("training_history.png", dpi=300)
