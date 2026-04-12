import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2

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

# 设置图片大小
img_height = 32
img_width = 32

# 加载模型
model = load_model("LeNet.keras")

src_path = cv2.imread("NEU-DET/validation/images/crazing/crazing_241.jpg")
src = cv2.resize(src_path, (img_height, img_width))
src = src.astype("int32")
src = src / 255

# 扩充数据维度
test_img = tf.expand_dims(src, 0)

preds = model.predict(test_img)
score = preds[0]

print(
    "模型预测的结果为：{}，概率为{}".format(
        class_names[np.argmax(score)], score[np.argmax(score)]
    )
)
