import numpy as np
import pandas as pd
from skimage import feature
from skimage.filters.rank import threshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import datasets

# 读取数据
cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target

# 划分数据集和测试集
x_train, x_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = MinMaxScaler(feature_range = (0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# print(x_train)

# 逻辑回归
model = LogisticRegression()
model.fit(x_train, y_train)

# 打印模型参数
# print("w:{}".format(model.coef_))
# print("b:{}".format(model.intercept_))

# 利用训练好的模型进行推理测试
pre_result = model.predict(x_test)
# print(pre_result)

# 打印结果的概率
pre_result_proba = model.predict_proba(x_test)
# print(pre_result_proba)

# 获取恶性肿瘤的概率
pre_list = pre_result_proba[:,1]
# print(pre_list)

# 设置阈值
thresholds = 0.2

# 设置保存结果的列表
result = []
result_name = []

for i in range(len(pre_list)):
    if pre_list[i] > thresholds:
        result.append(1)
        result_name.append('恶性')
    else:
        result.append(0)
        result_name.append('良性')

# 打印阈值调整后的结果
print(result)
print(result_name)

# 输出结果的精确率和召回率还有f1值
report = classification_report(y_test, result, labels = [0, 1], target_names = ['良性肿瘤', '恶性肿瘤'])
print(report)