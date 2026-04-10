# 定义数据集
# 定义数据特征
x_data = [1, 2, 3]
# 定义数据标签
y_data = [2, 4, 6]

# 初始化参数
w = 4       #初始梯度值
a = 0.01    #学习率

# 定义线性回归的模型
def forword(x):
    return w * x

# 定义损失函数
def loss(xs, ys):
    lossval = 0
    for x, y in zip(xs, ys):
        y_pred = forword(x)
        lossval += (y - y_pred) ** 2
    return lossval / len(xs)

# 定义计算梯度的函数
def gradient(xs, ys):
    gradval = 0
    for x, y in zip(xs, ys):
        gradval += 2 * x * (x * w -y)
    return gradval / len(xs)

# 梯度下降法计算误差
for epoch in range(100):
    # 计算误差
    loss_value = loss(x_data, y_data)
    # 计算梯度
    grad_value = gradient(x_data, y_data)
    # 梯度更新
    w = w - a * grad_value

    print("训练轮次{}: w = {}, loss = {}".format( epoch, w, loss_value))

print("100轮后的w已经训练好了，此时我们用训练好的w进行推理，学习时间为4个小时的最终得分为:{}".format(forword(4)))
