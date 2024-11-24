import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from nn import TwoLayerNet
from data.mnist import load_mnist

train_size = 0
iters_num = 10000
batch_size = 100
learning_rate = 0.1


if __name__ == "__main__":
    # 准备训练数据
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=True
    )

    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size / batch_size, 1)

    print(f"训练数据加载完成，训练量：{train_size}，训练批次：{batch_size}")

    # 初始化全连接网络结构
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度
        grad = network.gradient(x_batch, t_batch)

        # 更新参数
        for k in network.params.keys():
            network.params[k] -= learning_rate * grad[k]

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            print(
                f"训练进度：{i}/{iters_num}，训练集精度：{train_acc}，测试集精度：{test_acc}"
            )
