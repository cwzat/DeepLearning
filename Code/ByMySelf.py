# coding=utf-8
import numpy as np

m_train = 5
m_test = 3
npx = 12

train_set_x = np.random.random_sample((npx, m_train))
test_set_x = np.random.random_sample((npx, m_test))
train_y = np.random.randint(0, 2, (1, m_train))
test_y = np.random.randint(0, 2, (1, m_test))
# 初始化参数
w = np.zeros((npx, 1))
b = 0
cost = 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def iteration(x, w1, b1, y, learning_rate):
    """

    :param b1:
    :param w1:
    :param learning_rate: 学习率
    :param x: x是训练集
    :param y: y是训练集的标签 是或者不是
    :return: w : 新的一轮迭代以后的权重
             b : 新的一轮迭代以后的偏置
             cost : 新的一轮迭代以后的cost function
    """
    z = np.dot(w1.T, x) + b1  # (1,m_train)
    A = sigmoid(z)  # (1,m_train)
    # 每次计算所有的样本的L取和就是cost
    # noinspection PyTypeChecker
    curcost = (-1.0 / m_train) * (np.dot(y, np.log(A).T) + np.dot(1 - y, np.log(1 - A).T))
    dw = (1.0 / m_train) * np.dot(x, (A - y).T)
    # noinspection PyTypeChecker
    db = (1 / m_train) * np.sum(A - y)

    return dw, db, curcost


def classification(w, b, x):
    """

    :param w: w
    :param b: b
    :param y: y
    :param x: x
    :return: y_predict
    """

    A = sigmoid(np.dot(w.T, x) + b)
    m = x.shape[1]
    y_predict = np.zeros((1, m))
    for i in range(np.shape(A)[1]):
        if A[0, i] <= 0.5:
            y_predict[0, i] = 0
        else:
            y_predict[0, i] = 1

    assert (y_predict.shape == (1, m))
    return y_predict


def model(train_x, train_set_y, test_x, test_set_y, w1, b1, learning_rate, num_iteration):
    """

    :param b1:
    :param w1:
    :param test_x:
    :param train_x:
    :param train_set_y: train_set_y
    :param test_set_x:  test_set_x
    :param test_set_y:  test_set_y
    :param learning_rate: learning rate
    :param num_iteration: num_iteration
    :return:
    """

    new_w = w1
    new_b = b1
    for i in range(num_iteration):
        dw, db, cur_cost = iteration(train_x, new_w, new_b, train_set_y, learning_rate)
        new_w = w1 - learning_rate * dw
        new_b = b1 - learning_rate * db

        if i % 1 == 0:
            print str(i) + "当前的cost=" + str(cur_cost)

        y_train_predict = classification(new_w, new_b, train_x)
        y_test_predict = classification(new_w, new_b, test_x)

        acc_train = 100 - ((train_set_y - y_train_predict).sum() / m_train) * 100
        acc_test = 100 - ((test_set_y - y_test_predict).sum() / m_test) * 100

        print "训练集的准确率是" + str(acc_train) + "\n" + "测试集的准确率是" + str(acc_test)

model(train_set_x, train_y, test_set_x, test_y, w, b, learning_rate=0.1, num_iteration=10)