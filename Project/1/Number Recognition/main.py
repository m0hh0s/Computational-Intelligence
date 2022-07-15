import numpy as np
from read_MNIST import read_train_set, read_test_set
import matplotlib.pyplot as plt


learning_size = 60000
batch_size = 50
learning_rate = 1
epoch = 5
x = np.zeros((epoch, 1))  # for plotting
y = np.zeros((epoch, 1))  # for plotting
input = []  # 784*1
a1 = []  # 16*1
a2 = []  # 16*1
a3 = []  # 10*1
b1 = np.zeros((16, 1))  # 16*1
b2 = np.zeros((16, 1))  # 16*1
b3 = np.zeros((10, 1))  # 10*1
w1 = np.random.randn(16, 784)  # 16*784
w2 = np.random.randn(16, 16)  # 16*16
w3 = np.random.randn(10, 16)  # 10*16
grad_a1 = np.zeros((16, 1))  # 16*1
grad_a2 = np.zeros((16, 1))  # 16*1
grad_a3 = np.zeros((10, 1))  # 10*1
grad_b1 = np.zeros((16, 1))  # 16*1
grad_b2 = np.zeros((16, 1))  # 16*1
grad_b3 = np.zeros((10, 1))  # 10*1
grad_w1 = np.zeros((16, 784))  # 16*784
grad_w2 = np.zeros((16, 16))  # 16*16
grad_w3 = np.zeros((10, 16))  # 10*16
cost = np.zeros((10, 1))  # 10*1


def sigmoid(x):
    return np.divide(1, np.add(1, np.exp(-x)))


def sigmoid_derivative(x):
    return np.multiply(x, np.subtract(1, x))


def cost_for_image(image, label):
    global a1, a2, a3, w1, w2, w3, b1, b2, b3, cost
    a1 = sigmoid(np.add(np.matmul(w1, image), b1))
    a2 = sigmoid(np.add(np.matmul(w2, a1), b2))
    a3 = sigmoid(np.add(np.matmul(w3, a2), b3))
    cost = np.subtract(a3, label)


def initialize_grad():
    global grad_w1, grad_w2, grad_w3, grad_b1, grad_b2, grad_b3
    grad_b1 = np.zeros((16, 1))
    grad_b2 = np.zeros((16, 1))
    grad_b3 = np.zeros((10, 1))
    grad_w1 = np.zeros((16, 784))
    grad_w2 = np.zeros((16, 16))
    grad_w3 = np.zeros((10, 16))


def update_weights_and_biases():
    global a1, a2, a3, w1, w2, w3, b1, b2, b3, grad_w1, grad_w2, grad_w3, grad_b1, grad_b2, grad_b3, learning_rate
    w1 = np.subtract(w1, np.multiply(learning_rate, np.divide(grad_w1, batch_size)))
    w2 = np.subtract(w2, np.multiply(learning_rate, np.divide(grad_w2, batch_size)))
    w3 = np.subtract(w3, np.multiply(learning_rate, np.divide(grad_w3, batch_size)))
    b1 = np.subtract(b1, np.multiply(learning_rate, np.divide(grad_b1, batch_size)))
    b2 = np.subtract(b2, np.multiply(learning_rate, np.divide(grad_b2, batch_size)))
    b3 = np.subtract(b3, np.multiply(learning_rate, np.divide(grad_b3, batch_size)))


def grad_a(layer, k):
    sum = 0
    if layer == 3:
            sum = 2 * cost[k]
    elif layer == 2:
        for j in range(10):
            sum += grad_a3[j] * sigmoid_derivative(a3[j]) * w3[j][k]
    else:
        for j in range(16):
            sum += grad_a2[j] * sigmoid_derivative(a2[j]) * w2[j][k]
    return sum


def grad_w(layer, k ,j):
    if layer == 3:
        return grad_a3[j] * sigmoid_derivative(a3[j]) * a2[k]
    elif layer == 2:
            return grad_a2[j] * sigmoid_derivative(a2[j]) * a1[k]
    else:
            return grad_a1[j] * sigmoid_derivative(a2[j]) * input[k]


def grad_b(layer, k):
    if layer == 3:
        return grad_a3[k] * sigmoid_derivative(a3[k])
    elif layer == 2:
        return grad_a2[k] * sigmoid_derivative(a2[k])
    else:
        return grad_a1[k] * sigmoid_derivative(a1[k])


def grad_a_calc():
    global grad_a2, grad_a1, grad_a3
    grad_a3 = np.multiply(2, cost)
    grad_a2 = np.matmul(np.array(w3).transpose(), np.multiply(2, np.multiply(sigmoid_derivative(a3), cost)))
    grad_a1 = np.matmul(np.array(w2).transpose(), grad_a2)


def grad_w_calc():
    global grad_w1, grad_w2, grad_w3
    grad_w1 = np.add(grad_w1, np.matmul(grad_a1, np.array(input).transpose()))
    grad_w2 = np.add(grad_w2, np.matmul(grad_a2, np.array(a1).transpose()))
    grad_w3 = np.add(grad_w3, np.matmul(np.multiply(2, np.multiply(sigmoid_derivative(a3), cost)), np.array(a2).transpose()))


def gra_b_calc():
    global grad_b1, grad_b2, grad_b3
    # for k in range(16):
    #     grad_b1[k] += grad_b(1, k)
    # for k in range(16):
    #     grad_b2[k] += grad_b(2, k)
    # for k in range(10):
    #     grad_b3[k] += grad_b(3, k)
    grad_b1 = np.add(grad_b1, np.multiply(sigmoid_derivative(a1), grad_a1))
    grad_b2 = np.add(grad_b2, np.multiply(sigmoid_derivative(a2), grad_a2))
    grad_b3 = np.add(grad_b3, np.multiply(sigmoid_derivative(a3), grad_a3))


def cost_graph(e):
    global cost, x, y
    sum = 0
    for n in range(100):
        cost_for_image(train_set[n][0], train_set[n][1])
        cost = np.power(cost, 2)
        for m in range(10):
            sum += cost[m]
    sum /= 100
    x[e] = e
    y[e] = sum


def test_with_train_set():
    correct = 0
    for n in range(60000):
        cost_for_image(train_set[n][0], train_set[n][1])
        if np.nanargmax(train_set[n][1]) == np.nanargmax(a3):
            correct += 1
    correct = correct * 100 / 60000
    print("Test Result With Train Set Shows " + str(correct) + "% Accuracy")


def test_with_test_set():
    correct = 0
    for n in range(10000):
        cost_for_image(test_set[n][0], test_set[n][1])
        if np.nanargmax(test_set[n][1]) == np.nanargmax(a3):
            correct += 1
    correct = correct * 100 / 10000
    print("Test Result With Test Set Shows " + str(correct) + "% Accuracy")


print("Starting")
train_set = read_train_set(60000)
test_set = read_test_set(10000)
print("Train Set & Test Set Loaded")
for i in range(epoch):
    np.random.shuffle(train_set)
    for j in range(int(learning_size / batch_size)):
        initialize_grad()
        for k in range(batch_size):
            input = train_set[j * batch_size + k][0]
            cost_for_image(train_set[j * batch_size + k][0], train_set[j * batch_size + k][1])
            grad_a_calc()
            grad_w_calc()
            gra_b_calc()
        update_weights_and_biases()
    # cost_graph(i)
print("Training Over!")
test_with_train_set()
test_with_test_set()
# plt.plot(x, y)
# plt.show()
