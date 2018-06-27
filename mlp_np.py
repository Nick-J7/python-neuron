import numpy as np
import pdb

def sigmoid(x):
    return 1. / (1.+np.exp(-x))


n = 4

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
    ])

target = np.array([
    [0],
    [1],
    [1],
    [0]
    ])

W0 = np.random.randn(2, 3)
b0 = np.random.randn(3)

W1 = np.random.randn(3, 1)
b1 = np.random.randn(1)


lr = 0.1

for iteration in range(10000):

    avg_loss = 0
    for i in range(n):

        y0 = np.dot(X[i,:], W0) + b0
        out0 = sigmoid(y0)

        y1 = np.dot(out0, W1) + b1
        out1 = sigmoid(y1)

        loss = 0.5 * (target[i] - out1) ** 2
        avg_loss += loss

        dloss_dout1 = out1 - target[i]
        dout1_dy1 = out1 * (1 - out1)
        dy1_db1 = 1.
        dy1_dW1 = out0.reshape([3, 1])
        dy1_dout0 = W1.T
        dout0_dy0 = out0 * (1 - out0)
        dy0_db0 = 1.
        dy0_dW0 = X[i].reshape([1, 2]).T
        #pdb.set_trace()

        b1_grad = dloss_dout1 * dout1_dy1 * dy1_db1
        W1_grad = dloss_dout1 * dout1_dy1 * dy1_dW1

        b0_grad = dloss_dout1 * dout1_dy1 * dy1_dout0 * dout0_dy0 * dy0_db0
        W0_grad = np.matmul(dy0_dW0, (dloss_dout1 * dout1_dy1 * dy1_dout0 * dout0_dy0))

        b0 -= lr * b0_grad.reshape(b0.shape)
        W0 -= lr * W0_grad
        b1 -= lr * b1_grad
        W1 -= lr * W1_grad

    if iteration % 500 == 0:
        print("{iteration}, loss: {loss:0.4f}".format(iteration=iteration, loss=avg_loss[0]))
#        print(W0, b0)
#        print(W1, b1)
#        for ii in range(n):
#            print("input:", X[ii], "taget:", target[ii])
#            y0 = np.dot(X[ii], W0) + b0
#            out0 = sigmoid(y0)
#
#            y1 = np.dot(out0, W1) + b1
#            out1 = sigmoid(y1)
#            print("prediction:", out1)

print("Result")
for i in range(n):
    print(i)
    print("input:", X[i], "taget:", target[i])
    y0 = np.dot(X[i], W0) + b0
    out0 = sigmoid(y0)

    y1 = np.dot(out0, W1) + b1
    out1 = sigmoid(y1)
    print("prediction:", out1)
