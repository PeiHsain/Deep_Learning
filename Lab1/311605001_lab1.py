# -*- coding: utf-8 -*-
"""
2022 Summer DL lab1 created by Pei-Hsuan Tsai.
    Implement simple neural networks with forwarding pass and backpropagation using two hidden layers.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_linear(n=100):
    'Generate linear data.\nOutput : input and label data'
    # Random generate n input data x1 and x2 in range 0 to 1
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        # x1>x2 -> y=0; x1<=x2 -> y=1
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    'Generate XOR data.\nOutput : input and label data'
    inputs = []
    labels = []
    # Total 21 data in range 0 to 1
    for i in range(11):
        # slop = 1 -> y=0
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1 * i == 0.5:
            continue
        # slop = -1 -> y=1
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)


def show_result(x, y, pred_y):
    'Visualize the predictions and ground truth.\nx: inputs (2D array), y: ground truth label (1D array), pred_y: outputs of neural network (1D array)'
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')   
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()


class NeuralNetwork():
    'Create a two hidden layer neural network model.'
    def __init__(self, act):
        'initail weights'
        self.layer1 = [{'weight': [np.random.random() for i in range(2+1)]} for n in range(4)] #(2, 4)
        self.layer2 = [{'weight': [np.random.random() for i in range(4+1)]} for n in range(6)] #(4, 6)
        self.output = [{'weight': [np.random.random() for i in range(6+1)]} for n in range(1)] #(6, 1)
        self.act_method = act
    
    def __call__(self, x):
        'Forward'
        # the first hidden layer
        z1 = self.linear_sum(self.layer1, x)
        z1 = self.activation(z1)
        for i in range(len(self.layer1)):
            self.layer1[i]['output'] = z1[i]
        # # the second hidden layer
        z2 = self.linear_sum(self.layer2, z1)
        z2 = self.activation(z2)
        for i in range(len(self.layer2)):
            self.layer2[i]['output'] = z2[i]
        # # output of the nwtwork
        y = self.linear_sum(self.output, z2)
        y = self.activation(y)
        for i in range(len(self.output)):
            self.output[i]['output'] = y[i]
        return y

    def activation(self, x):
        'Activation functions'
        if self.act_method == 'sigmoid':
            return 1.0/(1.0 + np.exp(-x))
        if self.act_method == 'ReLU':
            return
        if self.act_method == 'tanh':
            return

    def derivative_activation(self, x):
        'The derivative of activation functions'
        if self.act_method == 'sigmoid':
            return np.multiply(x, 1.0 - x)
        if self.act_method == 'ReLU':
            return
        if self.act_method == 'tanh':
            return

    def linear_sum(self, layers, inputs):
        'Linear function to sum the weights and inputs.'
        channel = []
        # for each channel
        for weights in layers:
            outputs = weights['weight'][0]   # last weight -> bias term
            outputs += weights['weight'][1:] @ inputs
            channel.append(outputs)
        return np.array(channel)

    def backward(self, y):
        'The backward pass.\nOutput : the gradient'
        # compute wh for all weights from hidden layer to output layer, output layer -> mse backward
        for i in range(len(self.output)):   # each channel
            y_gradient = (self.output[i]['output'] - y.T)
            self.output[i]['delta'] = self.derivative_activation(self.output[i]['output']) * y_gradient #(1)
        # compute wi for all weights from input layer to hidden layer
        for i in range(len(self.layer2)):   # each channel
            layer2_gradient = 0
            for j in range(len(self.output)):
                layer2_gradient += self.output[j]['delta'] * self.output[j]['weight'][i+1]   # expect bias term
            self.layer2[i]['delta'] = self.derivative_activation(self.layer2[i]['output']) * layer2_gradient #(6)
        for i in range(len(self.layer1)):   # each channel
            layer1_gradient = 0
            for j in range(len(self.layer2)):
                layer1_gradient += self.layer2[j]['delta'] * self.layer2[j]['weight'][i+1]   # expect bias term
            self.layer1[i]['delta'] = self.derivative_activation(self.layer1[i]['output']) * layer1_gradient #(4)
        
    def updat_weight(self, ln, x):
        'Update the weights by learning rate.'
        # new_weight = old_weight - ln * backward_pass * forward_pass
        for i in range(len(self.layer1)):   # each channel -> 4
            # each weight (2+1)
            self.layer1[i]['weight'][0] -= float(ln * self.layer1[i]['delta'])
            self.layer1[i]['weight'][1:] -= ln * self.layer1[i]['delta'] * x
        for i in range(len(self.layer2)):   # each channel -> 6
            # each weight (4+1)
            data = [float(self.layer1[j]['output']) for j in range(len(self.layer1))]
            self.layer2[i]['weight'][0] -= float(ln * self.layer2[i]['delta'])
            self.layer2[i]['weight'][1:] -= ln * self.layer2[i]['delta'] * data
        for i in range(len(self.output)):   # each channel -> 1
            # each weight (6+1)
            data = [float(self.layer2[j]['output']) for j in range(len(self.layer2))]
            self.output[i]['weight'][0] -= float(ln * self.output[i]['delta'])
            self.output[i]['weight'][1:] -= ln * self.output[i]['delta'] * data


def MSE(pre_y, y):
    'MSE (mean-square error) loss function.\nOutput : error value'
    n = len(y)
    # MSE = sum((pre_y - y)^2) / n
    square = (pre_y - y) ** 2
    mse = np.sum(square) / n
    return mse


def train_model(nn, x, y):
    'Train the model to get network weights.'
    # Initialize network weights (often all random values)
    loss_log = []
    epoch = 100
    ln = 0.1
    # forEach training weights named ex
    for e in range(epoch):
        total_loss = 0
        for i in range(len(y)):
            x_tf = x[i].T
            prediction = nn(x_tf)   # forward pass
            actual = y[i]  # ground truth
            # compute error (prediction - actual) at the output units, calculate loss
            loss = MSE(prediction, actual)
            # using loss value to do backward pass to get the gradient
            nn.backward(actual)
            # update network weights //input layer not modified by error estimate
            nn.updat_weight(ln, x_tf)
            total_loss += loss
        # if e % 50 == 0: 
        print(f"epoch {e}, loss : {total_loss}")
        loss_log.append(total_loss)
    # until all examples classified correctly or another criterion satisfied
    # Visualize the learning curve
    learning_curve(loss_log)


def learning_curve(loss):
    'Plot the learning curve (loss, epoch) of the model.'
    epoch = len(loss)
    print(epoch)
    plt.title('Learning Curve', fontsize=18)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot([i for i in range(epoch)], loss)
    plt.show()


def test_model(nn, test_x):
    'Use testing data in the model to get the prediction.\nOutput : the prediction values and class'
    pred_y = []
    pred_class = []
    for x in test_x:
        # predict value
        x = x.T
        y = nn(x)
        pred_y.append(y)
        # classify
        if y < 0.5:
            c = 0
        else:
            c = 1
        pred_class.append(c)
    return np.array(pred_y).reshape(-1, 1), np.array(pred_class).reshape(-1, 1)


def accuracy(pred_value, pred_class, y):
    'Compute the accuracy of the prediction.'
    acc = 0
    n = len(y)
    for i in range(n):
        print(f"data{i} : {pred_value[i]}, pred = {pred_class[i]}, truth = {y[i]}")
        if pred_class[i] != y[i]:
            acc += 1
    acc /= n
    print(f"Accuracy of my model on test set: {acc}")


if __name__ == "__main__":
    n = 100
    # Prepare data
    # Generate input data x(x1, x2) and y
    # x_linear, y_linear = generate_linear(n)
    x_xor, y_xor = generate_XOR_easy()

    # Create model
    model = NeuralNetwork('sigmoid')
    # Training
    train_model(model, x_xor, y_xor)
    # train_model(model, x_linear, y_linear)
    # Testing
    pred_y, pred_c = test_model(model, x_xor)
    # pred_y, pred_c = test_model(model, x_linear)
    accuracy(pred_y, pred_c, y_xor)
    # accuracy(pred_y, pred_c, y_linear)

    # Visualize the result
    show_result(x_xor, y_xor, pred_c)
    # show_result(x_linear, y_linear, pred_c)
