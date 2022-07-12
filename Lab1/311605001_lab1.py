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
    def __init__(self, act='sigmoid'):
        'initail weights'
        self.layer1 = [{'weight': [np.random.random() for i in range(2+1)]} for n in range(4+1)] #(2, 4) + bias
        self.layer2 = [{'weight': [np.random.random() for i in range(4+1)]} for n in range(6+1)] #(4, 6) + bias
        self.output = [{'weight': [np.random.random() for i in range(6+1)]} for n in range(1)] #(6, 1)
        self.act_method = act
    
    def __call__(self, x):
        'Forward'
        # the first hidden layer
        z1 = self.linear_sum(self.layer1, x)
        z1 = self.activation(z1)
        for i in range(len(self.layer1)):
            self.layer1[i]['output'] = z1[i]
        # the second hidden layer
        z2 = self.linear_sum(self.layer2, z1)
        z2 = self.activation(z2)
        for i in range(len(self.layer2)):
            self.layer2[i]['output'] = z2[i]
        # output of the nwtwork
        y = self.linear_sum(self.output, z2)
        y = self.activation(y)
        for i in range(len(self.output)):
            self.output[i]['output'] = y[i]
        return y

    def activation(self, x):
        'Activation functions.\nsigmoid, ReLU, leaky ReLU, tanh, without'
        if self.act_method == 'sigmoid':
            return 1.0/(1.0 + np.exp(-x))
        if self.act_method == 'ReLU':
            return np.maximum(0.0, x)
        if self.act_method == 'leaky ReLU':
            return np.maximum(0.1*x, x)
        if self.act_method == 'tanh':
            return np.tanh(x)
        if self.act_method == 'without':
            return x

    def derivative_activation(self, x):
        'The derivative of activation functions.\nsigmoid, ReLU, leaky ReLU, tanh, without'
        if self.act_method == 'sigmoid':
            return np.multiply(x, 1.0 - x)
        if self.act_method == 'ReLU':
            if x < 0:
                return 0.0
            else:
                return 1.0
        if self.act_method == 'leaky ReLU':
            if x < 0:
                return 0.1
            else:
                return 1.0
        if self.act_method == 'tanh':
            return 1.0 - (x ** 2)
        if self.act_method == 'without':
            return 1.0

    def linear_sum(self, layers, inputs):
        'Linear function to sum the weights and inputs.'
        channel = []
        # for each channel
        for weights in layers:
            outputs = weights['weight'] @ inputs
            channel.append(outputs)
        return np.array(channel)

    def backward(self, y):
        'The backward pass.\nOutput : the gradient'
        # compute wh for all weights from hidden layer to output layer, output layer -> mse backward
        for i in range(len(self.output)):   # each channel, output layer
            y_gradient = (self.output[i]['output'] - y) # derivation of the loss function
            self.output[i]['delta'] = self.derivative_activation(self.output[i]['output']) * y_gradient
        # compute wi for all weights from input layer to hidden layer
        for i in range(len(self.layer2)):   # each channel, hidden layer 2
            layer2_gradient = 0
            for j in range(len(self.output)):
                layer2_gradient += self.output[j]['delta'] * self.output[j]['weight'][i]   # expect bias term
            self.layer2[i]['delta'] = self.derivative_activation(self.layer2[i]['output']) * layer2_gradient
        for i in range(len(self.layer1)):   # each channel, hidden layer 1
            layer1_gradient = 0
            for j in range(len(self.layer2)):
                layer1_gradient += self.layer2[j]['delta'] * self.layer2[j]['weight'][i]   # expect bias term
            self.layer1[i]['delta'] = self.derivative_activation(self.layer1[i]['output']) * layer1_gradient
        
    def updat_weight(self, ln, x):
        'Update the weights by learning rate.'
        # new_weight = old_weight - ln * backward_pass * forward_pass
        for i in range(len(self.layer1)):   # each channel
            # each weight
            self.layer1[i]['weight'] -= ln * x * self.layer1[i]['delta']
        for i in range(len(self.layer2)):   # each channel
            # each weight
            data = [self.layer1[j]['output'] for j in range(len(self.layer1))]
            data = np.array(data).flatten()
            self.layer2[i]['weight'] -= ln * data * self.layer2[i]['delta']
        for i in range(len(self.output)):   # each channel
            # each weight
            data = [float(self.layer2[j]['output']) for j in range(len(self.layer2))]
            data = np.array(data).flatten()
            self.output[i]['weight'] -= ln * data * self.output[i]['delta']

    def predict(self, test):
        'Get the prediction with testing data.'
        # Forward pass
        z1 = self.linear_sum(self.layer1, test)
        z1 = self.activation(z1)
        # the second hidden layer
        z2 = self.linear_sum(self.layer2, z1)
        z2 = self.activation(z2)
        # output of the nwtwork
        y = self.linear_sum(self.output, z2)
        y = self.activation(y)
        return y


def MSE(pre_y, y):
    'MSE (mean-square error) loss function.\nOutput : error value'
    n = len(y)
    # MSE = sum((pre_y - y)^2) / 2n
    square = 0.5 * (pre_y - y) ** 2
    mse = np.sum(square) / n
    return mse


def train_model(nn, x, y, epoch, ln):
    'Train the model to get network weights.'
    # Initialize network weights (often all random values)
    X = np.append(x, np.ones((len(x), 1)), axis=1) # add bias term at last
    loss_log = []
    # forEach training weights named ex
    for e in range(epoch):
        total_loss = 0
        for i in range(len(y)):
            x_data = np.atleast_2d(X[i]).T
            prediction = nn(x_data)   # forward pass
            actual = y[i]  # ground truth
            # compute error (prediction - actual) at the output units, calculate loss
            loss = MSE(prediction, actual)
            # using loss value to do backward pass to get the gradient
            nn.backward(actual)
            # update network weights //input layer not modified by error estimate
            nn.updat_weight(ln, X[i])
            total_loss += loss
        if e % 500 == 0: 
            print(f"epoch {e}, loss : {total_loss}")
            loss_log.append(total_loss)
    # until all examples classified correctly or another criterion satisfied
    # Visualize the learning curve
    learning_curve(loss_log)


def learning_curve(loss):
    'Plot the learning curve (loss, epoch) of the model.'
    epoch = len(loss)
    plt.title('Learning Curve', fontsize=18)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot([i*500 for i in range(epoch)], loss)
    plt.show()


def test_model(nn, test_x):
    'Use testing data in the model to get the prediction.\nOutput : the prediction values and class'
    pred_y = []
    pred_class = []
    # add bias term at last
    x_bias = np.append(test_x, np.ones((len(test_x), 1)), axis=1)
    for x in x_bias:
        # predict value
        x = np.atleast_2d(x).T
        y = nn.predict(x)
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
    error = 0
    n = len(y)
    for i in range(n):
        print(f"data{i} : {pred_value[i]}, pred = {pred_class[i]}, truth = {y[i]}")
        if pred_class[i] != y[i]:
            error += 1
    error /= n
    print(f"Accuracy of my model on test set: {1-error}")


if __name__ == "__main__": 
    # Prepare data
    n = 100
    # epoch_linear = 13000
    # ln_linear = 0.01
    epoch_xor = 15000
    ln_xor = 0.3

    # Generate input data x(x1, x2) and y 
    x_xor, y_xor = generate_XOR_easy()
    # x_linear, y_linear = generate_linear(n)

    # Create model, activation function -> 'sigmoid', 'ReLU', 'leaky ReLU', 'tanh', 'without'.
    model = NeuralNetwork('leaky ReLU')

    # Training
    train_model(model, x_xor, y_xor, epoch_xor, ln_xor)
    # train_model(model, x_linear, y_linear, epoch_linear, ln_linear)

    # Testing
    pred_y, pred_c = test_model(model, x_xor)
    accuracy(pred_y, pred_c, y_xor)
    # pred_y, pred_c = test_model(model, x_linear)
    # accuracy(pred_y, pred_c, y_linear)

    # # Visualize the result
    show_result(x_xor, y_xor, pred_c)
    # show_result(x_linear, y_linear, pred_c)
