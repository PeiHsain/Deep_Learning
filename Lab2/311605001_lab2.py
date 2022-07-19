# -*- coding: utf-8 -*-
"""
2022 Summer DL lab2 created by Pei-Hsuan Tsai.
    Implement simple EEG classification models which are EEGNet, DeepConvNet with BCI competition dataset.
    Additionally, you need to try different kinds of activation function including ReLU, Leaky ReLU, ELU.
"""

# In[]:
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataloader import read_bci_data
from sklearn.metrics import accuracy_score

# Check the GPU is avialible, else use the CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# In[]
class EEGNet(nn.Module):
    'EEGNet model'
    def __init__(self, activation='ELU'):
        # model architecture, input shape (1, 2, 750)
        super(EEGNet, self).__init__()
        # Activation function: 'ReLU', 'Leaky ReLU', 'ELU'
        if activation == 'ELU':
            self.activate = nn.ELU(alpha=1.0)
        elif activation == 'ReLU':
            self.activate = nn.ReLU()
        elif activation == 'Leaky ReLU':
            self.activate = nn.LeakyReLU(negative_slope=0.01)
        # First convolution
        self.firstConv = nn.Sequential(
            # (1, 2, 750) -> (16, 2, 750)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        # Depthwise convolution
        self.depthwiseConv = nn.Sequential(
            # (16, 2, 750) -> (32, 1, 187)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activate,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )
        # Separable convolution
        self.separableConv = nn.Sequential(
            # (32, 1, 187) -> (32, 1, 23)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            self.activate,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )
        # Classification
        self.classity = nn.Sequential(
            # 736 = 32 * 1 * 23
            nn.Flatten(),
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, x):
        # model forward pass
        x = self.firstConv(x) # First convolution
        x = self.depthwiseConv(x) # Depthwise convolution
        x = self.separableConv(x) # Separable convolution
        x = self.classity(x) # Classification
        return x

# In[]

class DeepConvNet(nn.Module):
    'DeepConvNet model'
    def __init__(self, activation='ELU'):
        # model architecture, input shape (1, 2, 750)
        super(DeepConvNet, self).__init__()
        # Activation function: 'ReLU', 'Leaky ReLU', 'ELU'
        if activation == 'ELU':
            self.activate = nn.ELU(alpha=1.0)
        elif activation == 'ReLU':
            self.activate = nn.ReLU()
        elif activation == 'Leaky ReLU':
            self.activate = nn.LeakyReLU(negative_slope=0.01)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2)) # Maxpool
        self.dropout = nn.Dropout2d(p=0.5) # Dropout
        # (1, 2, 750) -> (25, 2, 746)
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 5))
        # (25, 2, 746) -> (25, 1, 373)
        self.Conv2 = self.conv(in_channels=25, out_channels=25, kernel_size=(2, 1), num_features=25)
        # (25, 1, 373) -> (50, 1, 184)
        self.Conv3 = self.conv(in_channels=25, out_channels=50, kernel_size=(1, 5), num_features=50)
        # (50, 1, 184) -> (100, 1, 90)
        self.Conv4 = self.conv(in_channels=50, out_channels=100, kernel_size=(1, 5), num_features=100)
        # (100, 1, 90) -> (200, 1, 43)
        self.Conv5 = self.conv(in_channels=100, out_channels=200, kernel_size=(1, 5), num_features=200)
        # 200 * 1 * 43 = 8600
        self.Classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8600, out_features=2)
        )

    def conv(self, in_channels, out_channels, kernel_size, num_features):
        seq_modules = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=num_features, eps=1e-05, momentum=0.1),
            self.activate,
            self.pool,
            self.dropout
        )
        return seq_modules

    def forward(self, x):
        # model forward pass
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = self.Classify(x)
        return x


# In[]
def Train(model, train_loader, optimizer, loss_function, num):
    'Train the model.\nOutput : the loss and accuracy'
    model.train() # training mode
    total_loss = 0
    correct = 0
    for x, y in train_loader:
        # put data into gpu(cpu) enviroment
        signal = x.to(device, dtype=torch.float)
        label = y.to(device, dtype=torch.long)
        # initial optimizer, clear gradient
        optimizer.zero_grad()
        # train the model, forward -> backward -> update
        # put data into the model to do forward propagation
        output = model(signal)
        # calculate loss
        loss_value = loss_function(output, label)
        # use loss to do backward propagation and compute the gradient
        loss_value.backward()
        # do gradient descent
        optimizer.step()
        # loss value
        total_loss += loss_value.item()
        # correct prediction
        correct += (label == output.argmax(dim=1)).sum().item()
    return total_loss, correct / num


def Test(model, test_loader, num):
    'Test the model.\nOutput : the accuracy'
    model.eval()  # evaluate mode
    correct = 0
    for x, y in test_loader:
        # put data into gpu(cpu) enviroment
        image = x.to(device, dtype=torch.float)
        label = y.to(device, dtype=torch.long)
        # put data into the model to predict
        pred = model(image)
        # argmax to find the predict class with max value and compare to the truth
        correct += (label == pred.argmax(dim=1)).sum().item()
    return correct / num


def TrainAndTest(model_name, train_loader, test_loader, train_num, test_num, Learning_rate, epoch):
    'Train and test the model for each epoch with relu, leaky relu, and elu.'
    train_losslog = []
    train_acclog = []
    test_acclog = []
    best_acclog = []
    # Three activation function, 'ReLU', 'Leaky ReLU', 'ELU'
    activation_choice = ['ReLU', 'Leaky ReLU', 'ELU']
    for Activation in activation_choice:
        print(f'{Activation}')
        highest_acc = 0
        # Build the model
        if model_name == 'EEGNet':
            model = EEGNet(activation=Activation).to(device)
        elif model_name == 'DeepConvNet':
            model = EEGNet(activation=Activation).to(device)
        # Loss function and Optimizer
        Loss = nn.CrossEntropyLoss()
        Optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
        # Train and Test
        losslog_tmp = []
        acclog_train_tmp = []
        acclog_test_tmp = []
        for i in range(epoch):
            train_loss, train_acc = Train(model, train_loader, Optimizer, Loss, train_num)
            test_acc = Test(model, test_loader, test_num)
            print(f'Epoch {i}: Train: Loss = {train_loss:.4f}, Acc = {train_acc:.4f}; Test: Acc = {test_acc:.4f}')
            losslog_tmp.append(train_loss)
            acclog_train_tmp.append(train_acc)
            acclog_test_tmp.append(test_acc)
            # Find the hightest accuracy model parameters and copy it
            if test_acc > highest_acc:
                best_model = copy.deepcopy(model)
                highest_acc = test_acc
        # Save the results log
        train_losslog.append(losslog_tmp)
        train_acclog.append(acclog_train_tmp)
        test_acclog.append(acclog_test_tmp)
        best_acclog.append(highest_acc)
        # Save the best model parameters
        torch.save(best_model.state_dict(), f'./model/{model_name}_{Activation}_1.pt')
    return train_losslog, train_acclog, test_acclog, best_acclog


# In[]
def Plot_Acc(model, epoch, train_acc, test_acc):
    'Plot the comparision figure.'
    plt.title(f"Activation function comparision ({model})", fontsize=15)
    plt.xlabel(f"Epoch")
    plt.ylabel(f"Accuracy(%)")
    # ReLU
    plt.plot(range(epoch), train_acc[0], color='blue', label='relu_train')
    plt.plot(range(epoch), test_acc[0], color='cornflowerblue', label='relu_test')
    # Leaky ReLU
    plt.plot(range(epoch), train_acc[1], color='lime', label='leaky_relu_train')
    plt.plot(range(epoch), test_acc[1], color='green', label='leaky_relu_test')
    # ELU
    plt.plot(range(epoch), train_acc[2], color='red', label='elu_train')
    plt.plot(range(epoch), test_acc[2], color='brown', label='elu_test')
    plt.legend(loc='lower right')
    plt.show()


def Plot_Loss(model, epoch, train_loss):
    'Plot the comparision figure.'
    plt.title(f"Activation function comparision ({model}, Loss)", fontsize=15)
    plt.xlabel(f"Epoch")
    plt.ylabel(f"Loss")
    plt.plot(range(epoch), train_loss[0], color='darkorange', label='relu_loss')
    plt.plot(range(epoch), train_loss[1], color='lawngreen', label='leaky_relu_loss')
    plt.plot(range(epoch), train_loss[2], color='hotpink', label='elu_loss')
    plt.legend(loc='upper right')
    plt.show()


def Best_Acc(model, log):
    'Print the best accuracy of each model.'
    print('Best Accuracy:')
    print(f'{model} + ReLU = {log[0]:.4f}')
    print(f'{model} + LeakyReLU = {log[1]:.4f}')
    print(f'{model} + ELU = {log[2]:.4f}')


# In[]:
if __name__ == "__main__":
    # Hyperparameters
    Batch_size = 64
    Epochs = 300
    Learning_rate = 1e-2
    # Load the train and test data, 2 classes (right hand, left head)
    # 1080 data, 2 channels, 750 time points
    train_data, train_label, test_data, test_label = read_bci_data()
    train_num = len(train_label)
    test_num = len(train_label)
    # Data preprocess. Prepare image data for learning
    train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
    test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False)

# In[]
## EEG
# Training and testing
train_losslog, train_acclog, test_acclog, best_acclog = TrainAndTest('EEGNet', train_loader, test_loader, train_num, test_num, Learning_rate, Epochs)
# In[]
# Plot
Plot_Acc('EEGNet', Epochs, train_acclog, test_acclog)
Plot_Loss('EEGNet', Epochs, train_losslog)
Best_Acc('EEGNet', best_acclog)


# In[]
## DeepConv
# Training and testing
train_losslog, train_acclog, test_acclog, best_acclog = TrainAndTest('DeepConvNet', train_loader, test_loader, train_num, test_num, Learning_rate, Epochs)
# Plot
Plot_Acc('DeepConvNet', Epochs, train_acclog, test_acclog)
Plot_Loss('DeepConvNet', Epochs, train_losslog)
Best_Acc('DeepConvNet', best_acclog)


# In[]
## Demo Part
# Prepare testing data
Batch_size = 64
train_data, train_label, test_data, test_label = read_bci_data()
test_num = len(train_label)
test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False)
# Testing the model, 'EEGNet_ReLU', 'EEGNet_Leaky ReLU', 'EEGNet_ELU'
model = EEGNet('Leaky ReLU').to(device)
model.load_state_dict(torch.load('./model/EEGNet_Leaky ReLU.pt'))
acc = Test(model, test_loader, test_num)
print(f'The highest accuracy model = {acc}')

# %%
