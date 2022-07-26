# -*- coding: utf-8 -*-
"""
2022 Summer DL lab3 created by Pei-Hsuan Tsai.
    Analysis diabetic retinopathy in the following three steps:
    1. Write your own custom DataLoader.
    2. Classify diabetic retinopathy grading via the ResNet architecture.
    3. Calculate the confusion matrix to evaluate the performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from dataloader import RetinopathyLoader
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

# Check the GPU is avialible, else use the CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class ResNet18(nn.Module):
    'ResNet18 model'
    def __init__(self, use_pretrained=True):
        # model layer
        super(ResNet18, self).__init__()
        self.model_ft = models.resnet18(pretrained=use_pretrained)
        # Only want to update the parameters for the layer(s) we are reshaping.
        # Therefore, we do not need to compute the gradients of the parameters that we are not changing
        if use_pretrained == True:
            for param in self.model_ft.parameters():
                param.requires_grad = False
        # Initialize the new layer and by default the new parameters have .requires_grad=True
        # Change the input and output of the last layer
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(in_features=num_ftrs, out_features=5)
        
    def forward(self, x):
        # model structure
        x = self.model_ft(x)
        return x


class ResNet50(nn.Module):
    'ResNet50 model'
    def __init__(self, use_pretrained=True):
        # model layer
        super(ResNet50, self).__init__()
        self.model_ft = models.resnet50(pretrained=use_pretrained)
        # Only want to update the parameters for the layer(s) we are reshaping.
        # Therefore, we do not need to compute the gradients of the parameters that we are not changing
        if use_pretrained == True:
            for param in self.model_ft.parameters():
                param.requires_grad = False
        # Initialize the new layer and by default the new parameters have .requires_grad=True
        # Change the input and output of the last layer
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(in_features=num_ftrs, out_features=5)
        
    def forward(self, x):
        # model structure
        x = self.model_ft(x)
        return x


def Train(model, train_loader, optimizer, loss_function):
    'Train the model.\nOutput : the loss and accuracy'
    model.train() # training mode
    num = len(train_loader.dataset)
    total_loss = 0
    correct = 0
    for x, y in tqdm(train_loader):
        # put data into gpu(cpu) enviroment
        image = x.to(device, dtype=torch.float32)
        label = y.to(device, dtype=torch.long)
        # initial optimizer, clear gradient
        optimizer.zero_grad()
        # put data into the model to do forward propagation
        output = model(image)
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


def Test(model, test_loader):
    'Test the model.\nOutput : the accuracy and prediction'
    model.eval()  # evaluate mode
    num = len(test_loader.dataset)
    correct = 0
    batch_pred = []
    for x, y in tqdm(test_loader):
        # put data into gpu(cpu) enviroment
        image = x.to(device, dtype=torch.float32)
        label = y.to(device, dtype=torch.long)
        # put data into the model to predict
        pred = model(image)
        pred_result = pred.argmax(dim=1).cpu()
        # argmax to find the predict class with max value and compare to the truth
        correct += (label.cpu() == pred_result).sum()
        # save the prediction
        batch_pred.append(pred_result)
    return correct / num, batch_pred


def TrainAndTest(model_name, train_loader, test_loader, Learning_rate, epoch, pretrain):
    'Train and test the model for each epoch.'
    train_losslog = []
    train_acclog = []
    test_acclog = []
    highest_acc = 0
    # Build the model
    if model_name == 'ResNet18':
        model = ResNet18(use_pretrained=pretrain).to(device)
        best_model = copy.deepcopy(model)
    elif model_name == 'ResNet50':
        model = ResNet50(use_pretrained=pretrain).to(device)
        best_model = copy.deepcopy(model)
    # Loss function and Optimizer
    Loss = nn.CrossEntropyLoss()
    Optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate, momentum=0.9, weight_decay=5e-4)
    # feature extraction of pretrain model
    if pretrain == True:
        for i in range(3):
            train_loss, train_acc = Train(model, train_loader, Optimizer, Loss)
            # test_acc, prediction = Test(model, test_loader)
        for param in model.parameters():
            param.requires_grad = True
    # Train and Test
    for i in range(epoch):
        train_loss, train_acc = Train(model, train_loader, Optimizer, Loss)
        test_acc, prediction = Test(model, test_loader)
        print(f'Epoch {i}:Train: Loss = {train_loss:.4f}, Acc = {train_acc:.4f}; Test: Acc = {test_acc:.4f}')
        train_losslog.append(train_loss)
        train_acclog.append(train_acc)
        test_acclog.append(test_acc)
        # Find the hightest accuracy model parameters and copy it
        if test_acc > highest_acc:
            best_model = copy.deepcopy(model)
            highest_acc = test_acc
            best_pred = prediction
    # Save the best model
    torch.save(best_model.state_dict(), os.path.join('./model', f'{model_name}_{pretrain}_3.pt'))
    return train_losslog, train_acclog, test_acclog, highest_acc, best_pred


def Plot_Acc(model, epoch, train_acc, test_acc):
    'Plot the comparision figure.'
    plt.title(f"Result comparision ({model})", fontsize=15)
    plt.xlabel(f"Epochs")
    plt.ylabel(f"Accuracy(%)")
    # with pretrain
    plt.plot(range(epoch), train_acc[0], color='blue', marker='o', label='Train(with pretraining)')
    plt.plot(range(epoch), test_acc[0], color='brown', marker='o', label='Test(with pretraining)')
    # w/o pretrain
    plt.plot(range(epoch), train_acc[1], color='lime', label='Train(w/o pretraining)')
    plt.plot(range(epoch), test_acc[1], color='darkorange', label='Test(w/o pretraining)')
    plt.legend(loc='upper left')
    plt.show()
    plt.savefig(os.path.join("./result", f'comparision_{model}_3.jpg'))


def PlotMatrix(model, acc):
    'Plot the confusion matrix results of the testing.'
    label_len = 5
    fig, ax = plt.subplots()
    # Plot the heatmap
    im = ax.imshow(acc)    
    ax.set_title(f"Confusion matrix")
    ax.set_xlabel(f"Predicted label")
    ax.set_ylabel(f"True label")
    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(label_len), labels=[0, 1, 2, 3, 4])
    ax.set_yticks(np.arange(label_len), labels=[0, 1, 2, 3, 4])
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    for i in range(label_len):  # row
        for j in range(label_len):  #col
            text = ax.text(j, i, np.round(acc[i, j], 2), ha="center", va="center")
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()
    plt.savefig(os.path.join("./result", f"matrix_{model}_3.jpg"))


def ConfusionMatrix(model, acc, prediction, truth):
    'Calaulate the confusion matrix of accuracy and plot it.'
    # Print the best accuracy
    print(f"Accuracy of {model} model on test set:")
    print(f"with pretrain = {acc[0]}")
    print(f"w/o pretrain = {acc[1]}")
    # Calaulate the math rate of best model
    if acc[0] > acc[1]:
        pred = prediction[0]
    else:
        pred = prediction[1]
    math_rate = np.zeros((5, 5))
    batch_num = 0
    for img, label in truth:
        for num in range(len(label)):
            i = label[num]
            j = pred[batch_num][num]
            math_rate[int(i)][int(j)] += 1
        batch_num += 1
    math_rate /= np.sum(math_rate, axis=1).reshape(-1,1)
    # Plot
    PlotMatrix(model, math_rate)


if __name__ == "__main__":
    # Hyperparameters
    data_path = "./data"
    BATCH_SIZE = 16
    Learning_rate = 1e-3
    Epochs = [15, 10] # Resnet18 = 10, Resnet50 = 5
    
    train_acclog = []
    test_acclog = []
    highest_acc = []
    best_pred = []

    # Data preprocess. Prepare image data for learning
    train_dataset = RetinopathyLoader(data_path, "train")
    test_dataset = RetinopathyLoader(data_path, "test")

    train_loader = DataLoaderX(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoaderX(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ## ResNet18
    # Pretrain model
    pre18_train_losslog, pre18_train_acclog, pre18_test_acclog, pre18_highest_acc, pre18_best_pred = TrainAndTest('ResNet18', train_loader, test_loader, Learning_rate, Epochs[0], True)
    train_acclog.append(pre18_train_acclog)
    test_acclog.append(pre18_test_acclog)
    highest_acc.append(pre18_highest_acc)
    best_pred.append(pre18_best_pred)

    # No pretrain model
    no18_train_losslog, no18_train_acclog, no18_test_acclog, no18_highest_acc, no18_best_pred = TrainAndTest('ResNet18', train_loader, test_loader, Learning_rate, Epochs[0], False)
    train_acclog.append(no18_train_acclog)
    test_acclog.append(no18_test_acclog)
    highest_acc.append(no18_highest_acc)
    best_pred.append(no18_best_pred)

    # Comparision figure
    Plot_Acc('ResNet18', Epochs[0], train_acclog, test_acclog)
    # Confusion matrix for best model
    ConfusionMatrix('ResNet18', highest_acc, best_pred, test_loader)

    ## ResNet50
    # Pretrain model
    pre50_train_losslog, pre50_train_acclog, pre50_test_acclog, pre50_highest_acc, pre50_best_pred = TrainAndTest('ResNet50', train_loader, test_loader, Learning_rate, Epochs[1], True)
    train_acclog.append(pre50_train_acclog)
    test_acclog.append(pre50_test_acclog)
    highest_acc.append(pre50_highest_acc)
    best_pred.append(pre50_best_pred)

    # No pretrain model
    no50_train_losslog, no50_train_acclog, no50_test_acclog, no50_highest_acc, no50_best_pred = TrainAndTest('ResNet50', train_loader, test_loader, Learning_rate, Epochs[1], False)
    train_acclog.append(no50_train_acclog)
    test_acclog.append(no50_test_acclog)
    highest_acc.append(no50_highest_acc)
    best_pred.append(no50_best_pred)

    # Comparision figure
    Plot_Acc('ResNet50', Epochs[1], train_acclog, test_acclog)
    # Confusion matrix for best model
    ConfusionMatrix('ResNet50', highest_acc, best_pred, test_loader)

    # # Demo Part
    # # Testing the model, 'ResNet18', 'ResNet50'
    # model = ResNet18(use_pretrained=True).to(device)
    # model.load_state_dict(torch.load('./model/ResNet18_True_2.pt'))
    # acc, pre = Test(model, test_loader)
    # print(f'The highest accuracy model = {acc}')

    # ## Demo Part
    # acc = np.zeros(2)
    # # Testing the model, 'ResNet18', 'ResNet50'
    # model_wi = ResNet18(use_pretrained=True).to(device)
    # model_wi.load_state_dict(torch.load('./model/ResNet18_True_2.pt'))
    # acc[0], pre = Test(model_wi, test_loader)
    # model_wo = ResNet18(use_pretrained=False).to(device)
    # model_wo.load_state_dict(torch.load('./model/ResNet18_False_2.pt'))
    # acc[1], pre = Test(model_wo, test_loader)
    # # print(f'The highest accuracy model = {acc}')
    # print(f"The highest accuracy of ResNet18 model on test set:")
    # print(f"with pretrain = {acc[0]}")
    # print(f"w/o pretrain = {acc[1]}")