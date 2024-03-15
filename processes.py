from train_eval import trainNN, validateNN, train_and_validateNN, testNN
from Models.BaseCNN import BaseCNN
from earlystop import EarlyStopper
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def initModel(activation, pooling, img_size, device, elu_val = 1, lrelu_val = .01):
    model = BaseCNN(activation, pooling, img_size, elu_val = 1, lrelu_val = 0.01)
    model= model.to(device)
    return model

def trainModel(model, activation, pooling, img_size, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    early_stopper = EarlyStopper(patience=2, min_delta=3)
    model_train_loss, model_valid_loss = train_and_validateNN(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, print_freq=100,
                                                                        early_stopper=early_stopper)
    torch.save(model.state_dict(), f'{activation}-{pooling}.pt')

    return model_train_loss, model_valid_loss

def plotTrain(model_train_loss, model_valid_loss):
    plt.figure()
    plt.plot(model_train_loss, c = 'r', label = "Training Loss")
    plt.plot(model_valid_loss, c = 'b', label = "Validation Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    

def loadModel(activation, pooling, img_size, test_loader, classes, device):

    model = BaseCNN(activation, pooling, img_size, elu_val = 1, lrelu_val = 0.01)
    model.load_state_dict(torch.load(f'{activation}-{pooling}.pt'))
    model = model.to(device)

    _ , _ = testNN(model, test_loader, classes, device)