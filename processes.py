from train_eval import trainNN, validateNN, train_and_validateNN, testNN
from Models.BaseCNN import BaseCNN
from Models.ResNet18 import ResNet18
from Models.EfficientNet_V2_S import EfficientNet_V2_S
from earlystop import EarlyStopper
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def initModel(activation, pooling, img_size, device, elu_val = 1, lrelu_val = .01):
    model = BaseCNN(activation, pooling, img_size, elu_val = 1, lrelu_val = 0.01)
    model= model.to(device)
    return model

def trainModel(model, activation, pooling, img_size, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    early_stopper = EarlyStopper(patience=2, min_delta=0.25)
    model_train_loss, model_valid_loss = train_and_validateNN(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, print_freq=100,
                                                                        early_stopper=early_stopper)
    if activation and pooling:
        torch.save(model.state_dict(), f'{activation}-{pooling}.pt')
    
    return model_train_loss, model_valid_loss

# For other CNNs
def saveModel(model, name):
    torch.save(model.state_dict(), f'{name}.pt')

def plotTrain(model_train_loss, model_valid_loss):
    plt.figure()
    plt.style.use("seaborn-v0_8")

    epochs = range(1, len(model_train_loss) + 1) # Start at 1
    plt.plot(epochs, model_train_loss, c = 'red', label = "Training")
    plt.plot(epochs, model_valid_loss, c = 'blue', label = "Validation")

    best_epoch = model_valid_loss.index(min(model_valid_loss))+1 
    best_val_loss = min(model_valid_loss)
    plt.plot(best_epoch, best_val_loss, 'bo')  
    plt.text(best_epoch, best_val_loss, f'  Best Epoch = {best_epoch}\n  Loss = {best_val_loss:.3f}', color='blue')
    
    plt.xlabel("Number of Epochs")
    plt.xticks(epochs)
    plt.ylabel("Average Epoch Loss")
    plt.title("Training and Validation Loss by Epoch")
    plt.legend()
    plt.show()

    

def loadModel(model_name, activation, pooling, img_size, test_loader, classes, device):
    if model_name == 'resnet18':
        model = ResNet18(len(classes))
        model.load_state_dict(torch.load(f'{model_name}.pt'))
        model = model.to(device)
    elif model_name == 'effnetV2_S':
        model = EfficientNet_V2_S(len(classes))
        model.load_state_dict(torch.load(f'{model_name}.pt'))
        model = model.to(device)
    else:
        model = BaseCNN(activation, pooling, img_size, elu_val = 1, lrelu_val = 0.01)
        model.load_state_dict(torch.load(f'{activation}-{pooling}.pt'))
        model = model.to(device)

    return testNN(model, test_loader, classes, device)

# Turns ypred in tensor form and concatenates into a list
def ypredToList(ypreds):
    return [item for tensor in ypreds for item in tensor.tolist()]

def confMtx(labels, ypred, model_name, classes):
    cm = confusion_matrix(labels, ypred)
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = classes, yticklabels=classes)
    plt.title(f'{model_name} Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()