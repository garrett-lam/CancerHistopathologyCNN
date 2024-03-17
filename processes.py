from train_eval import trainNN, validateNN, train_and_validateNN, testNN
from Models.BaseCNN import BaseCNN
from earlystop import EarlyStopper
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

def initModel(activation, pooling, img_size, device, elu_val = 1, lrelu_val = .01):
    model = BaseCNN(activation, pooling, img_size, elu_val = 1, lrelu_val = 0.01)
    model= model.to(device)
    return model

def trainModel(model, activation, pooling, img_size, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    early_stopper = EarlyStopper(patience=2, min_delta=3)
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

    

def loadModel(activation, pooling, img_size, test_loader, classes, device):

    model = BaseCNN(activation, pooling, img_size, elu_val = 1, lrelu_val = 0.01)
    model.load_state_dict(torch.load(f'{activation}-{pooling}.pt'))
    model = model.to(device)

    return testNN(model, test_loader, classes, device)

def confMtx(preds, labels, act, pool):
    preds = [item for sublist in preds for item in sublist.tolist()]
    count = [[0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0]]
    
    for j in range(5):
        predsdigit = [preds for preds, labels in zip(preds, labels) if labels == j]
        for i in range(len(predsdigit)):
            count[j][predsdigit[i]] += 1

    sns.heatmap(count, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['0', '1', '2', '3', '4'])
    plt.title(f'{act}{pool} Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()