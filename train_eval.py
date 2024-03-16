import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

def trainNN(network, train_loader, loss_function, optimizer, device, print_freq=50):
    network.train()  # Set the network to training mode
    mini_batch_loss = 0.0 # Initialize running loss within epoch
    total_loss = 0.0 # Initialize running loss for whole epoch
    total_batches = 0 

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # Move the inputs and labels to the specified device
        optimizer.zero_grad()  # Zero the parameter gradients

        outputs = network(inputs)  # Forward pass
        loss = loss_function(outputs, labels)
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        # Track losses
        mini_batch_loss += loss.item()
        total_loss += loss.item()
        total_batches += 1

        # Print average loss for every 'print_freq' mini-batches
        if i % print_freq == print_freq - 1:
            print(f'Mini-batch (i: {i+1}): Average mini-batch loss: {mini_batch_loss / print_freq:.3f}')
            mini_batch_loss = 0.0
            
    avg_train_loss = total_loss / total_batches
    return avg_train_loss

def validateNN(network, valid_loader, loss_function, device):
    network.eval() # Set the network to evaluation mode
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():  # No gradients needed for validation
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move to the specified device
            outputs = network(inputs)  # Forward step
            loss = loss_function(outputs, labels)  # Calculate loss

            total_loss += loss.item() 
            total_batches += 1
    
    avg_valid_loss = total_loss / total_batches
    return avg_valid_loss

def train_and_validateNN(network, train_loader, valid_loader, loss_function, optimizer, epochs, device, print_freq=50, early_stopper=None):
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        print(f'[Epoch {epoch+1}/{epochs}]')

        # Train for one epoch
        avg_train_loss = trainNN(network, train_loader, loss_function, optimizer, device, print_freq)
        train_losses.append(avg_train_loss)

        # Validate after training
        avg_valid_loss = validateNN(network, valid_loader, loss_function, device)
        valid_losses.append(avg_valid_loss)

        print(f'End of Epoch {epoch+1} - train loss: {avg_train_loss:.4f}, valid loss: {avg_valid_loss:.4f}')

        # Early Stopping Check
        if early_stopper and early_stopper.stop(avg_valid_loss):
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    print('Finished Training.')
    return train_losses, valid_losses

def testNN(network, test_loader, classes, device):
    network.eval()  # Set the network to evaluation mode
    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))

    with torch.no_grad():  # No gradients needed for testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = network(images) # Forward
            
            _, predicted = torch.max(outputs, 1) # Get the predictions from the maximum value
            correct = (predicted == labels).squeeze() # Compare predictions with the true label
            
            # Update lists 
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # Calculate accuracies
    class_accuracies = {}
    for i, class_name in enumerate(classes):
        class_accuracy = (class_correct[i] / class_total[i])*100
        class_accuracies[class_name] = class_accuracy
    overall_accuracy = (sum(class_correct) / sum(class_total))*100

    # Print accuracy for each class using the classes names
    for i in range(len(classes)):
        print(f'Accuracy for {classes[i]}: {(class_correct[i]/class_total[i])*100:.1f}%')
    print(f'Overall Model Accuracy: {sum(class_correct)/sum(class_total)*100:.1f}%')
    return class_accuracies, overall_accuracy