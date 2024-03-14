class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.1):
        self.patience = patience # Number of epochs to wait since last validation loss improvement
        self.min_delta = min_delta # Minimum change in the validation loss to qualify as an improvement.
        self.counter = 0 # Counts epochs since the last improvement
        self.min_validation_loss = float('inf')

    def stop(self, validation_loss):
        # Better validation_loss, reset counter
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        # Validation loss didn't improve
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False