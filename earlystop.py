class EarlyStopper:
    def __init__(self, patience=1, min_delta=1):
        self.patience = patience # Number of epochs to wait for improvement before stopping.
        self.min_delta = min_delta # Minimum decrease in validation loss to constitute an improvement
        self.counter = 0 # Tracks epochs without improvement.
        self.min_validation_loss = float('inf')

    def stop(self, validation_loss):
         # Reset counter if validation loss improves.
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        # Increment counter if loss didn't improve based on min_delta
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False