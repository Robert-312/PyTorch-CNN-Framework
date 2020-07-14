from datetime import datetime
from tqdm import tqdm

import torch
import torch.utils.data


class TrainingLoop():
    
    """
    docstring
    """
    
    def __init__(self, device, net, optimizer, criterion, metrics):
        self.device = device
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        
        self.epoch_loss = 0
        self.losses_hx = {}
        
        self.epoch_metric_display_args = None
        
        
    def parseLoaderData(self, data):
        """
        The data loaders output a dictionary with 3 keys
        The first 2 keys hold single values for the ImageID and the actual tensor of the image
        The last key holds the ground truth vector of the 12 lables
        """ 

        ids, inputs, labels = data['id'], data['img'], data['labels']
        # move data to device GPU OR CPU
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        return ids, inputs, labels  
    
    def processBatch(self, data, is_validation=False):
        """
        Used for both training and validation.
        Validation will not pass in the optimizer.
        """

        # Convert output from loader
        ids, inputs, labels = self.parseLoaderData(data)

        if not is_validation:
            # zero the parameter gradients
            self.optimizer.zero_grad()

        # Convert output to predicitons
        outputs = self.net(inputs)

        return ids, inputs, labels, outputs    
    
    def backProp(self, outputs, labels):
        """
        Get loss value from criterion
        run backprop on the loss
        update weights in optimizer
        update epoch loss
        """

        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()    

    
    def train(self, num_epochs, train_loader, val_loader):
        """
        docstring
        """
        
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.epoch_loss = 0


            # Training
            self.net.train()
            for i, data in enumerate(tqdm(train_loader), 0):
                ids, inputs, labels, outputs = self.processBatch(data)
                self.metrics.appendEpochBatchData(ids, outputs)
                self.epoch_loss += self.backProp(outputs, labels)

            self.metrics.closeEpoch(epoch)
            training_time_elapsed = datetime.now() - start_time
            start_time = datetime.now()
            self.losses_hx[epoch] = self.epoch_loss

            # Validation
            self.net.eval()
            with torch.no_grad():
              for data in val_loader:          
                    ids, inputs, labels, outputs = self.processBatch(data, is_validation=True)
                    self.metrics.appendEpochBatchData(ids, outputs, is_validation=True)

            self.metrics.closeEpoch(epoch, is_validation=True)
            validation_time_elapsed = datetime.now() - start_time


            # stdout Results
            print('=' * 50 + f'\nEpoch [{epoch+1}/{num_epochs}]\n' + '=' * 50)
            print(f'Epoch Loss: {self.epoch_loss:.4f} \
        \nTraining Time: {training_time_elapsed})  \
        \nValidation Time: {validation_time_elapsed})')

            # Show metrics display (optional)
            if self.epoch_metric_display_args is not None:
                self.metrics.displayMetrics(*self.epoch_metric_display_args)
            
            
            
            
            