import sys
from datetime import datetime
# from tqdm import tqdm

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
        Validation will not touch in the optimizer.
        """

        # Convert output from loader
        ids, inputs, labels = self.parseLoaderData(data)

        if not is_validation:
            # zero the parameter gradients
            self.optimizer.zero_grad()

        # Get outputs from Model
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
        
        train_batch_count = len(train_loader)
        val_batch_count = len(val_loader)
        
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.epoch_loss = 0

            # Print epoch header
            print(u"\u2583" * 50 + f'\nEpoch [{epoch+1}/{num_epochs}]\n' + u"\u2594" * 50 + '\n' + u"\u2594" * 50)
            
            # Training
            self.net.train()
            #for i, data in enumerate(tqdm(train_loader), 0): # causes I/O timeouts on Google CoLab
            for i, data in enumerate(train_loader, 0):
                ids, inputs, labels, outputs = self.processBatch(data)
                self.metrics.appendEpochBatchData(ids, outputs)
                self.epoch_loss += self.backProp(outputs, labels)
                train_progress_value = i/train_batch_count
                if i>=train_batch_count: 
                    train_progress_value = 1.
                progress = f'{train_progress_value:.1%} training progress\r'
                sys.stdout.write(progress)
                sys.stdout.flush()

            self.metrics.closeEpoch(epoch)
            training_time_elapsed = datetime.now() - start_time
            start_time = datetime.now()
            self.losses_hx[epoch] = self.epoch_loss

            # Validation
            self.net.eval()
            with torch.no_grad():
              for i, data in enumerate(val_loader, 0):         
                    ids, inputs, labels, outputs = self.processBatch(data, is_validation=True)
                    self.metrics.appendEpochBatchData(ids, outputs, is_validation=True)
                    val_progress_Value = i/val_batch_count
                    if i>=val_batch_count: 
                        val_progress_Value = 1.
                    progress = f'{val_progress_Value:.1%} validation progress\r'
                    sys.stdout.write(progress)
                    sys.stdout.flush()

            self.metrics.closeEpoch(epoch, is_validation=True)
            validation_time_elapsed = datetime.now() - start_time


            # stdout Results
            print(f'Epoch Loss: {self.epoch_loss:.4f} \
        \nTime of Completion: {datetime.now()}  \
        \nTraining Duration: {training_time_elapsed}  \
        \nValidation Duration: {validation_time_elapsed}')

            # Show metrics display (optional)
            if self.epoch_metric_display_args is not None:
                self.metrics.displayMetrics(**self.epoch_metric_display_args)
            
            
            
            
            