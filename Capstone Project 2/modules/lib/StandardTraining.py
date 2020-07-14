from modules.lib.ChextXRayImages import *
from modules.lib.Metrics import *
from modules.lib.TrainingLoop import *

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

class StandardTraining():
    
    """
    docstring
    """
    
    def __init__(self,   number_images, 
                         batch_size, 
                         learning_rate, 
                         num_epochs,
                         device, 
                         net):
        
        self.number_images = number_images
        self.batch_size=batch_size
        self.learning_rate = learning_rate
        self.val_percent=0.20
        self.device = device
        self.net = net
        
        self.num_epochs = num_epochs
        

        self.loaders = Loaders()
        self.train_loader = None
        self.val_loader = None
        self.train_loader, self.val_loader = self.loaders.getDataTrainValidateLoaders(batch_size=self.batch_size, 
                                                                                val_percent=self.val_percent, 
                                                                                n_random_rows=self.number_images)

        self.target_columns = self.loaders.target_columns
        self.train_actual = self.loaders.train_df
        self.val_actual = self.loaders.val_df
        
        print(f'Number of Training Images: {len(self.train_actual):,}')
        print(f'Number of Validation Images: {len(self.val_actual):,}')
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        
        self.metrics = Metrics(self.target_columns, self.train_actual, self.val_actual, cc=0)
        self.trainingLoop = TrainingLoop(self.device, self.net, self.optimizer, self.criterion, self.metrics)
        self.trainingLoop.epoch_metric_display_args = (0, 
                                                  False, 
                                                  True, 
                                                  False, 
                                                  False,  
                                                  ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural_Effusion'])

    def train(self):
        self.trainingLoop.train(self.num_epochs, self.train_loader, self.val_loader)
        
    def displayMetrics(self):
        self.metrics.displayMetrics()