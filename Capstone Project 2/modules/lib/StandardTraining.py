import gc
from collections import namedtuple

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
                         net,
                         epoch_args='standard',
                         use_positivity_weights=True, 
                         image_width = 320,
                         image_height = 320,
                         affineDegrees=5, 
                         translatePrecent=0.05, 
                         shearDegrees=5, 
                         brightnessJitter=0.2, 
                         contrastJitter=0.1, 
                         augPercent=0.2,
                         observation_min_count = None):
        
        self.number_images = number_images
        self.batch_size=batch_size
        self.learning_rate = learning_rate
        self.val_percent=0.20
        self.device = device
        self.net = net
        
        self.num_epochs = num_epochs
        

        self.loaders = Loaders(  image_width = image_width,
                                 image_height = image_height,
                                 affineDegrees=affineDegrees, 
                                 translatePrecent=translatePrecent, 
                                 shearDegrees=shearDegrees, 
                                 brightnessJitter=brightnessJitter, 
                                 contrastJitter=contrastJitter, 
                                 augPercent=augPercent,
                                 observation_min_count=observation_min_count)

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
        
        if use_positivity_weights:
            df_positivity = self.train_actual[self.target_columns]
            positivity_weights = df_positivity[df_positivity!=1].count(axis=0) 
            positivity_weights = positivity_weights / df_positivity.sum(axis=0)
            print("\n\Positive Weights used in BCEWithLogitsLoss:")
            display(positivity_weights)
            pos_weight = torch.tensor(positivity_weights.values)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        
        self.metrics = Metrics(self.target_columns, self.train_actual, self.val_actual, cc=0)
        self.trainingLoop = TrainingLoop(self.device, self.net, self.optimizer, self.criterion, self.metrics)
        
        if epoch_args=='standard':
            self.trainingLoop.epoch_metric_display_args = (0, 
                                                  False, 
                                                  True, 
                                                  False, 
                                                  False,  
                                                  ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural_Effusion'])
        elif epoch_args is not None:
            self.trainingLoop.epoch_metric_display_args = epoch_args
        else:
            self.trainingLoop.epoch_metric_display_args = None
            
            
    def train(self):
        self.trainingLoop.train(self.num_epochs, self.train_loader, self.val_loader)
        
    def displayMetrics(self):
        self.metrics.displayMetrics()
                
class ModelLoop():
    
    """
    docstring
    """
    
    def getConfigObject(name, net, learning_rate=None, batch_size=None, observation_min_count=None):
        config = namedtuple("NetConfig", "name net learning_rate batch_size observation_min_count")
        return config(name, net, learning_rate, batch_size, observation_min_count)
        
    
    def __init__(self,   number_images, 
                         default_batch_size, 
                         default_learning_rate, 
                         num_epochs,
                         device, 
                         nets):  
        

        
        self.number_images = number_images
        self.default_batch_size=default_batch_size
        self.default_learning_rate = default_learning_rate
        self.val_percent=0.20
        self.device = device
     
        self.num_epochs = num_epochs
        
        self.epoch_metric_display_args = (2, 
                                          False, 
                                          True, 
                                          False, 
                                          False,  
                                          ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural_Effusion'])
        

        self.loaders = Loaders()
        
        self.target_columns = self.loaders.target_columns
        
        trainer = namedtuple("Trainer", "name train_actual val_actual train_loader val_loader metrics trainingLoop")
        self.trainers = []
        
        for i, config in enumerate(nets):
            if i >= len(nets): 
                    break
            name, net, lr, bs = config.name, config.net, config.learning_rate, config.batch_size
            
            if lr==0:
                lr = self.default_batch_size
                
            if bs==0:
                bs = self.default_batch_size
            
            train_loader, val_loader = self.loaders.getDataTrainValidateLoaders(batch_size=bs,
                                                                                val_percent=self.val_percent, 
                                                                                n_random_rows=self.number_images)
            
            train_actual = self.loaders.train_df
            val_actual = self.loaders.val_df
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(net.parameters(), lr=lr)   
            
            metrics = Metrics(self.target_columns, train_actual, val_actual, cc=0)
            trainingLoop = TrainingLoop(self.device, net, optimizer, criterion, metrics)
            
            trainer = (name, train_actual, val_actual, train_loader, val_loader, metrics, trainingLoop)
        
            self.trainers.append(trainer)

    def train(self):
        for trainer in self.trainers:
            gc.collect
            name, train_actual, val_actual, train_loader, val_loader, metrics, trainingLoop = trainer
            print(u"\u2586" * 30 + '\n')
            print(name + '\n')
            print(u"\u2585" * 30)
        
            print(f'Number of Training Images: {len(train_actual):,}')
            print(f'Number of Validation Images: {len(val_actual):,}')
        
            trainingLoop.train(self.num_epochs, train_loader, val_loader)
            metrics.displayMetrics(*self.epoch_metric_display_args)
            
            del trainingLoop
            del metrics
            
            print('\n' * 5)



