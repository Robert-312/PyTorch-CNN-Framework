import gc
import os
import pandas as pd
from collections import namedtuple
import pickle 

from modules.lib.CheXpertData import *
from modules.lib.Metrics import *
from modules.lib.TrainingLoop import *

import torch
import torch.utils.data
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
                         use_positivity_weights=False, 
                         image_width = 320,
                         image_height = 320,
                         affineDegrees=5, 
                         translatePrecent=0.05, 
                         shearDegrees=5, 
                         brightnessJitter=0.2, 
                         contrastJitter=0.1, 
                         augPercent=0.2,
                         observation_min_count=None,
                         l2_reg=0,
                         loss_reduction='mean',
                         target_columns=None,
                         target_thresholds=None,
                         save_path=None,
                         net_name=None,
                         net_kwargs=None):
        
        target_columns_string = ''
        if target_columns is not None:
            target_columns_string = ','.join(target_columns)
        
        target_thresholds_string = ''
        if target_thresholds is not None:
            target_thresholds_string = ','.join([str(st) for st in target_thresholds])
            
        net_kwargs_string = ''
        if net_kwargs is not None:
            net_kwargs_string = ','.join([str(k) + ':' + str(v) for k,v in net_kwargs.items()])
        
        self.run_parameters = {'number_images':number_images, 
                           'batch_size':batch_size, 
                           'learning_rate':learning_rate, 
                           'num_epochs':num_epochs,
                           'epoch_args':epoch_args,
                           'use_positivity_weights':use_positivity_weights, 
                           'image_width':image_width,
                           'image_height':image_height,
                           'affineDegrees':affineDegrees, 
                           'translatePrecent':translatePrecent, 
                           'shearDegrees':shearDegrees, 
                           'brightnessJitter':brightnessJitter, 
                           'contrastJitter':contrastJitter, 
                           'augPercent':augPercent,
                           'observation_min_count':observation_min_count,
                           'l2_reg':l2_reg,
                           'loss_reduction':loss_reduction,
                           'target_columns':target_columns_string,
                           'target_thresholds':target_thresholds_string,
                           'net_name':net_name,
                           'net_kwargs':net_kwargs_string}
        
        self.save_path = save_path
        
        self.number_images = number_images
        self.batch_size=batch_size
        self.learning_rate = learning_rate
        self.val_percent=0.20
        self.device = device
        self.net = net
        
        self.num_epochs = num_epochs
        
        self.total_training_time_elapsed  = None
        

        self.loaders = Loaders(  image_width = image_width,
                                 image_height = image_height,
                                 affineDegrees=affineDegrees, 
                                 translatePrecent=translatePrecent, 
                                 shearDegrees=shearDegrees, 
                                 brightnessJitter=brightnessJitter, 
                                 contrastJitter=contrastJitter, 
                                 augPercent=augPercent,
                                 observation_min_count=observation_min_count,
                                 target_columns=target_columns)

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
                
            print("\nPositive Weights used in BCEWithLogitsLoss:")
            display(positivity_weights)

            pos_weight = torch.Tensor(positivity_weights.values)
            pos_weight = pos_weight.to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=loss_reduction)
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction=loss_reduction)
        
        self.optimizer = optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=l2_reg)
        
        self.metrics = Metrics(self.target_columns, self.train_actual, self.val_actual, target_thresholds=target_thresholds, cc=0)
        self.trainingLoop = TrainingLoop(self.device, self.net, self.optimizer, self.criterion, self.metrics)
        
        display_columns = target_columns.copy()
        if len(target_columns) == 12: #all targets
            display_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural_Effusion']
        
        if epoch_args=='standard':
            self.trainingLoop.epoch_metric_display_args = {
                                                              'metricDataSource':0, 
                                                              'showCombinedMetrics':False, 
                                                              'showMetricDataFrame':True, 
                                                              'showROCCurves':False, 
                                                              'showPrecisionRecallCurves':False,
                                                              'include_targets':display_columns
                                                          }
        elif epoch_args is not None:
            self.trainingLoop.epoch_metric_display_args = epoch_args
        else:
            self.trainingLoop.epoch_metric_display_args = None
            
            
    def train(self):
        start_time = datetime.now()
        self.trainingLoop.train(self.num_epochs, self.train_loader, self.val_loader)
        self.total_training_time_elapsed = datetime.now() - start_time
        print(f'Training Duration: {self.total_training_time_elapsed }')
        
    def displayMetrics(self):
        self.metrics.displayMetrics()
        
    def displayEpochProgression(self):
        self.metrics.displayEpochProgression(include_targets = display_columns)
        
    def saveRunParameters(self):
        path = os.path.join(self.save_path, 'RunParameters.pkl')
        with open(path, 'wb') as filehandler:
            pickle.dump(self.run_parameters, filehandler)        
        
    def saveMetrics(self):
        path = os.path.join(self.save_path, 'Metrics.pkl')
        with open(path, 'wb') as filehandler:
            pickle.dump(self.metrics, filehandler)
                
    def saveNet(self):
        path = os.path.join(self.save_path, 'Net.pkl')
        with open(path, 'wb') as filehandler:
            torch.save(self.net.state_dict(), filehandler)      
                
     
    def save(self):
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            try:
                self.saveRunParameters()
            except RuntimeError as err:
                print('Save Run Parameters Error: ', err)

            try:
                self.saveMetrics()
            except RuntimeError as err:
                print('Save Metric Error: ', err)

            try:
                self.saveNet()
            except RuntimeError as err:
                print('Save Network Error: ', err)
            
    def displayRunParameters(dir):
        if os.path.exists(dir):
            path = os.path.join(dir, 'RunParameters.pkl')
            with open(path, 'rb') as filehandler:
                unpickler = pickle.Unpickler(filehandler)
                run_parameters = unpickler.load()
                
                net_name = run_parameters.pop("net_name")
                net_kwargs = run_parameters.pop("net_kwargs")
                target_columns = run_parameters.pop("target_columns")

                print('Network Name: ', net_name)
                print('Nework Arguments: ', net_kwargs)
                
                display(pd.DataFrame(list(run_parameters.items()), columns=['Paramter', 'Value']))
                print('Targets: ', target_columns)
        else:
            print('Directory does not exist')
            
    def loadMetrics(dir):
        if os.path.exists(dir):
            path = os.path.join(dir, 'Metrics.pkl')
            with open(path, 'rb') as filehandler:
                unpickler = pickle.Unpickler(filehandler)
                metrics = unpickler.load()
                return metrics
        else:
            print('Directory does not exist')
                
class ModelLoop():
    
    """
    docstring
    """
    
    def getConfigObject(name, 
                        net, 
                        learning_rate=None, 
                        batch_size=None, 
                        use_positivity_weights=False, 
                        observation_min_count=None,
                        l2_reg=0):
        
        config = namedtuple("NetConfig", "name net learning_rate batch_size use_positivity_weights observation_min_count l2_reg")
        return config(name, net, learning_rate, batch_size, use_positivity_weights, observation_min_count, l2_reg)
        
    
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
            use_positivity_weights, observation_min_count = config.use_positivity_weights, config.observation_min_count
            l2_reg = config.l2_reg
            
            self.loaders.observation_min_count = observation_min_count
            
            if lr==0:
                lr = self.default_batch_size
                
            if bs==0:
                bs = self.default_batch_size
            
            train_loader, val_loader = self.loaders.getDataTrainValidateLoaders(batch_size=bs,
                                                                                val_percent=self.val_percent, 
                                                                                n_random_rows=self.number_images)
            
            train_actual = self.loaders.train_df
            val_actual = self.loaders.val_df
            
            if use_positivity_weights:
                df_positivity = train_actual[self.target_columns]
                positivity_weights = df_positivity[df_positivity!=1].count(axis=0) 
                positivity_weights = positivity_weights / df_positivity.sum(axis=0)
                
                print("\nPositive Weights used in BCEWithLogitsLoss:")
                display(positivity_weights)
                
                pos_weight = torch.tensor(positivity_weights.values)
                pos_weight = pos_weight.to(self.device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.BCEWithLogitsLoss()
                
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=l2_reg)   
            
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



