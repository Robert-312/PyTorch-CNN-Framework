import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

from enum import Enum

import torch
import torch.utils.data

class MetricDataSource(Enum):
    Both = 0
    TrainingOnly = 1
    ValidationOnly = 2

class Metrics():
    
    """
    docstring
    """
    
    def __init__(self, target_columns, train_actual, val_actual, target_thresholds=None, cc=10, scc=5):
        self.target_columns = target_columns
        self.train_actual = train_actual
        self.val_actual = val_actual
        
        if target_thresholds is None:
            #Default thresholds for all targets is .5
            self.target_thresholds = np.ones(len(self.target_columns)) * .5
        else:
            #Custom Thresholds
            #i.e. set Edema threshold to .2.  If Edema probability >= .2, prediction = 1
            self.target_thresholds = target_thresholds
            
        self.target_thresholds = torch.tensor(self.target_thresholds)
        
        self.cc = cc
        self.scc = scc
        
        self.train_prediction_hx = {}
        self.train_probability_hx = {}

        self.val_prediction_hx = {}
        self.val_probability_hx = {}

        self.epoch_train_predictions = {}
        self.df_train_prediction = None
        
        self.epoch_train_probabilities = {}
        self.df_train_probability = None

        self.epoch_val_predictions = {}
        self.df_val_prediction = None
        
        self.epoch_val_probabilities = {}
        self.df_val_probability = None
        
        
        
 # DATA
        
    def getPredictionsFromOutput(self, outputs):
        """
        We are using BCEWithLogitsLoss for our loss
        In this loss funciton, each label gets the sigmoid (inverse of Logit) before the CE loss
        So our model outputs the raw values on the last FC layer
        This means we have to apply sigmoid to our outputs to squash them between 0 and 1
        We then take values >= .5 as Positive and < .5 as Negative 
        """

        probabilities = torch.sigmoid(outputs.data) 
        predictions = probabilities.clone()
        
        #We need to make sure all tensors are in the same memory space
        if self.target_thresholds.device != predictions.device:
            self.target_thresholds = self.target_thresholds.to(predictions.device)
        
        predictions[predictions >= self.target_thresholds] = 1 # assign 1 label to those with gt or equal to threshold
        predictions[predictions < self.target_thresholds] = 0 # assign 0 label to those with less than threshold   
        
        return probabilities, predictions   
            
    def updateProbabilities(self, dictionary, ids, probabilities):
        """
        Our model outputs are raw scores.
        We want to be able to build ROC curves for some or all of our targets
        For the ROC curve, we need the actuals for each label along with the probability of the prediction.
        So like the predicitons, we want to keep track of the probabilities
        """
        for i in range(len(ids)):
            id = ids[i].item()    
            dictionary[id] = [float(f.item()) for f in probabilities[i]]     
    
    def updatePredictions(self, dictionary, ids, predictions):
        """
        Keep track of predictions using the same index as our DataFrame
        This will allow us to compare to the actual labels

        We only are taking the last prediction for each x-ray, but we could extend this later if wanted.
        """

        for i in range(len(ids)):
            id = ids[i].item()    
            dictionary[id] = [int(f.item()) for f in predictions[i]]          
            
    def appendEpochBatchData(self, ids, outputs, is_validation=False):
        """
        docstring
        """
        
        probabilities, predictions = self.getPredictionsFromOutput(outputs)        

        if is_validation:
            self.updateProbabilities(self.epoch_val_probabilities, ids, probabilities)
            self.updatePredictions(self.epoch_val_predictions, ids, predictions)
        else:
            self.updateProbabilities(self.epoch_train_probabilities, ids, probabilities)
            self.updatePredictions(self.epoch_train_predictions, ids, predictions)    
            
    def getPredictionDataFrame(self, epoch_predictions):
        result = pd.DataFrame(epoch_predictions).transpose().sort_index()
        result.columns = self.target_columns
        return result

    def getProbilityDataFrame(self, epoch_probabilities):
        result = pd.DataFrame(epoch_probabilities).transpose().sort_index()
        result.columns = self.target_columns
        return result
        
    def closeEpoch(self, epochNumber, is_validation=False):
        """
        docstring
        """
        
        if is_validation:
            self.df_val_prediction = self.getPredictionDataFrame(self.epoch_val_predictions)
            self.val_prediction_hx[epochNumber] = self.df_val_prediction.copy()
            self.epoch_val_predictions = {}

            self.df_val_probability = self.getProbilityDataFrame(self.epoch_val_probabilities)
            self.val_probability_hx[epochNumber] = self.df_val_probability.copy()
            self.epoch_val_probabilities = {}
        else:
            self.df_train_prediction = self.getPredictionDataFrame(self.epoch_train_predictions)
            self.train_prediction_hx[epochNumber] = self.df_train_prediction.copy()
            self.epoch_train_predictions = {}

            self.df_train_probability = self.getProbilityDataFrame(self.epoch_train_probabilities)
            self.train_probability_hx[epochNumber] = self.df_train_probability.copy()
            self.epoch_train_probabilities = {}
        
# DISPLAY        
        
    def displayCombinedMetrics(self, y_true, y_pred, average):
        """
        docstring
        """
        

        accuracy_score = metrics.accuracy_score(y_true=y_true, y_pred=y_pred, normalize=True)
        hamming_loss = metrics.hamming_loss(y_true=y_true, y_pred=y_pred)
        recall_score = metrics.recall_score(y_true=y_true, y_pred=y_pred, average=average, zero_division=0)
        precision_score = metrics.precision_score(y_true=y_true, y_pred=y_pred, average=average, zero_division=0)
        f1_score = metrics.f1_score(y_true=y_true, y_pred=y_pred, average=average, zero_division=0)


        df_combined = pd.DataFrame({
                                    'Accuracy Score':[accuracy_score],
                                    'Hamming Loss':[hamming_loss],
                                    'Combined Recall':[recall_score],
                                    'Combined Precision':[precision_score],
                                    'Combined F1':[f1_score]})

        df_combined = df_combined.transpose()
        df_combined.columns = ['Score for all Targets']

        display(df_combined) 
        
    def displayMetricDataFrame(self, y_true, y_pred, y_prob, include_targets=None):
        """
        docstring
        """
        
        target_columns = self.target_columns
        
        # Recall_score, precision_score and f1_score return the number of targets
        # Except if there is only one target, than it returns 2 values
        # So we have to take just the first dim if we are dealing with a single target!
        single_target=False
        if len(target_columns) == 1:
            single_target=True

        if include_targets is None:
            include_targets = target_columns

        true_positive_count = y_true.sum(axis=0)
        pred_positive_count = y_pred.sum(axis=0)

        nans = np.ones(len(target_columns))
        nans[:] = np.nan
        errors = {}

        try:
            itemized_recall = metrics.recall_score(y_true=y_true, y_pred=y_pred, average=None, zero_division=0)
            if single_target: itemized_recall=itemized_recall[0]
        except Exception as e:
            errors['Recall'] = e
            itemized_recall = nans


        try:    
            itemized_precision = metrics.precision_score(y_true=y_true, y_pred=y_pred, average=None, zero_division=0)
            if single_target: itemized_precision=itemized_precision[0]
        except Exception as e:
            errors['Precision'] = e
            itemized_precision = nans

        try:    
            itemized_f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred, average=None, zero_division=0)
            if single_target: itemized_f1=itemized_f1[0]
        except Exception as e:
            errors['F1'] = e
            itemized_f1 = nans

        try:    
            itemized_auc = metrics.roc_auc_score(y_true=y_true, y_score=y_prob, average=None)
        except Exception as e:
            errors['ROC AUC'] = e
            itemized_auc = nans

        try:    
            itemized_ap = metrics.average_precision_score(y_true=y_true, y_score=y_prob, average=None)
        except Exception as e:
            errors['Avg Precision'] = e
            itemized_ap = nans

        df_itemized = pd.DataFrame({'Target':target_columns, 
                                    'True Positives':true_positive_count, 
                                    'Predicted Positives':pred_positive_count, 
                                    'Recall':itemized_recall, 
                                    'Precision':itemized_precision, 
                                    'F1':itemized_f1, 
                                    'ROC AUC':itemized_auc, 
                                    'Avg Precision':itemized_ap})

        df_itemized = df_itemized[df_itemized.Target.isin(include_targets)]

        display(df_itemized)

        if len(errors) > 0:
            print(errors)
            
    def plotROC(self, target_columns, Y_true, Y_prob, include_targets=None, cols=2, height=4, width=14):
        """
        docstring
        """
        
        if include_targets is None:
            include_targets = target_columns

        subplot_count = len(include_targets)
        plt_cols = cols
        plt_rows = int(np.ceil(subplot_count / plt_cols))
        figure_height = plt_rows * height
        f = plt.figure(figsize=(width, figure_height))
        gs = f.add_gridspec(plt_rows, plt_cols)
        current_plot = 0

        # Build ROC curves one label at a time
        for i in range(len(target_columns)):
            target_name = target_columns[i]

            if target_name in include_targets:
                target_true = Y_true[:,i]
                target_probs = Y_prob[:,i]

                fpr, tpr, threshold = metrics.roc_curve(target_true, target_probs)
                roc_auc = metrics.auc(fpr, tpr)

                ax = f.add_subplot(gs[current_plot])
                ax.set_title(target_name + ' - ROC')
                ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
                ax.legend(loc = 'lower right')
                ax.plot([0, 1], [0, 1],'r--')
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')

                current_plot+=1

        f.tight_layout()
        plt.show()             
            
    def plotPrecisionRecall(self, target_columns, Y_true, Y_prob, include_targets=None, cols=2, height=4, width=14):
        """
        docstring
        """
        
        if include_targets is None:
            include_targets = target_columns

        subplot_count = len(include_targets)
        plt_cols = cols
        plt_rows = int(np.ceil(subplot_count / plt_cols))
        figure_height = plt_rows * height
        f = plt.figure(figsize=(width, figure_height))
        gs = f.add_gridspec(plt_rows, plt_cols)
        current_plot = 0

        # Build ROC curves one label at a time
        for i in range(len(target_columns)):
            target_name = target_columns[i]

            if target_name in include_targets:
                target_true = Y_true[:,i]
                target_probs = Y_prob[:,i]

                precision, recall, threshold = metrics.precision_recall_curve(target_true, target_probs)
                ap = metrics.average_precision_score(target_true, target_probs)

                ax = f.add_subplot(gs[current_plot])
                ax.set_title(target_name + ' - Precision/Recall')
                ax.plot(recall, precision, 'b', label = 'AP = %0.2f' % ap)
                ax.legend(loc = 'lower left')
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')

                current_plot+=1

        f.tight_layout()
        plt.show()             
            
    def displayMetrics(self, 
                       metricDataSource = MetricDataSource.Both, #train, val, both
                       showCombinedMetrics=True,
                       showMetricDataFrame=True,
                       showROCCurves=True,
                       showPrecisionRecallCurves=True,
                       include_targets=None,
                       combinedAverageMethod='samples',
                       gridSpecColumnCount=4,
                       gridSpecHeight=3,
                       gridSpecWidth=20):
        """
        docstring
        """
        

        target_columns = self.target_columns
        train_actual = self.train_actual
        val_actual = self.val_actual

        df_train_prediction = self.df_train_prediction
        df_train_probability = self.df_train_probability

        df_val_prediction = self.df_val_prediction
        df_val_probability = self.df_val_probability

        y_train_true = None
        y_train_pred = None
        y_train_prob = None

        y_val_true = None
        y_val_pred = None
        y_val_prob = None

        cc = self.cc # repeat character count
        scc = self.scc # short repeat character count

        if metricDataSource != MetricDataSource.ValidationOnly:
            print('=' * cc + '\nTRAINING\n' + '=' * cc)
            print(u"\u2594" * 5)
            y_train_true = train_actual[target_columns].to_numpy()
            y_train_pred = df_train_prediction.to_numpy()
            y_train_prob = df_train_probability.to_numpy()

            if showCombinedMetrics:
                self.displayCombinedMetrics(y_train_true, y_train_pred, average=combinedAverageMethod)

            if showMetricDataFrame:
                self.displayMetricDataFrame(y_train_true, y_train_pred, y_train_prob, include_targets=include_targets)

            if showROCCurves:
                print('*' * scc + ' ROC ' + '*' * scc)
                self.plotROC(target_columns, y_train_true, y_train_prob, include_targets=include_targets,
                        cols=gridSpecColumnCount, height=gridSpecHeight, width=gridSpecWidth)

            if showPrecisionRecallCurves:
                print('*' * scc + ' Precision / Recall ' + '*' * scc)
                self.plotPrecisionRecall(target_columns, y_train_true, y_train_prob, include_targets=include_targets, 
                        cols=gridSpecColumnCount, height=gridSpecHeight, width=gridSpecWidth)

        if metricDataSource != MetricDataSource.TrainingOnly:
            print('=' * cc + '\nVALIDATION\n' + '=' * cc)
            print(u"\u2594" * 7)
            y_val_true = val_actual[target_columns].to_numpy()
            y_val_pred = df_val_prediction.to_numpy()
            y_val_prob = df_val_probability.to_numpy()

            if showCombinedMetrics:
                self.displayCombinedMetrics(y_val_true, y_val_pred, average=combinedAverageMethod)

            if showMetricDataFrame:
                self.displayMetricDataFrame(y_val_true, y_val_pred, y_val_prob, include_targets=include_targets)

            if showROCCurves:
                print('*' * scc + ' ROC ' + '*' * scc)
                self.plotROC(target_columns, y_val_true, y_val_prob, include_targets=include_targets, 
                        cols=gridSpecColumnCount, height=gridSpecHeight, width=gridSpecWidth)

            if showPrecisionRecallCurves:
                print('*' * scc + ' Precision / Recall ' + '*' * scc)
                self.plotPrecisionRecall(target_columns, y_val_true, y_val_prob, include_targets=include_targets,
                        cols=gridSpecColumnCount, height=gridSpecHeight, width=gridSpecWidth)     
                
                
                

            
    def plotEpochAccuracy(self, train_pred_hx, val_pred_hx, 
                             train_y_true, val_y_true, 
                             include_targets=None, height=4, width=12):
        """
        docstring
        """
        
        target_columns = self.target_columns
        target_count = len(target_columns)
        
        if include_targets is None:
            include_targets = target_columns
            
        epochs = [e+1 for e in self.val_prediction_hx.keys()] #1 based for display
        
        train_epoch_score = {}
        val_epoch_score = {}
        for target_id in range(target_count):
            for epochNumber in epochs:
                #train
                train_epoch_pred = train_pred_hx[epochNumber-1].to_numpy()[:,target_id]
                target_train_y_true = train_y_true[:,target_id]
                if not epochNumber in train_epoch_score:
                    train_epoch_score[epochNumber] = []
                train_epoch_score[epochNumber].append(metrics.accuracy_score(y_true=target_train_y_true, y_pred=train_epoch_pred))
                #train
                val_epoch_pred = val_pred_hx[epochNumber-1].to_numpy()[:,target_id]
                target_val_y_true = val_y_true[:,target_id]
                if not epochNumber in val_epoch_score:
                    val_epoch_score[epochNumber] = []
                val_epoch_score[epochNumber].append(metrics.accuracy_score(y_true=target_val_y_true, y_pred=val_epoch_pred))

        df_train = pd.DataFrame(train_epoch_score)
        df_train.index = target_columns

        df_val = pd.DataFrame(val_epoch_score)
        df_val.index = target_columns

        f = plt.figure(figsize=(width, height))
        gs = f.add_gridspec(1, 2)
        for i in range(2):

            ax = f.add_subplot(gs[i])
            if i == 0:
                df_train.transpose()[include_targets].plot(ax=ax)
                ax.get_legend().remove()
                ax.set_title(f'Train')
            else:
                df_val.transpose()[include_targets].plot(ax=ax)
                ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
                ax.set_title(f'Validation')

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_xticks(epochs)
            
        plt.show()                  
                
                
    def plotEpochProgression(self, train_score_hx, val_score_hx, 
                             train_y_true, val_y_true, 
                             score_function, score_name, is_prob=False,
                             include_targets=None, height=4, width=12):
        """
        docstring
        """
        
        target_columns = self.target_columns
        
        if include_targets is None:
            include_targets = target_columns
            
        epochs = [e+1 for e in self.val_prediction_hx.keys()] #1 based for display 
        
        train_epoch_score = {}
        val_epoch_score = {}
        for epochNumber in epochs:
            if is_prob:
                #train
                train_epoch_prob = train_score_hx[epochNumber-1].to_numpy()
                train_epoch_score[epochNumber] = score_function(y_true=train_y_true, y_score=train_epoch_prob, 
                                                                average=None)
                #train
                val_epoch_prob = val_score_hx[epochNumber-1].to_numpy()
                val_epoch_score[epochNumber] = score_function(y_true=val_y_true, y_score=val_epoch_prob, 
                                                              average=None)
            else:
                #train
                train_epoch_pred = train_score_hx[epochNumber-1].to_numpy()
                train_epoch_score[epochNumber] = score_function(y_true=train_y_true, y_pred=train_epoch_pred, 
                                                                average=None, zero_division=0)
                #train
                val_epoch_pred = val_score_hx[epochNumber-1].to_numpy()
                val_epoch_score[epochNumber] = score_function(y_true=val_y_true, y_pred=val_epoch_pred, 
                                                              average=None, zero_division=0)

        df_train = pd.DataFrame(train_epoch_score)
        df_train.index = target_columns

        df_val = pd.DataFrame(val_epoch_score)
        df_val.index = target_columns

        f = plt.figure(figsize=(width, height))
        gs = f.add_gridspec(1, 2)
        for i in range(2):

            ax = f.add_subplot(gs[i])
            if i == 0:
                df_train.transpose()[include_targets].plot(ax=ax)
                ax.get_legend().remove()
                ax.set_title(f'Train')
            else:
                df_val.transpose()[include_targets].plot(ax=ax)
                ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
                ax.set_title(f'Validation')

            ax.set_xlabel('Epoch')
            ax.set_ylabel(score_name)
            ax.set_xticks(epochs)
            
        plt.show()  
            
    def displayEpochProgression(self, 
                                showResultDataFrames=False,
                                showAccuracyProgression=True,
                                showRecallProgression=True,
                                showPrecisionProgression=True,
                                showF1Progression=True,
                                showROCAUCProgression=True,
                                showAvgPrecisionProgression=True,
                                include_targets=None):
        """
        docstring
        """

        target_columns = self.target_columns
        train_actual = self.train_actual
        val_actual = self.val_actual

        df_train_prediction = self.df_train_prediction
        df_train_probability = self.df_train_probability

        df_val_prediction = self.df_val_prediction
        df_val_probability = self.df_val_probability

        y_train_true = None
        y_train_pred = None
        y_train_prob = None

        y_val_true = None
        y_val_pred = None
        y_val_prob = None

        cc = self.cc # repeat character count
        scc = self.scc # short repeat character count


        y_train_true = train_actual[target_columns].to_numpy()
        y_train_pred = df_train_prediction.to_numpy()
        y_train_prob = df_train_probability.to_numpy()

        y_val_true = val_actual[target_columns].to_numpy()
        y_val_pred = df_val_prediction.to_numpy()
        y_val_prob = df_val_probability.to_numpy()
                
        if showResultDataFrames:
            print('=' * cc + '\nTRAIN FINAL RESULTS\n' + '=' * cc)
            print(u"\u2594" * 8)
            display(train_actual[target_columns]) 
            display(df_train_prediction) 
            display(df_train_probability) 
            print('=' * cc + '\nVALIDATION FINAL RESULTS\n' + '=' * cc)
            print(u"\u2594" * 8)
            display(val_actual[target_columns]) 
            display(df_val_prediction) 
            display(df_val_probability) 
                
        if showAccuracyProgression:
            print('=' * cc + '\nACCURACY\n' + '=' * cc)
            print(u"\u2594" * 4)
            self.plotEpochAccuracy(self.train_prediction_hx, self.val_prediction_hx, 
                                      y_train_true, y_val_true, 
                                      include_targets=include_targets)   
                
        if showRecallProgression:
            print('=' * cc + '\nRECALL\n' + '=' * cc)
            print(u"\u2594" * 3)
            self.plotEpochProgression(self.train_prediction_hx, self.val_prediction_hx, 
                                      y_train_true, y_val_true, 
                                      metrics.recall_score, 'Recall',
                                      include_targets=include_targets)   
                
        if showPrecisionProgression:
            print('=' * cc + '\nPRECISION\n' + '=' * cc)
            print(u"\u2594" * 6)
            self.plotEpochProgression(self.train_prediction_hx, self.val_prediction_hx, 
                                      y_train_true, y_val_true, 
                                      metrics.precision_score, 'Precision',
                                      include_targets=include_targets)     
                
        if showF1Progression:
            print('=' * cc + '\nF1\n' + '=' * cc)
            print(u"\u2594" * 1)
            self.plotEpochProgression(self.train_prediction_hx, self.val_prediction_hx, 
                                      y_train_true, y_val_true, 
                                      metrics.f1_score, 'F1',
                                      include_targets=include_targets)       
                
        if showROCAUCProgression:
            print('=' * cc + '\nROC AUC\n' + '=' * cc)
            print(u"\u2594" * 4)
            self.plotEpochProgression(self.train_probability_hx, self.val_probability_hx, 
                                      y_train_true, y_val_true, 
                                      metrics.roc_auc_score, 'ROC AUC', is_prob=True,
                                      include_targets=include_targets)          
                
        if showAvgPrecisionProgression:
            print('=' * cc + '\nAVERAGE PRECISION\n' + '=' * cc)
            print(u"\u2594" * 8)
            self.plotEpochProgression(self.train_probability_hx, self.val_probability_hx, 
                                      y_train_true, y_val_true, 
                                      metrics.average_precision_score, 'Average Precision', is_prob=True,
                                      include_targets=include_targets)         





