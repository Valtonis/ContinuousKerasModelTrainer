
# coding: utf-8

# In[ ]:

import os

import pickle
import contextlib
from datetime import datetime

class ContinuousKerasModelTrainer:
    def __init__(self, model, trainInfiniteGenerator, testFiniteGenerator, storeFolder=None):
        self.__model = model
        
        self.__numberOfStepsBeforeTesting = 20
        
        self.__trainInfiniteGenerator = trainInfiniteGenerator
        self.__testFiniteGenerator = testFiniteGenerator
        
        self.__showModelOutputDuringTrainingSteps = True
        
        if storeFolder is None:
            dt = datetime.now()
            self.__storeFolder = "store_" + str(dt.microsecond)
        
    def train(self, numberOfEpochs=None):
        self.__continueTraining = True
        if numberOfEpochs is None:
            if self.__continueTraining == True:
                self._trainStep()
        else:
            i = 0
            while i < numberOfEpochs:
                self._trainStep()
                i += 1
                
    def stop(self):
        self.__continueTraining = False
            
    def _trainStep(self):
        if self.__showModelOutputDuringTrainingSteps == False:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                self.__model.fit_generator(self.__trainInfiniteGenerator, 
                                           steps_per_epoch=self.__numberOfStepsBeforeTesting, 
                                           epochs=1, 
                                           use_multiprocessing=True, 
                                           class_weight=self.__classWeight)
        else:
            self.__model.fit_generator(self.__trainInfiniteGenerator, 
                                       steps_per_epoch=self.__numberOfStepsBeforeTesting, 
                                       epochs=1, 
                                       use_multiprocessing=True, 
                                       class_weight=self.__classWeight)
        loss = self.__model.evaluate_generator(self.__testFiniteGenerator, use_multiprocessing=True)
        if self.__minimumLoss is None or loss < self.__minimumLoss:
            self.__minimumLoss = loss
            self._saveModel("bestModel")
            with open(os.path.join(self.__storeFolder, "bestModelLoss"), "w") as fp:
                fp.write("Best model loss: " + str(self.__minimumLoss))
    
    def getLossOnTrainData(self):
        if self.__trainFiniteGenerator is None:
            return None
        loss = self.__model.evaluate_generator(self.__trainFiniteGenerator, use_multiprocessing=True)
        return loss
    
    def getLossOnValidationData(self):
        if self.__trainFiniteGenerator is None:
            return None
        loss = self.__model.evaluate_generator(self.__validationFiniteGenerator, use_multiprocessing=True)
        return loss
        
    
    def save(self):
        with open(os.path.join(self.__storeFolder, 'data.pickle'), "wb") as fp:
            pickle.dump(self, fp)
        
    def _saveModel(self, modelName):
        self.__model.save(os.path.join(self.__storeFolder, modelName))
        
    @classmethod
    def load(cls, folderName):
        obj = None
        with open(os.path.join(folderName, 'data.pickle'), 'rb') as fp:
            obj = pickle.load(fp)
        return obj
        
    # Setters and Getters:
    
    def showModelOutputDuringTrainingSteps(self, val):
        if val == True:
            __showModelOutputDuringTrainingSteps = True
        else:
            __showModelOutputDuringTrainingSteps = False
            
    def setClassWeight(self, classWeight):
        self.__classWeight = classWeight
    
    def setNumberOfStepsBeforeTesting(self, number):
        self.__numberOfStepsBeforeTesting = number
        
    def setStoreFolder(self, folderName):
        self.__storeFolder = folderName
        
    # Fields:
    
    __model = None
    
    __classWeight = None
    
    __numberOfStepsBeforeTesting = None
    
    __minimumLoss = None
    
    __continueTraining = None
    
    __trainInfiniteGenerator = None
    __testFiniteGenerator = None
    
    __trainFiniteGenerator = None
    __validationFiniteGenerator = None
    
    __showModelOutputDuringTrainingSteps = None
    
    __storeFolder = None
    

